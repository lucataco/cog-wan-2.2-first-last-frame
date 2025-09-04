from typing import Optional, Tuple
import gc
import os
import time
import torch
import inspect
import tempfile
import subprocess
import numpy as np
from PIL import Image
import torch.nn.functional as F
try:
    # Try to call with enable_gqa to see if it's supported
    sig = inspect.signature(F.scaled_dot_product_attention)
    has_enable_gqa = "enable_gqa" in sig.parameters
except (ValueError, TypeError):
    # If we can't get signature (builtin function), assume it doesn't have enable_gqa
    has_enable_gqa = False

if not has_enable_gqa:
    _orig_sdpa = F.scaled_dot_product_attention
    def _sdpa_compat(query, key, value, *args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(query, key, value, *args, **kwargs)
    F.scaled_dot_product_attention = _sdpa_compat

from cog import BasePredictor, Input, Path
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video

MODEL_ID = "Wan2.2-I2V-A14B-bf16-Diffusers"
MODEL_URL = "https://weights.replicate.delivery/default/cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Downloading weights...")
        if not os.path.exists(MODEL_ID):
            download_weights(MODEL_URL, MODEL_ID)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Constants mirrored from the Space UI
        self.fixed_fps = 16
        self.min_frames_model = 8
        self.max_frames_model = 81

        # Load pipeline with explicit transformer models (matching original Gradio app)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            transformer=WanTransformer3DModel.from_pretrained(
                MODEL_ID, 
                subfolder='transformer',
                torch_dtype=self.dtype,
                device_map='cuda' if torch.cuda.is_available() else 'cpu',
            ),
            transformer_2=WanTransformer3DModel.from_pretrained(
                MODEL_ID, 
                subfolder='transformer_2',
                torch_dtype=self.dtype,
                device_map='cuda' if torch.cuda.is_available() else 'cpu',
            ),
            torch_dtype=self.dtype,
        )

        # Scheduler per Space config
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, shift=8.0
        )

        # Move to device and apply light optimizations
        self.pipe.to(self.device)
        try:
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
        except Exception:
            pass

    def _compute_output_dims(self, start_img: Image.Image, end_img: Optional[Image.Image]) -> Tuple[int, int]:
        # Flexible dimension rules from Space
        max_dim = 832
        min_dim = 480
        multiple = 16
        square_size = 480

        width, height = start_img.size
        if end_img is not None:
            # Prefer aspect of the start image; end image will be resized to match later
            pass

        if width == height:
            return square_size, square_size

        new_w, new_h = float(width), float(height)
        # Clamp to within [min_dim, max_dim]
        if max(new_w, new_h) > max_dim:
            scale = max_dim / max(new_w, new_h)
            new_w *= scale
            new_h *= scale
        if min(new_w, new_h) < min_dim:
            scale = min_dim / min(new_w, new_h)
            new_w *= scale
            new_h *= scale

        final_w = int(round(new_w / multiple) * multiple)
        final_h = int(round(new_h / multiple) * multiple)
        final_w = max(min_dim, min(max_dim, final_w))
        final_h = max(min_dim, min(max_dim, final_h))
        return final_w, final_h

    def _resize_to(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        return image.resize(size, Image.Resampling.LANCZOS)

    def _duration_to_frames(self, duration_seconds: float) -> int:
        frames = int(round(self.fixed_fps * float(duration_seconds)))
        return int(np.clip(frames, self.min_frames_model, self.max_frames_model))

    def resize_and_crop_to_match(self, target_image, reference_image):
        ref_width, ref_height = reference_image.size
        target_width, target_height = target_image.size
        scale = max(ref_width / target_width, ref_height / target_height)
        new_width, new_height = int(target_width * scale), int(target_height * scale)
        resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
        return resized.crop((left, top, left + ref_width, top + ref_height))

    def predict(
        self,
        start_image: Path = Input(description="Start frame image (RGB)"),
        end_image: Path = Input(description="Optional end frame image (RGB)"),
        prompt: str = Input(description="Prompt describing the transition between the two images", default="animate"),
        negative_prompt: str = Input(description="Negative prompt", default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,过曝，"),
        duration_seconds: float = Input(description="Video duration in seconds (16 fps)", default=5.0, ge=0.5, le=10.0),
        num_inference_steps: int = Input(description="Inference steps", default=8, ge=1, le=30),
        guidance_scale: float = Input(description="Guidance scale - high noise", default=3.0, ge=0.0, le=10.0),
        guidance_scale_2: float = Input(description="Guidance scale - low noise", default=3.0, ge=0.0, le=10.0),
        shift: float = Input(description="Shift", default=8.0, ge=1.0, le=10.0),
        seed: int = Input(description="Random seed if <= 0", default=0),
    ) -> Path:
        """Run a single prediction to generate a video from start (and optional end) frames."""
        # Seed
        if seed <= 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        print("Using seed: ", seed)
        generator = torch.Generator(device=self.device)
        generator = generator.manual_seed(int(seed))

        # Load images
        start_img = Image.open(str(start_image)).convert("RGB")
        end_img = Image.open(str(end_image)).convert("RGB")
        end_img_pil = self.resize_and_crop_to_match(end_img, start_img)

        # Compute target dims and resize
        out_w, out_h = self._compute_output_dims(start_img, end_img)
        start_img = self._resize_to(start_img, (out_w, out_h))
        end_img_pil = self._resize_to(end_img_pil, (out_w, out_h))

        # Frames based on duration and fps, clamped to model limits
        num_frames = self._duration_to_frames(duration_seconds)

        # Scheduler per Space config
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, shift=shift
        )

        # Prepare pipeline kwargs - standard image-to-video call
        pipe_kwargs = dict(
            prompt=prompt if prompt else None,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=start_img,
            last_image=end_img_pil,
            num_frames=num_frames,
            height=out_h,
            width=out_w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            generator=generator,
            output_type="pil",
        )

        # Remove None values
        pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}
        
        # Inference
        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            result = self.pipe(**pipe_kwargs)
        
        # Extract frames from the result
        frames = result.frames[0]
        
        # Export to video
        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "output.mp4")
        export_to_video(frames, out_path, fps=self.fixed_fps)

        # Clear cache between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return Path(out_path)
