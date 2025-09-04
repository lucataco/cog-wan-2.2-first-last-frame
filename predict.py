from typing import Optional
import gc
import os
import time
import torch
import tempfile
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video

MODEL_ID = "Wan2.2-I2V-A14B-bf16-Diffusers"
MODEL_URL = "https://weights.replicate.delivery/default/cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers/model.tar"

# HF Space constants for image processing
MAX_DIMENSION = 832
MIN_DIMENSION = 480
DIMENSION_MULTIPLE = 16
SQUARE_SIZE = 480


def download_weights(url: str, dest: str) -> None:
    """Download model weights using pget."""
    start = time.time()
    print(f"Downloading {url} to {dest}")
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print(f"Download completed in {time.time() - start:.2f}s")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model weights and initialize the pipeline."""
        print("Loading models exactly as HF Space...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        
        # Download and load weights if needed
        if not os.path.exists(MODEL_ID):
            download_weights(MODEL_URL, MODEL_ID)
            
        # Load pipeline components exactly as HF Space does
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        # Load and fuse Lightning LoRA adapters with HF Space configuration
        print("Loading/fusing Lightning LoRA adapters...")
        self.pipe.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            adapter_name="lightx2v"
        )
        kwargs_lora = {"load_into_transformer_2": True}
        self.pipe.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            adapter_name="lightx2v_2",
            **kwargs_lora
        )
        
        # Configure and fuse LoRA adapters with HF Space scale values
        self.pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
        self.pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3.0, components=["transformer"])
        self.pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1.0, components=["transformer_2"])
        self.pipe.unload_lora_weights()
        print("LoRA fusion completed successfully")
        
        # Model frame constraints
        self.min_frames_model = 5
        self.max_frames_model = 121
        self.fixed_fps = 16.0
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Setup complete!")

    def process_image_for_video(self, image: Image.Image) -> Image.Image:
        """
        Process image following HF Space logic exactly.
        This determines the final canvas size for the video.
        """
        width, height = image.size
        if width == height:
            return image.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
        
        original_width, original_height = width, height
        aspect_ratio = original_width / original_height
        
        # Calculate new dimensions based on aspect ratio
        if aspect_ratio > 1:  # Landscape
            new_width = min(original_width, MAX_DIMENSION)
            new_height = new_width / aspect_ratio
        else:  # Portrait
            new_height = min(original_height, MAX_DIMENSION)
            new_width = new_height * aspect_ratio
        
        # Scale if below minimum dimension
        if min(new_width, new_height) < MIN_DIMENSION:
            if new_width < new_height:
                scale = MIN_DIMENSION / new_width
            else:
                scale = MIN_DIMENSION / new_height
            new_width *= scale
            new_height *= scale
        
        # Round to multiple of DIMENSION_MULTIPLE
        final_width = int(round(new_width / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
        final_height = int(round(new_height / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
        
        # Ensure minimum dimensions
        final_width = max(final_width, MIN_DIMENSION if aspect_ratio < 1 else SQUARE_SIZE)
        final_height = max(final_height, MIN_DIMENSION if aspect_ratio > 1 else SQUARE_SIZE)
        
        return image.resize((final_width, final_height), Image.Resampling.LANCZOS)

    def resize_and_crop_to_match(self, target_image: Image.Image, reference_image: Image.Image) -> Image.Image:
        """
        Resize and crop target image to exactly match reference image dimensions.
        This ensures perfect frame alignment.
        """
        ref_width, ref_height = reference_image.size
        target_width, target_height = target_image.size
        
        # Calculate scale to cover the reference dimensions
        scale = max(ref_width / target_width, ref_height / target_height)
        new_width = int(target_width * scale)
        new_height = int(target_height * scale)
        
        # Resize and center crop
        resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = (new_width - ref_width) // 2
        top = (new_height - ref_height) // 2
        
        return resized.crop((left, top, left + ref_width, top + ref_height))

    def duration_to_frames(self, duration_seconds: float) -> int:
        """Convert duration to frame count, clamped to model limits."""
        frames = int(round(self.fixed_fps * duration_seconds))
        return max(self.min_frames_model, min(frames, self.max_frames_model))

    def predict(
        self,
        start_image: Path = Input(description="Start frame image"),
        end_image: Path = Input(description="End frame image"),
        prompt: str = Input(
            description="Prompt describing the transition between images",
            default="animate"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,过曝，"
        ),
        duration_seconds: float = Input(
            description="Video duration in seconds",
            default=2.063,
            ge=0.5,
            le=10.0
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            default=8,
            ge=1,
            le=30
        ),
        guidance_scale: float = Input(
            description="Guidance scale for high noise",
            default=1.0,
            ge=0.0,
            le=10.0
        ),
        guidance_scale_2: float = Input(
            description="Guidance scale for low noise",
            default=1.0,
            ge=0.0,
            le=10.0
        ),
        shift: float = Input(
            description="Scheduler shift parameter",
            default=8.0,
            ge=1.0,
            le=10.0
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducibility (None for random)", 
            default=None
        ),
    ) -> Path:
        """Generate video from start and end frame images."""
        
        print("Running prediction...")
        
        # Handle seed
        if seed is None:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
        print(f"Using seed: {seed}")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Load images
        start_img = Image.open(start_image).convert("RGB")
        end_img = Image.open(end_image).convert("RGB")
        
        # This prevents ghosting artifacts by ensuring perfect frame alignment
        processed_start = self.process_image_for_video(start_img)
        processed_end = self.resize_and_crop_to_match(end_img, processed_start)
        
        # Calculate video parameters
        num_frames = self.duration_to_frames(duration_seconds)
        target_height, target_width = processed_start.height, processed_start.width
        
        # Configure scheduler with shift parameter
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, shift=shift
        )
        
        # Generate video
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processed_start,
                last_image=processed_end,
                num_frames=num_frames,
                height=target_height,
                width=target_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                generator=generator,
                output_type="pil",
            )
        
        # Export video
        frames = result.frames[0]
        output_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
        export_to_video(frames, output_path, fps=self.fixed_fps)
        
        return Path(output_path)
