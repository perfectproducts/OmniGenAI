import torch
from PIL import Image
from enum import Enum
from diffusers import FluxPipeline
from transformers import pipeline
from huggingface_hub import hf_hub_download

import warnings


class FluxState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


class FluxWrapper:
    def __init__(self):
        self.state = FluxState.UNINITIALIZED
        self.pipeline = None
        
        # Check CUDA availability
        print(f"CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            # Get the current device
            current_device = torch.cuda.current_device()
            
            # Print device properties
            print(f"\nCurrent CUDA device: {torch.cuda.get_device_name(current_device)}")
            print(f"Device capability: {torch.cuda.get_device_capability(current_device)}")
            print(f"Total memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")            
        else:
            print("No GPU available - please to install torch with GPU support")

    def shutdown(self):
        print("shutdown...")        
        self.pipeline = None
        self.state = FluxState.UNINITIALIZED

    async def initialize(self):
        if self.state in [FluxState.INITIALIZING, FluxState.READY]:
            return self.state == FluxState.READY
        
        self.state = FluxState.INITIALIZING
        print("initializing...")
        
        try:
            warnings.filterwarnings("ignore", message="You set `add_prefix_space`.*")
        
            # Initialize Flux pipeline
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            )

            # Load and fuse LoRA weights
            self.pipeline.load_lora_weights(
                hf_hub_download(
                    "ByteDance/Hyper-SD",
                    "Hyper-FLUX.1-dev-8steps-lora.safetensors",                
                )
            )
            self.pipeline.fuse_lora(lora_scale=0.125)
            self.pipeline.to(device="cuda", dtype=torch.bfloat16)
            self.state = FluxState.READY
            return True
        except Exception as e:
            print(f"Error initializing Flux pipeline: {e}")
            self.state = FluxState.ERROR
            return False

    async def generate(self,
                       prompt,
                       height=1024,
                       width=1024,
                       inference_steps=8,
                       guidance_scale=3.5,
                       seed=None) -> Image.Image:
        if self.state != FluxState.READY:
            await self.initialize()
        if self.state != FluxState.READY:
            print("Flux pipeline not initialized")
            return None
                   
        formatted_prompt = f"wbgmsst, 3D, {prompt} ,white background"
        
        # Set seed if not provided
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()
        print("generating...")
        # Generate image
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_image = self.pipeline(
                prompt=[formatted_prompt],
                generator=torch.Generator().manual_seed(seed),
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                max_sequence_length=256
            ).images[0]
        
        # debug: save image to file
        print(f"generated_image_{seed}.png")
        #generated_image.save(f"generated_image_{seed}.png")
        return generated_image
    

