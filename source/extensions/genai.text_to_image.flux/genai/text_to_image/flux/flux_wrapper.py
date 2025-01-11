import torch
from PIL import Image
from enum import Enum
from diffusers import FluxPipeline
from transformers import pipeline
from huggingface_hub import hf_hub_download
import gc
import warnings
import time


class FluxWrapper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("creating flux_instance...")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        print("initializing flux wrapper...")
        self.pipeline = None
        self._initialize()
        # Check CUDA availability
        print(f"CUDA is available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("No GPU available - please to install torch with GPU support")

    @classmethod
    def destroy_instance(cls):
        if cls._instance is not None:
            print("destroying flux_instance...")
            cls._instance.shutdown()
            print("cleanup")
            # Force CUDA memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Run garbage collection multiple times
            for _ in range(3):

                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)

            print("flux_instance shutdown done")
        else:
            print("flux_instance is not initialized")

    def shutdown(self):
        print("pipeline shutdown...")
        if self.pipeline is not None:
            # destroy the pipeline
            self.pipeline.shutdown()
            del self.pipeline
            self.pipeline = None
            print("pipeline shutdown done")
        else:
            print("pipeline is not initialized")

    def _initialize(self):
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
            return True
        except Exception as e:
            print(f"Error initializing Flux pipeline: {e}")
            return False

    def generate(self,
                 prompt,
                 height=1024,
                 width=1024,
                 inference_steps=8,
                 guidance_scale=3.5,
                 seed=None) -> Image.Image:

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
        #print(f"generated_image_{seed}.png")
        #generated_image.save(f"generated_image_{seed}.png")
        return generated_image
