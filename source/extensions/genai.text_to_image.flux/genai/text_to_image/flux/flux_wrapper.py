import torch
from PIL import Image
from enum import Enum
from diffusers import FluxPipeline
from transformers import pipeline
from huggingface_hub import hf_hub_download
import gc
import warnings
import time
import subprocess
import re


class FluxWrapper:
    _instance = None

    def __init__(self):
        print("initializing flux wrapper...")

        if self._instance is not None:
            print("flux wrapper already initialized")
            return
        self.pipeline = None
        self.initialized = False
        self._initialize()
        # Check CUDA availability
        print(f"CUDA is available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("No GPU available - please to install torch with GPU support")        

    def shutdown(self):
        print("pipeline shutdown...")
        if self.pipeline is not None:
            # destroy the pipeline
            del self.pipeline
            self.pipeline = None
            # Force CUDA memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Run garbage collection multiple times
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)
            print("pipeline shutdown done")

        else:
            print("pipeline is not initialized")

    def _check_free_memory(self, needed_memory_gb:float):
        gpu_info = self.get_gpu_memory_info()
        if gpu_info is None:
            print("Unable to get GPU memory info - nvidia-smi not available")
            return False
        total_memory_gb = gpu_info['total_gb']
        reserved_memory_gb = gpu_info['used_gb']
        free_memory_gb = total_memory_gb - reserved_memory_gb  # free
        print(f"total memory: {total_memory_gb:.2f}GB, reserved: {reserved_memory_gb:.2f}GB, free: {free_memory_gb:.2f}GB")
        if free_memory_gb < needed_memory_gb:
            print(f"not enough free memory - please to install torch with GPU support")
            return False
        return True

    def _initialize(self):
        print("initializing...")
        if self.initialized:
            print("flux wrapper already initialized")
            return True
        if not self._check_free_memory(32.3):
            print("not enough free memory - please to install torch with GPU support")
            return False
        try:
            print("initializing flux pipeline...")
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
            self.initialized = True
            self.print_memory_status()
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
        if not self.initialized:
            if not self._initialize():
                print("flux wrapper is not initialized")
                return []

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
        print("generated image saved")
        return generated_image

    def print_memory_status(self):
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            print("\nGPU Memory Status:")
            print(f"  System-wide:")
            print(f"    Total:     {gpu_info['total_gb']:.2f}GB")
            print(f"    Used:      {gpu_info['used_gb']:.2f}GB")
            print(f"    Free:      {gpu_info['free_gb']:.2f}GB")
            print(f"  This Process:")
            print(f"    Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
            print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_gpu_memory_info(self):
        try:
            # Run nvidia-smi command
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'])
            total, used, free = map(int, result.decode('utf-8').strip().split(','))

            return {
                'total_mb': total,
                'used_mb': used,
                'free_mb': free,
                'free_gb': free / 1024,
                'used_gb': used / 1024,
                'total_gb': total / 1024
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Unable to get GPU memory info - nvidia-smi not available")
            return None
