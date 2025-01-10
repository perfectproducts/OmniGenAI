import torch
from PIL import Image
from enum import Enum
from diffusers import FluxPipeline
from transformers import pipeline
from huggingface_hub import hf_hub_download
import gc
import warnings


class FluxState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


_flux_instance = None

def dump_memory():
    # Confirm GPU memory is freed
    print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

def destroy_flux_instance():
    global _flux_instance
    print("destroy flux_instance...")
    dump_memory()
    if _flux_instance is not None:
        _flux_instance.shutdown()
        _flux_instance = None
        torch.cuda.empty_cache()
        gc.collect()
        dump_memory()

        print("flux_instance shutdown done")
    else:
        print("flux_instance is not initialized")

def get_flux_instance():
    """Get the singleton instance of FluxWrapper."""
    print("get_flux_instance...")
    global _flux_instance
    if _flux_instance is None:
        print("creating flux_instance...")
        _flux_instance = FluxWrapper()
    return _flux_instance


class FluxWrapper:
    def __init__(self):
        self.state = FluxState.UNINITIALIZED
        self.pipeline = None

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
            torch.cuda.empty_cache()
            gc.collect()
            print("pipeline shutdown done")

        else:
            print("pipeline is not initialized")
        self.state = FluxState.UNINITIALIZED


    def initialize(self):
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

    def generate(self,
                 prompt,
                 height=1024,
                 width=1024,
                 inference_steps=8,
                 guidance_scale=3.5,
                 seed=None) -> Image.Image:
        if self.state != FluxState.READY:
            self.initialize()
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
        #print(f"generated_image_{seed}.png")
        #generated_image.save(f"generated_image_{seed}.png")
        return generated_image
