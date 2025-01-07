from enum import Enum
import torch
from .TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline   
from .TRELLIS.trellis.utils import postprocessing_utils
from PIL import Image
import io


class TrellisState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


_trellis_instance = None


class TrellisWrapper:
    def __init__(self):
        self.state = TrellisState.UNINITIALIZED
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
            self.initialize()
        else:
            print("No GPU available - please to install torch with GPU support")
            self.state = TrellisState.ERROR

    def initialize(self):
        if self.state in [TrellisState.INITIALIZING, TrellisState.READY]:
            return self.state == TrellisState.READY
                
        self.state = TrellisState.INITIALIZING
        print("initializing...")

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()
        print("cuda initialized")

        self.state = TrellisState.READY
        return True
    
    def shutdown(self):
        if self.state != TrellisState.READY:
            return
        # make sure we destroy the pipeline and all the models and free the GPU memory
        self.pipeline.cpu()
        # destroy the pipeline
        del self.pipeline
        self.pipeline = None
        self.state = TrellisState.UNINITIALIZED
        print("shutdown complete")
    
    def generate(self,
                       image: Image.Image,
                       ss_sampling_steps: int = 12,
                       ss_guidance_strength: float = 7.5,
                       slat_sampling_steps: int = 12,
                       slat_guidance_strength: float = 3.0,
                       seed: int = 1,
                       mesh_simplify: float = 0.95,
                       texture_size: int = 1024) -> bytes:
        
        if self.state != TrellisState.READY:
            self.initialize()
        if self.state != TrellisState.READY:
            print("Trellis pipeline not initialized")
            return None
            # Run the pipeline
       
        outputs = self.pipeline.run(
            image,
            seed=seed,
            formats=["mesh", "gaussian"],
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            }
        )
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify,
            texture_size=texture_size
        )    
        buffer = io.BytesIO()
        glb.export(buffer, file_type="glb")
        return buffer.getvalue()
    
def get_trellis_instance():
    global _trellis_instance
    if _trellis_instance is None:
        _trellis_instance = TrellisWrapper()
    return _trellis_instance
