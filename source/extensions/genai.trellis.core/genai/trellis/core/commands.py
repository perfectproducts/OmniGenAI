import omni.kit.commands
from PIL import Image
from .trellis_wrapper import get_trellis_instance
import omni.kit.asset_converter as converter
import os
import asyncio
from pxr import Usd

class Generate3dFromImageTrellis(omni.kit.commands.Command):
    def __init__(self,
                 image_path: str,
                 asset_usd_path: str,
                 ss_sampling_steps: int = 12,
                 ss_guidance_strength: float = 7.5,
                 slat_sampling_steps: int = 12,
                 slat_guidance_strength: float = 3.0,
                 seed: int = 1,
                 mesh_simplify: float = 0.95,
                 texture_size: int = 1024):
        self.image_path = image_path
        self.ss_sampling_steps = ss_sampling_steps
        self.ss_guidance_strength = ss_guidance_strength
        self.slat_sampling_steps = slat_sampling_steps
        self.slat_guidance_strength = slat_guidance_strength
        self.seed = seed
        self.mesh_simplify = mesh_simplify
        self.texture_size = texture_size
        self.asset_usd_path = asset_usd_path    

    def progress_callback(self, current_step: int, total: int):
        # Show progress
        print(f"converting {current_step} of {total}")

    async def convert(self, input_asset_path, output_asset_path):
        task_manager = converter.get_instance()
        task = task_manager.create_converter_task(input_asset_path, output_asset_path, self.progress_callback)
        success = await task.wait_until_finished()
                
        if not success:
            detailed_status_code = task.get_status()
            detailed_status_error_string = task.get_error_message()
            print(f"Failed to convert asset: {detailed_status_error_string} {detailed_status_code}")
            return False
        print(f"Asset converted successfully: {output_asset_path}")
        return True
    
    def do(self) -> bool:
        trellis = get_trellis_instance()
        # start a future to generate the 3d model, as this is an async operation
        # for now just call the generate method directly
        if not os.path.exists(self.image_path):
            print(f"Image file not found: {self.image_path}")
            return False
        image = Image.open(self.image_path)
        print(f"Generating 3D model from image {self.image_path} size: {image.size}")
        # check if image is valid:
        if image.size == (0, 0):
            print("Image is invalid")
            return False
        glb_bytes = trellis.generate(image,
                                     self.ss_sampling_steps,
                                     self.ss_guidance_strength,
                                     self.slat_sampling_steps,
                                     self.slat_guidance_strength,
                                     self.seed, self.mesh_simplify,
                                     self.texture_size)
        
        if glb_bytes is None:
            print("Failed to generate 3d model")
            return False
        # glb path is the same as the asset_usd_path basename with the .glb extension  e.g. "model.usda" or "model.usd"  -> "model.glb"
        base, ext = os.path.splitext(self.asset_usd_path)
        glb_path = base + ".glb"
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        # convert the glb to usd
        # create a dummy usd file to stand in for the converted usd file
        if os.path.exists(self.asset_usd_path):
            os.remove(self.asset_usd_path)
        asyncio.ensure_future(self.convert(glb_path, self.asset_usd_path))
        return True
        

    