import omni.kit.commands
from PIL import Image
from .flux_wrapper import get_flux_instance


class GenerateImageFlux(omni.kit.commands.Command):

    def __init__(self,
                 prompt,
                 asset_png_path,
                 height=1024,
                 width=1024,
                 inference_steps=8,
                 guidance_scale=3.5,
                 seed=None):
        self.prompt = prompt
        self.asset_png_path = asset_png_path
        self.height = height
        self.width = width
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

    def do(self):
        flux = get_flux_instance()
        try:
            image: Image.Image = flux.generate(self.prompt,
                                               self.height,
                                               self.width,
                                               self.inference_steps,
                                               self.guidance_scale,
                                               self.seed)
            image.save(self.asset_png_path)
        except Exception as e:
            print(f"Error generating image: {e}")
            return False
        return True
