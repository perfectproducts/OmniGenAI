from .Hunyuan3D_2.hy3dgen.rembg import BackgroundRemover
from .Hunyuan3D_2.hy3dgen.shapegen import (Hunyuan3DDiTFlowMatchingPipeline,
                                           FaceReducer,
                                           FloaterRemover,
                                           DegenerateFaceRemover
                                           )
from .Hunyuan3D_2.hy3dgen.text2image import HunyuanDiTPipeline
from .Hunyuan3D_2.hy3dgen.texgen import Hunyuan3DPaintPipeline
from PIL import Image


class HunyuanWrapper:
    def __init__(self):
        self.model_path = 'tencent/Hunyuan3D-2'
        self.t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        self.i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path)
        self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(self.model_path)
        self.rembg = BackgroundRemover()
        self.face_reducer = FaceReducer()
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()

    def text_to_image(self, prompt='a car') -> Image:
        return self.t2i(prompt)

    def image_to_3d(self, image: Image, glb_output_path='i23d_demo.glb') -> None:
        image = self.rembg(image)
        mesh = self.i23d(image, num_inference_steps=30, mc_algo='mc')[0]
        mesh = self.floater_remover(mesh)
        mesh = self.degenerate_face_remover(mesh)
        mesh = self.face_reducer(mesh)
        mesh = self.pipeline_tex(mesh, image)
        mesh.export(glb_output_path)

    def text_to_3d(self, prompt='a car', glb_output_path='t2i_demo.glb') -> None:
        image = self.text_to_image(prompt)
        self.image_to_3d(image, glb_output_path)
