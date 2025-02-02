from .Hunyuan3D_2.hy3dgen.rembg import BackgroundRemover
from .Hunyuan3D_2.hy3dgen.shapegen import (Hunyuan3DDiTFlowMatchingPipeline,
                                           FaceReducer,
                                           FloaterRemover,
                                           DegenerateFaceRemover
                                           )
from .Hunyuan3D_2.hy3dgen.text2image import HunyuanDiTPipeline
from .Hunyuan3D_2.hy3dgen.texgen import Hunyuan3DPaintPipeline



class HunyuanWrapper:
    def __init__(self):
        pass

    def text_to_3d(self, prompt='a car'):
        print("text_to_3d")
        rembg = BackgroundRemover()
        print("rembg")
        model_path = 'tencent/Hunyuan3D-2'
        print("model_path")
        t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        print("t2i")
        i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        print("i23d")
        pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        print("pipeline_tex")



        print("i23d")
        image = t2i(prompt)
        image = rembg(image)
        mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)
        mesh = pipeline_tex(mesh, image)
        mesh.export('t2i_demo.glb')
