#import torch
#from PIL import Image

from .hy3dgen.rembg import BackgroundRemover
from .hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from .hy3dgen.text2image import HunyuanDiTPipeline


# Functions and vars are available to other extensions as usual in python:


def test_image_to_3d(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('mesh.glb')

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        mesh = pipeline(mesh, image=image)
        mesh.export('texture.glb')
    except Exception as e:
        print(e)
        print('Please try to install requirements by following README.md')


def test_text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    model_path = 'tencent/Hunyuan3D-2'
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    image = t2i(prompt)
    image = rembg(image)
    mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('t2i_demo.glb')
