# GenAI Image To 3D Trellis [genai.image_to_3d.trellis]

This is an example of pure python Kit extension. It is intended to be copied and to serve as a template to create new ones.

SEE  
https://github.com/microsoft/TRELLIS/issues/3

git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS

this worked for conda:

conda create -n trellis python=3.10
conda activate trellis 

pip install torch==2.5.1 torchvision --index-url=https://download.pytorch.org/whl/cu124
pip install xformers==0.0.28.post3 --index-url=https://download.pytorch.org/whl/cu124
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers
# Install utils3d package from a specific commit
# This package provides 3D utilities and helper functions
# The commit hash (9a4eb15) is pinned to ensure consistent behavior
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl
# Install Nvidia Kaolin library, which provides 3D deep learning tools
# The -f flag specifies a custom package index URL that contains builds compatible with CUDA 12.4 and PyTorch 2.5.1
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

git clone https://github.com/NVlabs/nvdiffrast.git ./tmp/extensions/nvdiffrast
pip install ./tmp/extensions/nvdiffrast

git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ./tmp/extensions/diffoctreerast
pip install ./tmp/extensions/diffoctreerast

git clone https://github.com/autonomousvision/mip-splatting.git ./tmp/extensions/mip-splatting
pip install ./tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

cp -r ./extensions/vox2seq ./tmp/extensions/vox2seq
pip install ./tmp/extensions/vox2seq

pip install spconv-cu120
pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

set ATTN_BACKEND=flash-attn
set SPCONV_ALGO=native
python ./app.py