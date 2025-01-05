# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext
#from .TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
#from .TRELLIS.trellis.utils import render_utils, postprocessing_utils
import imageio
from PIL import Image
import omni.kit.pipapi
import os
import shutil
import io
import base64
import git
import subprocess

# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    """This is a blank extension template."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.
    def check_packages(self):
        print("Checking packages")
        # Add Ninja installation check/install at the beginning
        try:
            import ninja
            print(f"ninja: {ninja.__version__}")
            # Download ninja-win.zip
            import urllib.request
            import zipfile
            import sys
            
            ninja_url = "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip"
            ninja_zip = "ninja-win.zip"
            
            # Download ninja zip file
            urllib.request.urlretrieve(ninja_url, ninja_zip)
            
            # Extract ninja.exe
            with zipfile.ZipFile(ninja_zip, 'r') as zip_ref:
                zip_ref.extractall()
            
            # Add ninja directory to PATH
            ninja_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
            if ninja_dir not in os.environ['PATH']:
                os.environ['PATH'] = ninja_dir + os.pathsep + os.environ['PATH']
            
            # Clean up zip file
            os.remove(ninja_zip)
            print("check output")            
            # Check ninja version
            r = subprocess.check_output('ninja --version'.split())
            print(f"ninja check: {r}")

            

            

        except ImportError:
            print("ninja not found, installing...")
            r = omni.kit.pipapi.install(
                package="ninja",
                surpress_output=False,
                ignore_cache=True,
                use_online_index=True,
            )
            print(f"ninja installed: {r}")

        try:
            import torch
            print(f"torch: {torch.__version__}")
        except ImportError:
            print("torch not found")
        
        try:
            import torchvision
            print(f"torchvision: {torchvision.__version__}")
        except ImportError:
            print("torchvision not found")
            
        try:
            import kaolin
            print(f"kaolin: {kaolin.__version__}")
        except ImportError:
            # pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
            print("kaolin not found, installing...")
            # uninstall existing kaolin
            r = omni.kit.pipapi.call_pip(
                args=["uninstall", "kaolin", "-v" , "-y"],
                surpress_output=False,                
            )
            print(f"kaolin uninstalled: {r}")
            print("installing kaolin...")
            r = omni.kit.pipapi.install(
                package="kaolin",
                surpress_output=False,
                ignore_cache=True,
                ignore_import_check=True,
                use_online_index=True,
                extra_args=[
                    "-f", "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html",
                    "--extra-index-url", "https://download.pytorch.org/whl/cu124",
                    "-v"
                ],
            )
            print(f"kaolin installed: {r}")
        
        try:
            import flash_attn
            print(f"flash_attn: {flash_attn.__version__}")
        except ImportError:
            print("flash_attn not found")
            r = omni.kit.pipapi.install(
                package="flash_attn",
                surpress_output=False,
                ignore_cache=True,
                ignore_import_check=True,
                use_online_index=True,
                extra_args=[
                    "-f", "https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl",
                    "-v"
                ],
            )
            print(f"flash_attn installed: {r}")

        try:
            import nvdiffrast
            print(f"nvdiffrast: {nvdiffrast.__version__}")
        except ImportError:
            print("git clone https://github.com/NVlabs/nvdiffrast.git ./tmp/extensions/nvdiffrast")
            
            repo_url = "https://github.com/NVlabs/nvdiffrast.git"
            target_path = "./tmp/extensions/nvdiffrast"
            print("nvdiffrast not found, installing...")
            # first clone the repo
            try:
                git.Repo.clone_from(repo_url, target_path)
            except Exception as e:
                print(f"Error cloning repo: {e}")
            # install the repo
            r = omni.kit.pipapi.call_pip(
                args=["install", "./tmp/extensions/nvdiffrast"])
            print(f"nvdiffrast installed: {r}")

        # 'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8',
        try:
            import utils3d
            torch = utils3d.torch
            print(f"torch: {torch}")
            
        except ImportError:
            #try:
            #    repo_url = "https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
            #    target_path = "./tmp/extensions/utils3d"
            #    git.Repo.clone_from(repo_url, target_path)
            #except Exception as e:
            #    print(f"Error cloning repo: {e}")
            #abs_path = os.path.abspath("./tmp/extensions/utils3d/utils3d")
            #print(f"utils3d path: {abs_path}")
            r = omni.kit.pipapi.call_pip(
                args=["install", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8#egg=utils3d"],
                surpress_output=False,                
            )
            print(f"utils3d installed: {r}")
            import utils3d
            print(f"utils3d: {utils3d}")
            print(f"utils3d.torch: {utils3d.torch}")

            
            

            
            

        #git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ./tmp/extensions/diffoctreerast
        #pip install ./tmp/extensions/diffoctreerast

        #git clone https://github.com/autonomousvision/mip-splatting.git ./tmp/extensions/mip-splatting
        #pip install ./tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

        #cp -r ./extensions/vox2seq ./tmp/extensions/vox2seq
        #pip install ./tmp/extensions/vox2seq


        from .TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
        print(f"TrellisImageTo3DPipeline: {TrellisImageTo3DPipeline}")

    def on_startup(self, _ext_id):
        self.check_packages()
        
        """This is called every time the extension is activated."""
        print("[genai.image_to_3d.trellis] Extension startup")     
        from .TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline   
        from .TRELLIS.trellis.utils import postprocessing_utils

        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        print(f"pipeline: {pipeline}")
        pipeline.cuda()
        print("cuda done")
        
        # Load an image
        img_path= "TRELLIS/assets/example_image/T.png"
        full_path = os.path.join(os.path.dirname(__file__), img_path)
        image = Image.open(full_path)

        # Run the pipeline
        ss_sampling_steps = 12
        ss_guidance_strength = 7.5
        slat_sampling_steps = 12
        slat_guidance_strength = 3.0
        outputs = pipeline.run(
            image,
            seed=1,
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
        print(f"outputs: {outputs}")
        # Generate GLB file
        mesh_simplify = True
        texture_size = 1024
        print("postprocessing_utils.to_glb")
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify,
            texture_size=texture_size
        )
        print(f"glb: {len(glb)}")

        # Save GLB to bytes buffer and convert to base64
        buffer = io.BytesIO()
        glb.export(buffer, file_type="glb")
        glb_base64 = base64.b64encode(buffer.getvalue()).decode()
        print(f"glb_base64: {len(glb_base64)}")
        # write the glb to a file
        with open("output.glb", "wb") as f:
            f.write(glb_base64.encode())
            print("glb written to file")

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.image_to_3d.trellis] Extension shutdown")
