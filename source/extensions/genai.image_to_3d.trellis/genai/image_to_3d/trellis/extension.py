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
import os
import io
from .check_packages import check_packages
from PIL import Image
import base64

# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
def test_trellis(prompt: str):
    
    print(f"[genai.image_to_3d.trellis] test_trellis was called with {prompt}")
    print("[genai.image_to_3d.trellis] Extension startup")     
    from .TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline   
    from .TRELLIS.trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    print(f"pipeline: {pipeline}")
    pipeline.cuda()
    print("cuda done")
    
    # Load an image
    #img_path= "TRELLIS/assets/example_image/T.png"
    img_path= "TRELLIS/assets/example_image/typical_misc_television.png"
    full_path = os.path.join(os.path.dirname(__file__), img_path)
    image = Image.open(full_path)
    # scale the image to 256x256
    
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
    mesh_simplify = 0.95
    texture_size = 1024
    print("postprocessing_utils.to_glb")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=mesh_simplify,
        texture_size=texture_size
    )    

    
    # Save GLB to bytes buffer and convert to base64
    
    
    buffer = io.BytesIO()
    glb.export(buffer, file_type="glb")
    with open("output.glb", "wb") as f:
        f.write(buffer.getvalue())
        print("glb written to file")
    #return buffer.getvalue()
    
    

class MyExtension(omni.ext.IExt):
    """This is a blank extension template."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.

    def on_startup(self, _ext_id):
        
        check_packages()
        
        """This is called every time the extension is activated."""
        

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.image_to_3d.trellis] Extension shutdown")

    

