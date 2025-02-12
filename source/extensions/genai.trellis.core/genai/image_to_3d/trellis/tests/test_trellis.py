# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add support for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
# Import extension python module we are testing with absolute import path, as if we are an external user (other extension)
import genai.image_to_3d.trellis as trellis
from genai.image_to_3d.trellis import Generate3dFromImageTrellis
import omni.kit.test
import asyncio
import omni.kit.asset_converter as converter
import os
import shutil

# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of the module
# will make it auto-discoverable by omni.kit.test
class TrellisTest(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        # delete output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
       
        

    # After running each test
    async def tearDown(self):
        pass

    # Actual test, notice it is an "async" function, so "await" can be used if needed
    async def test_trellis_generate(self):
        # generate a 3d model from an image
        image_path = os.path.join(self.data_dir, "typical_building_building.png")        
        output_path = os.path.abspath(os.path.join(self.output_dir, "typical_building_building.usd"))
        (result, err) = omni.kit.commands.execute("Generate3dFromImageTrellis",
                                                  image_path=image_path,
                                                  asset_usd_path=output_path)
        self.assertTrue(result)
        # wait for the conversion to finish
        await asyncio.sleep(3)  # wait for the conversion to finish            
        self.assertTrue(os.path.exists(output_path))

    
    async def test_convert_glb_to_usd(self):
        glb_path = os.path.join(self.data_dir, "typical_building_building.glb")
        output_path = os.path.abspath(os.path.join(self.output_dir, "typical_building_building.usd"))
        cmd = Generate3dFromImageTrellis(image_path="", asset_usd_path="") # we don't need the image path for this test
        await cmd.convert(glb_path, output_path)        
        self.assertTrue(os.path.exists(output_path))
        

