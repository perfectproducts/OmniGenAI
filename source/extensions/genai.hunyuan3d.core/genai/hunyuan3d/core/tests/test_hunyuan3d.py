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
from genai.hunyuan3d.core.hunyuan_wrapper import HunyuanWrapper
import omni.kit.test
import os
import shutil
from PIL import Image
import sys
import io

# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of the module
# will make it auto-discoverable by omni.kit.test
class Test(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        # delete output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        # redirect stderr to our own stream
        self.text_prompt = "an apple"

    # After running each test
    async def tearDown(self):
        pass

    # Any class derived from `omni.ext.IExt` in the top level module (defined in
    # `python.modules` of `extension.toml`) will be instantiated when the extension
    # gets enabled, and `on_startup(ext_id)` will be called. Later when the
    # extension gets disabled on_shutdown() is called.

    async def test_text_to_image(self):
        hunyuan_wrapper = HunyuanWrapper()

        image = hunyuan_wrapper.text_to_image(prompt=self.text_prompt)
        image.save(os.path.join(self.output_dir, "text_to_image_demo.png"))
        del hunyuan_wrapper

    async def test_text_to_3d(self):
        hunyuan_wrapper = HunyuanWrapper()
        glb_path = os.path.join(self.output_dir, "text_to_3d_demo.glb")
        hunyuan_wrapper.text_to_3d(prompt=self.text_prompt, glb_output_path=glb_path)
        del hunyuan_wrapper
        # check if the file exists
        self.assertTrue(os.path.exists(glb_path))


    async def test_image_to_3d(self):
        hunyuan_wrapper = HunyuanWrapper()
        image_path = os.path.join(self.data_dir, "image_to_3d_demo.png")
        image = Image.open(image_path)
        # resize image to 512x512
        image = image.resize((512, 512))
        glb_path = os.path.join(self.output_dir, "image_to_3d_demo.glb")
        print(f"Running image_to_3d with {glb_path}")
        print(f"Image size: {image.size}")
        hunyuan_wrapper.image_to_3d(image, glb_path)
        del hunyuan_wrapper
        # check if the file exists
        self.assertTrue(os.path.exists(glb_path))
