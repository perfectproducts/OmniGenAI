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
from genai.hunyuan3d.core import HunyuanWrapper
import omni.kit.test
import os
import shutil


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


    # After running each test
    async def tearDown(self):
        pass


    async def test_text_to_image(self):
        hunyuan_wrapper = HunyuanWrapper()
        prompt = "award-winning artwork depicting a majestic phoenix rising from the ashes, surrounded by swirling flames and embers, with vibrant colors and dynamic composition, masterpiece, trending on artstation"
        image = hunyuan_wrapper.text_to_image(prompt=prompt)
        image.save(os.path.join(self.output_dir, "text_to_image_demo.png"))


    async def test_text_to_3d(self):
        hunyuan_wrapper = HunyuanWrapper()
        prompt = "award-winning artwork depicting a majestic phoenix rising from the ashes, surrounded by swirling flames and embers, with vibrant colors and dynamic composition, masterpiece, trending on artstation"
        hunyuan_wrapper.text_to_3d(prompt=prompt)
        pass
