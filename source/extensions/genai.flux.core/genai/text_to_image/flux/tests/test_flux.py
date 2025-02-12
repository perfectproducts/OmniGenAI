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

import omni.kit.test
from genai.text_to_image.flux import FluxWrapper
import sys
import io
import contextlib
import os
import shutil


# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of the module
# will make it auto-discoverable by omni.kit.test
class TestFlux(omni.kit.test.AsyncTestCase):
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
    async def test_flux_init(self):
        flux = FluxWrapper.get_instance()
        img = flux.generate("a red and green apple", height=512, width=512)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (512, 512))

    async def test_flux_command(self):
        outpath = os.path.join(self.output_dir, "apple_output.png")
        (result, err) = omni.kit.commands.execute("GenerateImageFlux",
                                                  prompt="a red and green apple",
                                                  asset_png_path=outpath,
                                                  height=512,
                                                  width=512)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(outpath))
        outpath = os.path.join(self.output_dir, "cnc_output.png")
        (result, err) = omni.kit.commands.execute("GenerateImageFlux",
                                                  prompt="an industrial cnc machine",
                                                  asset_png_path=outpath,
                                                  height=512,
                                                  width=512)
        self.assertTrue(result)
