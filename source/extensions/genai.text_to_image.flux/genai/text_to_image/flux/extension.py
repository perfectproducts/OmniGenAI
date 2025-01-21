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
from .commands import GenerateImageFlux
import carb
import os
from huggingface_hub import login

# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class FluxExtension(omni.ext.IExt):
    """This is a blank extension template."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.
    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        print("[genai.text_to_image.flux] Extension startup")
        settings = carb.settings.get_settings()
        token = settings.get_as_string("huggingface_hub_token")
        if not token:
            # try to get the token from the environment variable
            token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token)
        else:
            print("No huggingface token found")
            
        omni.kit.commands.register(GenerateImageFlux)

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.text_to_image.flux] Extension shutdown")
        omni.kit.commands.unregister(GenerateImageFlux)
