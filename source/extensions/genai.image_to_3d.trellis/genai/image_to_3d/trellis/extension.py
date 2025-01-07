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
from .trellis_wrapper import get_trellis_instance
import asyncio
from .commands import Generate3dFromImageTrellis



class TrellisExtension(omni.ext.IExt):
    """This is a blank extension template."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.

    def on_startup(self, _ext_id):        
        check_packages()        
        
        # register the commands
        omni.kit.commands.register(Generate3dFromImageTrellis)        
        """This is called every time the extension is activated."""        

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.image_to_3d.trellis] Extension shutdown")
        omni.kit.commands.unregister(Generate3dFromImageTrellis)
        get_trellis_instance().shutdown()
        print("extension shutdown complete")

