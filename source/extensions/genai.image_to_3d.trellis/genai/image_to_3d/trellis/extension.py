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
from .check_packages import check_packages
from .trellis_wrapper import destroy_trellis_instance
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
        destroy_trellis_instance()
        print("extension shutdown complete")
