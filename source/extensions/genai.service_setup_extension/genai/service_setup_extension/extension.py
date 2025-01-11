# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext
from omni.services.core import main
from .text_to_image_service import router


class ServiceSetupExtension(omni.ext.IExt):
    """This extension manages the service setup"""
    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        main.register_router(router)
        print("[genai.service_setup_extension] ServiceSetupExtension startup : Local Docs -  http://localhost:8011/docs")

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        main.deregister_router(router)
        print("[genai.service_setup_extension] ServiceSetupExtension shutdown")
