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
from .text_to_image_service import router as text_to_image_router
import carb.settings

class GenAIServiceSetupExtension(omni.ext.IExt):
    """This extension manages the service setup"""
    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        settings = carb.settings.get_settings()
        self.is_text_to_image_enabled = settings.get_as_bool("text_to_image_enabled")
        print(f"[genai.service_setup_extension] text_to_image_enabled: {self.is_text_to_image_enabled}")
        if self.is_text_to_image_enabled:
            print("[genai.service_setup_extension] Registering text_to_image_router")
            main.register_router(text_to_image_router)

        local_host = settings.get_as_string("exts/omni.services.transport.server.http/host")
        if local_host == "0.0.0.0":
            local_host = "localhost"
        local_port = settings.get_as_int("exts/omni.services.transport.server.http/port")
        print(f"[genai.service_setup_extension] ServiceSetupExtension startup : Local Docs -  http://{local_host}:{local_port}/docs")

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        if self.is_text_to_image_enabled:
            main.deregister_router(text_to_image_router)
        print("[genai.service_setup_extension] ServiceSetupExtension shutdown")
