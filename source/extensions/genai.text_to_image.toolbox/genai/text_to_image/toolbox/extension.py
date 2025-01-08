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
import omni.ui as ui


# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class TextToImageExtension(omni.ext.IExt):
    """This extension manages a simple counter UI."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.
    def on_generate_clicked(self):
        print("Generate button clicked")

    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        print("[genai.text_to_image.toolbox] Extension startup")

        self._count = 0
        self._window = ui.Window(
            "Generative AI Text To Image Toolbox", width=300, height=300
        )
        with self._window.frame:
            with ui.VStack():
                ui.Label("Flux Text To Image", style={"font_size": 18.0}, height=24)
                 # prompt input
                ui.Label("Prompt")
                self.prompt_input_model = ui.StringField().model
                self.prompt_input_model.set_value("a fruit basket")
                # add a image view
                ui.Label("Image")
                with ui.HStack():
                    # center the image
                    ui.Spacer()
                    self.image_view = ui.Image(width=256, height=256, alignment=ui.Alignment.CENTER)
                    ui.Spacer()
                # add a button to generate image
                self.generate_button = ui.Button("Generate", clicked_fn=self.on_generate_clicked)

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.text_to_image.toolbox] Extension shutdown")
