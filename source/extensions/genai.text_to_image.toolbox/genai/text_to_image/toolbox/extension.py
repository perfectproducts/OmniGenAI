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
import os
from omni.kit.window.file_importer import get_file_importer
from genai.text_to_image.flux import destroy_flux_instance

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
        prompt = self.prompt_input_model.get_value_as_string()
        outpath = os.path.join(self._image_directory, f"{prompt.replace(' ', '_').replace('.', '_').replace('/', '_').replace(',', '_')}.png")
        (result, err) = omni.kit.commands.execute("GenerateImageFlux",
                                                  prompt=prompt,
                                                  asset_png_path=outpath,
                                                  height=self._image_height,
                                                  width=self._image_width)
        if result:
            self.image_path = outpath

        else:
            print(err)
            self.image_path = None
        self.update_image()


    def on_select_image_directory_handler(self,
                                          filename: str,
                                          dirname: str,
                                          extension: str = "",
                                          selections: list = []):
        print(f"> open '{filename}{extension}' from '{dirname}' with additional selections '{selections}'")
        if not dirname.endswith("/"):
            dirname += "/"
        filepath = f"{dirname}{filename}{extension}"
        print(filepath)
        self._image_directory = dirname

    def on_select_image_directory_clicked(self):
        file_importer = get_file_importer()
        if not file_importer:
            return
        file_importer.show_window(
            title="select image directory",
            import_button_label="open",
            import_handler=self.on_select_image_directory_handler,
            show_only_folders=True,
            filename_url=self._image_directory+"/")


    def update_image(self):
        if self.image_path:
            self.image_view.source_url = self.image_path
        else:
            self.image_view.source_url = self._empty_image_path

    def on_deactivate_clicked(self):
        print("deactivate clicked")
        destroy_flux_instance()
        self.image_path = None
        self.update_image()

    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        print("[genai.text_to_image.toolbox] Extension startup")
        self.image_path = None
        self._data_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/../../../data")
        print("-->" +self._data_dir)
        self._empty_image_path = f"{self._data_dir}/image_icon.svg"
        self._image_directory = self._data_dir
        self._count = 0
        self._image_height = 512
        self._image_width = 512

        self._window = ui.Window(
            "Generative AI Text To Image Toolbox", width=410, height=670
        )
        with self._window.frame:
            with ui.VStack():

                ui.Label("Flux Text To Image", style={"font_size": 18.0}, height=24)

                # prompt input
                ui.Label("Prompt")
                self.prompt_input_model = ui.StringField(height=100).model
                self.prompt_input_model.set_value("a fruit basket")
                with ui.HStack():
                    ui.Button("...", clicked_fn=self.on_select_image_directory_clicked,height=40, width=40, tooltip="select image directory")
                    ui.Button("Generate Image", clicked_fn=self.on_generate_clicked,height=40)
                    ui.Button("X", clicked_fn=self.on_deactivate_clicked,height=40, width=40, tooltip="clear pipline")
                # add a image view
                ui.Label("Image")
                self.image_view = ui.Image(width=400, height=400, alignment=ui.Alignment.CENTER)
                # add a button to generate image
        self.update_image()

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.text_to_image.toolbox] Extension shutdown")
