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
import carb
import requests
import base64
from PIL import Image
import io
from .flux_service_client import FluxServiceClient
from omni.kit.window.popup_dialog import FormDialog
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
        if self._use_service:
            # call server
            # use request to send a post request to the server
            # the server will return the image encoded in base64

            client = FluxServiceClient(host=self._service_host, port=self._service_port)
            image_bytes64 = client.generate_image(prompt=prompt, height=self._image_height, width=self._image_width, seed=0)
            image_bytes = base64.b64decode(image_bytes64)
            image = Image.open(io.BytesIO(image_bytes))
            print(f"image size: {image.size} -> save to {outpath}")
            image.save(outpath)
            self.image_path = outpath

            # the client will update the image path
        else:
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


    def _on_settings_ok(self, dialog: FormDialog):
        values = dialog.get_values()
        self._use_service = values["use_service"]
        self._service_host = values["host"]
        self._service_port = values["port"]
        settings = carb.settings.get_settings()
        settings.set("/persistent/flux/use_service", self._use_service)
        settings.set("/persistent/flux/host", self._service_host)
        settings.set("/persistent/flux/port", self._service_port)
        dialog.hide()

    # build the dialog just by adding field_defs
    def _build_settings_dialog(self) -> FormDialog:

        field_defs = [
            FormDialog.FieldDef("use_service", "use service:  ", ui.CheckBox, self._use_service),
            FormDialog.FieldDef("host", "host:  ", ui.StringField, self._service_host),
            FormDialog.FieldDef("port", "port:  ", ui.IntField, self._service_port),
        ]
        dialog = FormDialog(
            title="Settings",
            message="Please specify the following paths:",
            field_defs=field_defs,
            ok_handler=self._on_settings_ok,
        )
        return dialog

    def on_configure_clicked(self):
        dlg = self._build_settings_dialog()
        dlg.show()

    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        print("[genai.text_to_image.toolbox] Extension startup")
        self.image_path = None
        self._data_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/../../../data")

        self._empty_image_path = f"{self._data_dir}/image_icon.svg"
        self._image_directory = self._data_dir+"/images"
        if not os.path.exists(self._image_directory):
            os.makedirs(self._image_directory)
        self._count = 0
        self._image_height = 512
        self._image_width = 512
        settings = carb.settings.get_settings()

        self._use_service = settings.get_as_bool("/persistent/flux/use_service")
        self._service_host = settings.get_as_string("host")
        if self._service_host == "":
            self._service_host = "192.168.178.198"
        self._service_port = settings.get_as_int("/persistent/flux/port")
        if self._service_port == 0:
            self._service_port = 8011

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
                    ui.Button(image_url=f"{self._data_dir}/folder.svg", clicked_fn=self.on_select_image_directory_clicked,height=40, width=40, tooltip="select image directory")
                    ui.Button("Generate Image", clicked_fn=self.on_generate_clicked,height=40)
                    ui.Button(image_url=f"{self._data_dir}/settings.svg", clicked_fn=self.on_configure_clicked,height=40, width=40, tooltip="configure")
                # add a image view
                ui.Label("Image")
                self.image_view = ui.Image(width=400, height=400, alignment=ui.Alignment.CENTER)
                # add a button to generate image
        self.update_image()

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.text_to_image.toolbox] Extension shutdown")
