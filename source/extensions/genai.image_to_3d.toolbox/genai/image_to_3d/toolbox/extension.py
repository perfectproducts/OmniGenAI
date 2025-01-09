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
import asyncio

# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class ImageTo3dToolboxExtension(omni.ext.IExt):
    """This extension manages a simple counter UI."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.

    def on_generate_3d_clicked(self):
        print("Generate 3D clicked")
        if self._image_path is None:
            print("No image selected")
            return

        print("Generate 3D", self._image_path)
        base, ext = os.path.splitext(self._image_path)
        usd_path = base + ".usd"

        (result, err) = omni.kit.commands.execute("Generate3dFromImageTrellis",
                                            image_path=self._image_path,
                                            asset_usd_path=usd_path)
        # wait for the conversion to finish by polling the usd file

        if not result:
            print(f"Failed to generate 3D from image {err}")
            return
        print(f"Waiting for 3D to be generated {result, err}")
        timout = 10
        while not os.path.exists(usd_path) and timout > 0:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
            timout -= 1

        if not os.path.exists(usd_path):
            print("Failed to generate 3D from image")
            return

        ctx = omni.usd.get_context()
        ctx.open_stage(usd_path)


    def on_open_image_handler(self,
                              filename: str,
                              dirname: str,
                              extension: str = "",
                              selections: list = []):
        print(f"> open '{filename}{extension}' from '{dirname}' with additional selections '{selections}'")
        if not dirname.endswith("/"):
            dirname += "/"
        filepath = f"{dirname}{filename}{extension}"
        self._image_path = filepath
        self.update_image()

    def on_select_image_clicked(self):
        file_importer = get_file_importer()
        if not file_importer:
            return
        file_importer.show_window(
            title="open image",
            import_button_label="open",
            import_handler=self.on_open_image_handler,
            show_only_folders=False,
            file_extension_types=[("*.png", "PNG files"),
                                  ("*.jpg", "JPG files"),
                                  ("*.jpeg", "JPEG files")],
            file_extension="png")

    def update_image(self):
        print("update image", self._image_path)
        if self._image_path is None:
            self.image_preview.source_url = self._empty_image_path
        else:
            self.image_preview.source_url = self._image_path

    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        print("[genai.image_to_3d.toolbox] Extension startup")
        self._data_dir = os.path.dirname(os.path.realpath(__file__))+"/../../../data"
        self._image_path = None

        self._empty_image_path = f"{self._data_dir}/image_icon.svg"
        self._window = ui.Window(
            "GenAI Image To 3D Toolbox", width=400, height=360
        )

        with self._window.frame:
            with ui.VStack():
                ui.Label("Trellis Image To 3D", style={"font_size": 18.0}, height=24)
                with ui.HStack():
                    # center the image
                    ui.Spacer()
                    self.image_preview = ui.Image(width=256, height=256, alignment=ui.Alignment.CENTER)
                    ui.Spacer()
                with ui.HStack():
                    ui.Button("Generate 3D", clicked_fn=self.on_generate_3d_clicked,height=40)
                    ui.Button("...", clicked_fn=self.on_select_image_clicked,height=40, width=40)

        self.update_image()

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[genai.image_to_3d.toolbox] Extension shutdown")
