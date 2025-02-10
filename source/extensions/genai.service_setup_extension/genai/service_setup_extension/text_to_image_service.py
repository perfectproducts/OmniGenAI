# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path
from pydantic import BaseModel, Field
import io
import base64

from omni.services.core.routers import ServiceAPIRouter
from genai.flux.core import FluxWrapper
from PIL import Image
router = ServiceAPIRouter(tags=["GenAI Text to Image Service Setup"])


class TextToImageDataModel(BaseModel):

    prompt: str = Field(
        default="a beautiful image of a cat",
        title="Prompt",
        description="Prompt for the image to be generated",
    )

    height: int = Field(
        default=1024,
        title="Height",
        description="Height of the image to be generated",
    )
    width: int = Field(
        default=1024,
        title="Width",
        description="Width of the image to be generated",
    )
    inference_steps: int = Field(
        default=8,
        title="Inference Steps",
        description="Number of inference steps to be used",
    )
    guidance_scale: float = Field(
        default=3.5,
        title="Guidance Scale",
        description="Guidance scale to be used",
    )
    seed: int = Field(
        default=None,
        title="Seed",
        description="Seed to be used",
    )



@router.post(
    "/text_to_image",
    summary="Generate an image",
    description="An endpoint to generate an image",
)
async def generate_image(image_data: TextToImageDataModel) -> bytes:
    print("[genai.service_setup_extension] generate_image was called")

    flux = FluxWrapper.get_instance()

    image: Image.Image = flux.generate(image_data.prompt,
                                       image_data.height,
                                       image_data.width,
                                       image_data.inference_steps,
                                       image_data.guidance_scale,
                                       image_data.seed)

    # return image as bytes with base64 encoding
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    image_bytes_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return image_bytes_base64
