# SyncTwin Omniverse GenAI Toolkit


## Overview

The SyncTwin Omniverse GenAI Toolkit is a Omniverse KIT Application with GenAI Extensions for educational purposes.
We develop on the Windows platform with KIT SDK 106.5

![Image to 3D Toolbox](docs/screenshot_image_to_3d.png)


### KIT Extensions

- **genai.image_to_3d.toolbox:** toolbox for image to 3D generation
- **genai.image_to_3d.trellis:** the currently best image to 3d pipeline
- **genai.text_to_image.toolbox:** toolbox for text to image generation (to be released)
- **genai.text_to_image.flux:** the dev pipeline from black forrest labs (to be released)

### Prerequisites

Download and install [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)


### Installation

In order to use the app just clone the repository e.g. into c:\synctwin and launch it with

```bash
git clone https://github.com/perfectproducts/OmniGenAI.git

cd OmniGenAI

repo build
repo launch -d

```
The first launch installs additional dependencies and might take up to 20 minutes depending on your internet connection.

### Usage
You can use either the toolbox UI or issue a command from your extension.

#### Image to 3D Toolbox:

- click "..." button to select image
- click "Generate 3D" button to generate the 3d asset

#### Command:
```python
Generate3dFromImageTrellis(
                 image_path: str,
                 asset_usd_path: str,
                 ss_sampling_steps: int = 12,
                 ss_guidance_strength: float = 7.5,
                 slat_sampling_steps: int = 12,
                 slat_guidance_strength: float = 3.0,
                 seed: int = 1,
                 mesh_simplify: float = 0.95,
                 texture_size: int = 1024)
```

## License

see [LICENSE](LICENSE) file for detailed license statements

### Omniverse SDK
Development using the Omniverse Kit SDK is subject to the licensing terms detailed [here](https://docs.omniverse.nvidia.com/dev-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).

### TRELLIS
MIT License

### FLUX
The genai.text_to_image.flux extension incorporates FLUX.1 [dev], which is licensed by Black Forest Labs, Inc. under the FLUX.1 [dev] Non-Commercial License.



## Data Collection
The Omniverse Kit SDK collects anonymous usage data to help improve software performance and aid in diagnostic purposes. Rest assured, no personal information such as user email, name or any other field is collected.

To learn more about what data is collected, how NVIDIA uses it and how you can change the data collection setting [see details page](readme-assets/additional-docs/data_collection_and_use.md).


## Additional Resources


- [Santa's generative AI pipeline](https://medium.com/@mtw75/santas-generative-ai-pipeline-transforming-wishes-into-3d-magic-a6a1a452b01f)

- [Generative AI Image to 3D Visual Benchmark](https://medium.com/@mtw75/generative-ai-image-to-3d-services-apis-a-benchmark-2fb119d96a95)

- [Kit App Template Companion Tutorial](https://docs.omniverse.nvidia.com/kit/docs/kit-app-template/latest/docs/intro.html)

- [Usage and Troubleshooting](readme-assets/additional-docs/usage_and_troubleshooting.md)

- [Developer Bundle Extensions](readme-assets/additional-docs/developer_bundle_extensions.md)

- [Omniverse Kit SDK Manual](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/index.html)


## Contributing

We provide this source code as-is, feel free to contribute e.g. for other pipelines or platforms
