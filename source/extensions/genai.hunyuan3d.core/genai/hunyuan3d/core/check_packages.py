import git
import os
import sys
import subprocess
import omni.kit.app
import omni.kit.pipapi as pip
import torch

def _check_hunyuan3d_repo():
    print("check Hunyuan3D submodules")
    # check if we have a Hunyuan3D folder
    hunyuan3d_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Hunyuan3D_2")
    if not os.path.exists(hunyuan3d_path):
        print("Hunyuan3D not found, checking out...")
        hunyuan3d_git = "https://github.com/Tencent/Hunyuan3D-2.git"
        try:
            git.Repo.clone_from(hunyuan3d_git, hunyuan3d_path, recursive=True)
            print("git cloned.")
        except Exception as e:
            print(f"Error cloning Hunyuan3D repo: {e}")
            return False
    else:
        print("Hunyuan3D already exists")
    return True

def _add_hunyuan3d_to_path():
    hunyuan3d_path = os.path.join(os.path.dirname(__file__), "Hunyuan3D_2")
    if hunyuan3d_path not in sys.path:
        sys.path.append(hunyuan3d_path)
        print(f"Added Hunyuan3D_2 to sys.path: {hunyuan3d_path}")

def _setup_custom_rasterizer():
    print("setup custom rasterizer")
    try:
        import custom_rasterizer_kernel
    except ImportError:
        print("custom_rasterizer_kernel not found, installing...")
        custom_rasterizer_path = os.path.join(os.path.dirname(__file__), "Hunyuan3D_2", "hy3dgen", "texgen", "custom_rasterizer")
        pip.call_pip(
                args=["install", "--no-build-isolation", "-e", custom_rasterizer_path]
            )
        print("custom_rasterizer installed")

def _remove_pip(package):
    try:
        pip.call_pip(args=["uninstall", package, "-y"])
    except Exception as e:
        print(f"Error uninstalling {package}: {e}")

def uninstall_dependencies(self):
    print("uninstall dependencies")
    _remove_pip("torch")
    _remove_pip("torchvision")
    _remove_pip("torchaudio")
    _remove_pip("torchmetrics")
    _remove_pip("torchdiffeq")
    _remove_pip("torchvision")







def check_packages():
    print("check packages")
    _check_hunyuan3d_repo()
    _add_hunyuan3d_to_path()
    _setup_custom_rasterizer()
