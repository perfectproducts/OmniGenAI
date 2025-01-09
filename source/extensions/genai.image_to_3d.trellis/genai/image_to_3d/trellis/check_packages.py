import git
import subprocess
import os
import omni.kit.pipapi as pip
import tempfile
import torch
import subprocess

# init more peculiar packages with special requirements for windows

def _check_ninja():
    try:

        # Download ninja-win.zip
        import urllib.request
        import zipfile
        import sys

        ninja_url = "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip"
        ninja_zip = "ninja-win.zip"
        ninja_exe = "ninja.exe"

        if os.path.exists(ninja_exe):
            print(f"ninja already installed: {ninja_exe}")
            return True

        # Download ninja zip file
        urllib.request.urlretrieve(ninja_url, ninja_zip)

        # Extract ninja.exe
        with zipfile.ZipFile(ninja_zip, 'r') as zip_ref:
            zip_ref.extractall()

        # Add ninja directory to PATH
        ninja_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        if ninja_dir not in os.environ['PATH']:
            os.environ['PATH'] = ninja_dir + os.pathsep + os.environ['PATH']

        # Clean up zip file
        os.remove(ninja_zip)
        print("check output")
        # Check ninja version
        r = subprocess.check_output('ninja --version'.split())
        print(f"ninja check: {r}")
        return True
    except Exception as e:
        print(f"ninja check failed: {e}")
        return False


def _check_utils3d():
    try:
        import utils3d
        torch = utils3d.torch
        if torch is None:
            print("utils3d.torch is None")
            return False
    except ImportError:
        pip.call_pip(
            args=["install", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8#egg=utils3d"]
        )
        import utils3d
        print(f"utils3d: {utils3d}")
        print(f"utils3d.torch: {utils3d.torch}")
        return True
    except Exception as e:
        print(f"utils3d check failed: {e}")
        return False
    return True


def _check_flash_attn():
    try:
        import flash_attn
        print(f"flash_attn: {flash_attn.__version__}")
    except ImportError:
        print("flash_attn not found")
        r = pip.call_pip(args=["install", "https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"] )
        print(f"flash_attn installed: {r}")
    return True


def _check_kaolin():
    print("check kaolin")
    try:
        import kaolin
        print(f"kaolin version: *{kaolin.__version__}*")
        import warp as wp
        print(f"warp version: *{wp.__version__}*")

    except ImportError:
        print("kaolin not found")
        r = pip.call_pip(args=["install", "kaolin", "-f", "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"])
        print(f"kaolin installed: {r}")

    return True

def _check_nvdiffrast():
    print("check diffrast")

    try:
        import nvdiffrast
        print(f"nvdiffrast: {nvdiffrast.__version__}")
    except ImportError:
        tmp_dir = tempfile.mkdtemp()
        print("build nvdiffrast")

        repo_url = "https://github.com/NVlabs/nvdiffrast.git"
        target_path = os.path.join(tmp_dir, "nvdiffrast")
        print("nvdiffrast not found, installing...")
        # first clone the repo
        try:
            git.Repo.clone_from(repo_url, target_path)
        except Exception as e:
            print(f"Error cloning repo: {e}")
        # install the repo
        r = pip.call_pip(
            args=["install", target_path])
        print(f"nvdiffrast installed: {r}")
    return True

def _check_diffoctreerast():
    print("check diffoctreerast")
    try:
        import diffoctreerast
        print(f"diffoctreerast: {diffoctreerast}")

    except ImportError:
        # as seen in https://github.com/nitinmukesh/TRELLIS-for-windows/blob/main/1%E3%80%81install-uv-qinglong.ps1 :
        r = pip.call_pip(args=["install", "--no-build-isolation", "git+https://github.com/JeffreyXiang/diffoctreerast.git"])
        print(f"diffoctreerast installed: {r}")
    return True
    # git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ./tmp/extensions/diffoctreerast


def _check_diff_gaussian_rasterization():
    print("check diff_gaussian_rasterization")
    try:
        import diff_gaussian_rasterization
        print(f"diff_gaussian_rasterization: {diff_gaussian_rasterization}")
    except ImportError:
        r = pip.call_pip(args=["install", "--no-build-isolation", "git+https://github.com/sdbds/diff-gaussian-rasterization"])
        print(f"diff_gaussian_rasterization installed: {r}")
    return True

def _check_trellis_repo():
    print("check trellis submodules")
    # check if we have a TRELLIS folder
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "TRELLIS")):
        print("trellis not found, checking out...")
        trellis_git = "https://github.com/microsoft/TRELLIS.git"
        trellis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TRELLIS")
        try:
            git.Repo.clone_from(trellis_git, trellis_path, recursive=True)
            print("git cloned.")
        except Exception as e:
            print(f"Error cloning trellis repo: {e}")
            return False
    else:
        print("trellis already exists")
    return True


def _check_trellis_submodules():
    print("check trellis submodules")
    try:
        from .TRELLIS import trellis
        print(f"trellis: {trellis}")
    except ImportError:
        print("trellis not found")
        return False
    return True


def _check_cuda():
    try:
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    version = line.split("release")[-1].strip().split(",")[0]
                    print("CUDA version:", version)
        else:
            print("nvcc not found. CUDA might not be installed.")
    except FileNotFoundError:
        print("nvcc not found. Ensure CUDA is installed and added to PATH.")
    if version != "12.4":
        print("CUDA version is not 12.4 - please install torch with CUDA 12.4")
        cuda_url = "https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local"
        print(f"Please download CUDA 12.4 from {cuda_url}")
        # open browser to the url
        import webbrowser
        webbrowser.open(cuda_url)
        return False
    return True

def _remove_pip(package):
    try:
        pip.call_pip(args=["uninstall", package, "-y"])
    except Exception as e:
        print(f"Error uninstalling {package}: {e}")

def uninstall_packages():

    _remove_pip("kaolin")
    _remove_pip("flash_attn")
    _remove_pip("utils3d")
    _remove_pip("nvdiffrast")

def check_packages():
    # try this if installation fails
    #uninstall_packages()
    #--------------------------------

    if not _check_cuda():
        ## doesnt make sense to continue if cuda is not 12.4
        return
    print("Checking packages")
    print("trellis:", _check_trellis_repo())
    print("ninja:", _check_ninja())
    print("utils3d:", _check_utils3d())
    print("flash_attn:", _check_flash_attn())
    print("kaolin:", _check_kaolin())
    print("nvdiffrast:", _check_nvdiffrast())
    print("diffoctreerast:", _check_diffoctreerast())
    print("diff_gaussian_rasterization:", _check_diff_gaussian_rasterization())
    print("trellis_submodules:", _check_trellis_submodules())
