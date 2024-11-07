# Readme
## Create conda environment
```bash
conda create -n deep python=3.12
conda activate deep
```

## Without CUDA
Remove the line `-i https://download.pytorch.org/whl/cu124` from requirements.txt file.

## How to install CUDA
Note that step 3 is only neccessary if you are using another version than the one specified in requirements.txt.

1. Under Compute Platform, check the latest supported CUDA toolkit version for PyTorch: https://pytorch.org/get-started/locally/.
2. Download corresponding CUDA toolkit version from https://developer.nvidia.com/cuda-toolkit-archive.
3. (Optional) Select your OS, package=pip, and CUDA version, and run the pip install command showing on the PyTorch webpage (https://pytorch.org/get-started/locally/). Remember to activating conda enviroment. The command could look like:
```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Install dependencies
```bash
pip install -r requirements.txt
```