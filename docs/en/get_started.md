## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you run using GPU)

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n easyfl python=3.7 -y
    conda activate easyfl
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```
    or
    ```shell
    pip install torch==1.10.1 torchvision==0.11.2
    ```

4. _You can skip the following CUDA-related content if you plan to run it on CPU._ Make sure that your compilation CUDA version and runtime CUDA version match. 

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.,` 1. If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g.,` 2. If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt package,
    you can use more CUDA versions such as 9.0.

### Install EasyFL

```shell
pip install easyfl
```

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMDetection with conda.

```shell
conda create -n easyfl python=3.7 -y
conda activate easyfl

# Without GPU
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch -y

# With GPU
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install easyfl
git clone https://github.com/EasyFL-AI/easyfl.git
cd easyfl
pip install -v -e .
```

## Verification

To verify whether EasyFL is installed correctly, we can run the following sample code to test.

```python
import easyfl

easyfl.init()
```

The above code is supposed to run successfully after you finish the installation.
