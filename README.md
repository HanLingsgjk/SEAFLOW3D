**SEAFLOW3D**

This repository contains the official implementation of our paper on stereo scene flow estimation.

**Paper**

Paper link: [Add your paper link here]

If you find this project useful in your research, please consider citing our paper.

**Environment**

The code has been tested with:
PyTorch 2.0.1
CUDA 11.8
Python 3.9

We recommend creating a new conda environment before installation.

**Installation**

Create the environment:

conda create -n seaflow3d python=3.9
conda activate seaflow3d

Install PyTorch and other dependencies:

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib==3.5
pip install opencv-python==4.8.1.78
pip install tqdm
pip install pypng
pip install scipy
pip install einops
pip install tensorboard
pip install timm==0.6.13
pip install scikit-image==0.21.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0
pip install h5py

Compile the custom correlation operator:

cd alt_cuda_corr/
pip install -e . --no-build-isolation

After installation, return to the project root directory before training or testing.

**Dataset Preparation**

Before training, please set the dataset path correctly in the corresponding training scripts or configuration files.

Make sure the dataset structure is consistent with the code requirements.

**Training**

Run the training command with the desired GPU devices:

CUDA_VISIBLE_DEVICES=0,1 python train_flow3d_ab.py --name pretrain --stage abtrain

Please modify the dataset path before running training.

**Testing**

Before testing, please set the paths of the left and right input images in the Demo_runtime function in RFlow3Dtest.py:

def Demo_runtime(model):
    left_root = '/mnt/hdd/hanling/KITTI/testing/image_2/'
    right_root = '/mnt/hdd/hanling/KITTI/testing/image_3/'

Then run:

python RFlow3Dtest.py --model=/mnt/hdd/home/linghan/SEAFLOW3D/checkpoints/tartain.pth

Please make sure the checkpoint path is correct before testing.

**Pretrained Weights**

We provide the official pretrained model at:
[Add checkpoint link here]

Download the checkpoint and update the model path in the testing command accordingly.

**Notes**

This project includes custom CUDA operators. Please ensure that your CUDA, PyTorch, and compiler versions are compatible.

If you encounter installation issues, first check whether the environment versions match the settings listed above.
