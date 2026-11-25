
# SEAFLOW3D

This repository contains the official implementation of our paper on stereo scene flow estimation.
<img width="806" height="395" alt="image" src="https://github.com/user-attachments/assets/07abcc97-7fcf-4dbe-9afb-ce0dceee746e" />

## Update (2026-11): bug fix and retrained weights

- **Bug fix.** In the conference code the update block's returned hidden state was thrown away at every iteration
  (`net, ... = self.update_block(net8, ...)`), so the GRU had no memory across iterations at 1/16, 1/8 and 1/4 resolution.
  `model_home/SEAFlow3D.py` now carries it on to the next iteration (`net8, ... = self.update_block(net8, ...)`).
  The same three lines are fixed in `model_home/RFlow3D_MSabv4_2.py` (the same network under its old name) and in `model_home/SEAFlow3DP.py`.
  Layer names are unchanged.
- **Retrained weights.** The model was retrained from scratch with the fixed code (commands in [Training](#training)).
  The previous checkpoints were trained with the bug, so please use the new ones in [Pretrained Weights](#pretrained-weights).
- **Training and evaluation code.** `train_flowp2p.py`, `flowp2p_validation.py`, `Dataset_home/dataset_3dflowuvd.py` and the modules the model imports
  are now the exact versions that trained the released weights. Dataset roots are passed on the command line.
- **KITTI submission.** New `kitti_submission.py`: GA-Net disparities as the network input and as D1, D2 = D1 / τ,
  and the rigid background refinement of CamLiFlow turned on by default.

The conference version of the code and its checkpoint links remain available at commit `abb6421`.

## Paper

**Paper link:** [Add your paper link here]

If you find this project useful in your research, please consider citing our paper.


## Run on ZED X
<img width="2979" height="1646" alt="git" src="https://github.com/user-attachments/assets/364201cf-f229-4c61-bd81-a6b6513d15b6" />

## Environment

The code has been tested with the following setup:

- Python 3.9
- PyTorch 2.0.1
- CUDA 11.8

We recommend creating a new conda environment before installation.

## Installation

Create the environment:

```bash
conda create -n seaflow3d python=3.9
conda activate seaflow3d
```

Install PyTorch and other dependencies:

```bash
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
```

Compile the custom correlation operator:

```bash
cd alt_cuda_corr
pip install -e . --no-build-isolation
cd ..
```

## Dataset Preparation

Pass the dataset roots to `train_flowp2p.py` with `--things_root`, `--driving_root`, `--sintel_root`, `--kitti_root` and `--vkitti_root`.
The expected directory layouts are defined in `Dataset_home/dataset_3dflowuvd.py`.

For KITTI, the stereo disparities fed to the network during training are read from `<kitti_root>/training/disp_monster_training/`
(MonSter disparities of both frames: `000000_10.png`, `000000_11.png`, ...), next to the usual `image_2`, `flow_occ`, `disp_occ_0` and `disp_occ_1`.

## Training

The released weights were trained with the two commands below, on 2 GPUs with 4 samples per GPU, in fp32:

```bash
# 1) pretraining on the mixed set: FlyingThings3D clean + final, Driving x2, Sintel clean + final x6, KITTI 200 pairs x200, VKITTI2
CUDA_VISIBLE_DEVICES=0,1 python train_flowp2p.py --name seaflow3d_mixed --stage pretrain --gpus 0 1 --batch_size 8 \
    --image_size 320 640 --lr 0.00025 --num_steps 150000 --val_freq 10000 --num_workers 16 --prefetch_factor 4 \
    --things_root datasets/FlyingThings3D --driving_root datasets/Driving --sintel_root datasets/Sintel \
    --kitti_root datasets/KITTI --vkitti_root datasets/VKITTI2

# 2) fine-tuning on all 200 KITTI training pairs
CUDA_VISIBLE_DEVICES=0,1 python train_flowp2p.py --name seaflow3d_kitti --stage kitti --restore_ckpt checkpoints/seaflow3d_mixed.pth \
    --gpus 0 1 --batch_size 8 --image_size 320 960 --lr 0.000125 --num_steps 45000 --val_freq 5000 \
    --num_workers 16 --prefetch_factor 4 --kitti_root datasets/KITTI
```

- Keep at least 4 samples per GPU. The update block uses BatchNorm, and under `DataParallel` its statistics are computed on each GPU separately.
- Train in fp32. Mixed precision overflowed in that BatchNorm in an earlier run.
- The KITTI validation printed during training uses training pairs that the model is trained on, so it only shows convergence.

## Demo

Before testing, please set the paths of the left and right input images in `RFlow3Dtest.py`.
The demo computes the disparities with MonSter (`MonSter/demogo.py`) before running SEA-Flow3D.

```python
def Demo_runtime(model):
    left_root = '/mnt/hdd/hanling/KITTI/testing/image_2/'
    right_root = '/mnt/hdd/hanling/KITTI/testing/image_3/'
```

Then run:

```bash
python RFlow3Dtest.py --model checkpoints/seaflow3d_kitti.pth
```

## KITTI 2015 Submission

`kitti_submission.py` writes `kitti_sceneflow2015.zip` (`disp_0`, `disp_1`, `flow`) and `kitti_flow2015.zip` (`flow`) for the 200 test pairs.

1. **Network.** The input is the two left images plus the GA-Net disparities of both frames. D1 is the GA-Net disparity with its invalid pixels
   filled the same way as the devkit's `interpolateBackground`. D2 = D1 / τ, where τ = Z2 / Z1 comes from the predicted disparity change.
   Inference runs at full resolution in fp32 without test-time augmentation.
2. **Rigid background refinement (on by default).** This follows CamLiFlow's `refine_background.py`, which in turn comes from RigidMask.
   - The background is the static classes (label ≤ 22) of the DDR-Net segmentation, shrunk by 5 pixels (`--erode`) to stay away from segmentation errors at object boundaries.
   - A camera motion is fitted on the background, and the background flow and D2 are replaced by the rigid solution. D1 is unchanged.
   - The essential matrix uses OpenCV's `findEssentialMat` branch already in CamLiFlow's code, so NG-RANSAC is not needed. A GPU is required.
   - Use `--no_refine` to skip this step, or `--erode 0` to use the original background mask.

Data:

- Test images: `<kitti_root>/testing/image_2` from the [KITTI scene flow 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) dataset.
- GA-Net disparities of the test set, released by RAFT-3D: [disp_ganet_testing](https://drive.google.com/file/d/1_rP09NpCKEoxnbPF9_S88FHS6p6u3o6H/view?usp=sharing).
  Default location: `<kitti_root>/testing/disp_ganet` (or pass `--disp_dir`).
- For the refinement:
  - a clone of [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow). Its `refine_utils.py` is used as is and is not included here. Default location: `third_party/CamLiFlow`.
  - the DDR-Net segmentation [semantic_ddr.zip](https://drive.google.com/file/d/1dVSJeE9BBmVv2rCe5TR0PVanEv2WzwIy/view?usp=sharing) from the CamLiFlow README. Default location: `<kitti_root>/testing/semantic_ddr`.
  - the KITTI calibration (`data_scene_flow_calib.zip`, folder `calib_cam_to_cam`). Default location: `<kitti_root>/testing/calib_cam_to_cam`.

```bash
git clone https://github.com/MCG-NJU/CamLiFlow.git third_party/CamLiFlow
python kitti_submission.py --ckpt checkpoints/seaflow3d_kitti.pth --kitti_root datasets/KITTI --out submission
```

## Pretrained Weights

The checkpoints are attached to the [v2.0 release](https://github.com/HanLingsgjk/SEAFLOW3D/releases/tag/v2.0).

| Checkpoint | Training | MD5 |
|---|---|---|
| [[all_mix] seaflow3d_mixed.pth](https://github.com/HanLingsgjk/SEAFLOW3D/releases/download/v2.0/seaflow3d_mixed.pth) | `--stage pretrain`, 150k steps | `4e5dbbc782aad5a1eea51b8a15271e37` |
| [[kitti] seaflow3d_kitti.pth](https://github.com/HanLingsgjk/SEAFLOW3D/releases/download/v2.0/seaflow3d_kitti.pth) | `--stage kitti` from all_mix, 45k steps | `861672b47d23720e03cf3c33fc95bba9` |

Both checkpoints were saved from `nn.DataParallel`, so their keys start with `module.`.
They load with `strict=True` into `nn.DataParallel(SEAFLOW3D(args))` (`SEAFLOW3D` from `model_home/SEAFlow3D.py`), as in `RFlow3Dtest.py`;
`kitti_submission.py` strips the prefix and loads them into `SEAFLOW3D` directly.

## Notes

- This project includes custom CUDA operators.
- Please make sure your CUDA, PyTorch, and compiler versions are compatible.
- If you encounter installation issues, first check whether your environment matches the settings listed above.
- The rigid background refinement relies on code from [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow) and [RigidMask](https://github.com/gengshan-y/rigidmask).
  That code is not redistributed here.
