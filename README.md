
# SEAFLOW3D

Official implementation of **SEA-Flow3D: Simplified, Efficient, and Accurate Scene Flow via Spatial Vector Sampling and Multi-scale Refinement** (CVPR 2026).

**Paper:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Ling_SEA-Flow3D_Simplified_Efficient_and_Accurate_Scene_Flow_via_Spatial_Vector_CVPR_2026_paper.html) · [Citation](#citation)

<img width="806" height="395" alt="image" src="https://github.com/user-attachments/assets/07abcc97-7fcf-4dbe-9afb-ce0dceee746e" />

## Updates

- **2026-11-28, v2.1.**
  - New model option `--model seaflow3d_cd` (`model_home/SEAFlow3D_corrd.py`). It passes the update block the depth-change cue relative to the current estimate.
  - New KITTI weights for it, trained with the KITTI schedule in [Training](#training).
  - `kitti_submission.py` now reproduces our two current KITTI 2015 entries (see [KITTI 2015 Results](#kitti-2015-results)).
    Both use an MSeg background mask shrunk by 5 px for the rigid background refinement.
- **2026-11-25, v2.0.** The conference code threw away the GRU hidden state at every iteration. It is now carried across iterations, and the weights were retrained.

The conference version of the code and its checkpoint links remain available at commit `abb6421`.

## KITTI 2015 Results

Scene flow test set, outliers in % (background / foreground / all pixels), as shown by the KITTI server for our two entries submitted on 2026-11-28.
Both use the v2.1 checkpoint `seaflow3d_driving_kitti.pth` and the rigid background refinement with the MSeg mask shrunk by 5 px.
They differ only in the stereo disparities, which serve both as the network input and as D1.

| Entry | Disparities | Rigid background refinement | D1-all | D2-all | Fl-all | SF-bg | SF-fg | SF-all |
|---|---|---|---|---|---|---|---|---|
| [SEA-Flow3D + Monster](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=scene_flow&result=70059c08e1e64750da2d91a471ad41a7bf6489ed) | MonSter++ | MSeg mask, shrunk by 5 px | 1.37 | 2.30 | 2.37 | 2.09 | 9.42 | 3.31 |
| [SEA-Flow3D+GANET](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=scene_flow&result=523304ac2cc2cd5138436a0622bc0008a3401baa) | GA-Net | MSeg mask, shrunk by 5 px | 1.81 | 2.95 | 2.95 | 2.82 | 10.97 | 4.18 |

The network runs in 0.17 s per pair on an NVIDIA L40. How to reproduce both entries is described in [KITTI 2015 Submission](#kitti-2015-submission).

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

The v2.1 weights were trained with the two commands below, on 2 GPUs with 3 samples per GPU, in fp32:

```bash
# 1) driving stage from scratch: VKITTI2 + the 200 KITTI training pairs x200 + Driving
CUDA_VISIBLE_DEVICES=0,1 python train_flowp2p.py --name seaflow3d_driving --model seaflow3d_cd --stage driving --gpus 0 1 \
    --batch_size 6 --image_size 320 960 --lr 0.00025 --num_steps 160000 --val_freq 10000 --num_workers 16 --prefetch_factor 4 \
    --driving_root datasets/Driving --kitti_root datasets/KITTI --vkitti_root datasets/VKITTI2

# 2) fine-tuning on the 200 KITTI training pairs
CUDA_VISIBLE_DEVICES=0,1 python train_flowp2p.py --name seaflow3d_driving_kitti --model seaflow3d_cd --stage kitti \
    --restore_ckpt checkpoints/seaflow3d_driving.pth --gpus 0 1 --batch_size 6 --image_size 320 960 --lr 0.0001 \
    --num_steps 20000 --val_freq 5000 --num_workers 16 --prefetch_factor 4 --kitti_root datasets/KITTI
```

The v2.0 weights (`--model seaflow3d`, the default) were trained with these two commands, on 2 GPUs with 4 samples per GPU, in fp32:

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

- The update block uses BatchNorm, and under `DataParallel` its statistics are computed on each GPU separately. Keep the number of samples per GPU of the run you reproduce (3 for v2.1, 4 for v2.0).
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
python RFlow3Dtest.py --model checkpoints/seaflow3d_driving_kitti.pth                        # v2.1 weights
python RFlow3Dtest.py --model checkpoints/seaflow3d_kitti.pth --network seaflow3d          # v2.0 weights
```

## KITTI 2015 Submission

`kitti_submission.py` reproduces our two KITTI 2015 scene flow entries (see [KITTI 2015 Results](#kitti-2015-results)). The defaults give the GA-Net entry.
It writes `kitti_sceneflow2015.zip` (`disp_0`, `disp_1`, `flow`) and `kitti_flow2015.zip` (`flow`) for the 200 test pairs.

1. **Network.** `seaflow3d_driving_kitti.pth` with `--model seaflow3d_cd` (the default).
   - Input: the two left images plus the GA-Net disparities of both frames.
   - Inference runs at full resolution in fp32 without test-time augmentation.
2. **D1 and D2.**
   - D1 is the GA-Net disparity with its invalid pixels filled the same way as the devkit's `interpolateBackground`.
   - D2 = D1 / τ, where τ = Z2 / Z1 comes from the predicted disparity change.
3. **Rigid background refinement (on by default).** This follows CamLiFlow's `refine_background.py`, which in turn comes from RigidMask.
   - **Background mask.** The MSeg universal-taxonomy map minus 26 movable classes: animals, vehicles, persons and riders (`MSEG_FOREGROUND` in the script).
     The mask is shrunk by 5 pixels (`--erode 5`) to stay away from segmentation errors at object boundaries.
   - **Rigid solution.** A camera motion is fitted on the background pixels whose flow stays inside the image, leaving out the top 80 rows.
     The flow and D2 of all background pixels are then replaced by the rigid solution. D1 is unchanged.
   - **Essential matrix.** It uses OpenCV's `findEssentialMat` branch already in CamLiFlow's code, so NG-RANSAC is not needed. A GPU is required.
   - **Options.**
     - `--mask ddr` uses CamLiFlow's DDR-Net segmentation instead (static classes, label ≤ 22).
     - `--erode 0` keeps the mask unshrunk.
     - `--no_refine` skips this step.
4. **Zip.** `kitti_sceneflow2015.zip` contains the refined outputs.
   `kitti_flow2015.zip`, for the optical flow benchmark, contains the network's own flow without the rigid background refinement.

Data and tools (none of them is included here):

- Test images: `<kitti_root>/testing/image_2` from the [KITTI scene flow 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) dataset.
- GA-Net disparities of the test set, released by RAFT-3D: [disp_ganet_testing](https://drive.google.com/file/d/1_rP09NpCKEoxnbPF9_S88FHS6p6u3o6H/view?usp=sharing).
  Default location: `<kitti_root>/testing/disp_ganet` (or pass `--disp_dir`).
- The KITTI calibration (`data_scene_flow_calib.zip`, folder `calib_cam_to_cam`). Default location: `<kitti_root>/testing/calib_cam_to_cam`.
- A clone of [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow). Its `refine_utils.py` is used as is. Default location: `third_party/CamLiFlow`.
- The MSeg masks of the test set. Default location: `<kitti_root>/testing/mseg/gray` (or pass `--mseg_dir`).
  Make them with [MSeg](https://github.com/mseg-dataset/mseg-semantic) as below. We used mseg-api `2bd00ae` and mseg-semantic `ac92617`.

```bash
git clone https://github.com/mseg-dataset/mseg-api.git third_party/mseg-api
git clone https://github.com/mseg-dataset/mseg-semantic.git third_party/mseg-semantic
pip install --no-deps -e third_party/mseg-api -e third_party/mseg-semantic
pip install hydra-core
# numpy >= 1.24 removed np.int and np.bool, which these two files still use:
sed -i '339s/np\.int(/int(/' third_party/mseg-semantic/mseg_semantic/model/seg_hrnet.py
sed -i 's/astype(np\.bool)/astype(bool)/g' third_party/mseg-api/mseg/utils/mask_utils_detectron2.py
# download "MSeg (3M)-480p" (mseg-3m-480p.pth) from the MSeg README to checkpoints/, then:
python -u third_party/mseg-semantic/mseg_semantic/tool/universal_demo.py \
    --config=third_party/mseg-semantic/mseg_semantic/config/test/default_config_360_ms.yaml \
    model_name mseg-3m model_path checkpoints/mseg-3m-480p.pth \
    input_file datasets/KITTI/testing/image_2 save_folder datasets/KITTI/testing/mseg
# -> datasets/KITTI/testing/mseg/gray/<frame>.png, one universal-taxonomy class id per pixel (only the *_10 frames are used)
```

- `save_folder` must be given. Otherwise MSeg writes into its own repository folder, under a name taken from the input folder.

Then run:

```bash
git clone https://github.com/MCG-NJU/CamLiFlow.git third_party/CamLiFlow
# SEA-Flow3D+GANET: GA-Net disparities, MSeg background mask shrunk by 5 px
python kitti_submission.py --ckpt checkpoints/seaflow3d_driving_kitti.pth --kitti_root datasets/KITTI --out submission_ganet
# SEA-Flow3D + Monster: MonSter++ disparities of both frames (MonSter's own PNG encoding), same MSeg background mask
python kitti_submission.py --ckpt checkpoints/seaflow3d_driving_kitti.pth --kitti_root datasets/KITTI \
    --disp_dir <MonSter++ disparities of the test set> --disp_format monster --out submission_monster
```

- With these inputs, the commands above reproduced every PNG of our two uploaded submissions byte for byte on our machine (NVIDIA L40).
- The MonSter++ disparities of the test set are not included here.
  `--disp_format monster` reads MonSter's PNG encoding, `((B << 16) | G) / 1024`, where B and G are the 16-bit channels.
- **Why the mask is shrunk.** Without shrinking, the MSeg mask counts about 2.4% of the moving-object pixels (KITTI `obj_map`, training set) as background, mostly at object boundaries, and those pixels are then replaced by the camera motion.
  On the 40 KITTI training pairs 000000, 000005, ..., 000195, the foreground SF outliers rise from 0.86% to 1.98%. These pairs were seen in training, so this is only a sanity check.

## Pretrained Weights

The checkpoints are attached to the [v2.1](https://github.com/HanLingsgjk/SEAFLOW3D/releases/tag/v2.1) and [v2.0](https://github.com/HanLingsgjk/SEAFLOW3D/releases/tag/v2.0) releases.

| Checkpoint | `--model` | Training | MD5 |
|---|---|---|---|
| [[v2.1] seaflow3d_driving.pth](https://github.com/HanLingsgjk/SEAFLOW3D/releases/download/v2.1/seaflow3d_driving.pth) | `seaflow3d_cd` | `--stage driving` from scratch, 160k steps | `eb69a922be12b256fdb8d2a700fab785` |
| [[v2.1] seaflow3d_driving_kitti.pth](https://github.com/HanLingsgjk/SEAFLOW3D/releases/download/v2.1/seaflow3d_driving_kitti.pth) | `seaflow3d_cd` | `--stage kitti` from seaflow3d_driving, 20k steps; used for our KITTI entry | `96ce489436a159b02d7f0552367c4112` |
| [[v2.0] seaflow3d_mixed.pth](https://github.com/HanLingsgjk/SEAFLOW3D/releases/download/v2.0/seaflow3d_mixed.pth) | `seaflow3d` | `--stage pretrain`, 150k steps | `4e5dbbc782aad5a1eea51b8a15271e37` |
| [[v2.0] seaflow3d_kitti.pth](https://github.com/HanLingsgjk/SEAFLOW3D/releases/download/v2.0/seaflow3d_kitti.pth) | `seaflow3d` | `--stage kitti` from seaflow3d_mixed, 45k steps | `861672b47d23720e03cf3c33fc95bba9` |

- All checkpoints were saved from `nn.DataParallel`, so their keys start with `module.`.
  They load with `strict=True` into `nn.DataParallel(model)`, as in `RFlow3Dtest.py`. `kitti_submission.py` strips the prefix and loads them into the model directly.
- **Always pass the `--model` listed here** (`--network` in `RFlow3Dtest.py`). The two models have the same layer names, so a checkpoint also loads into the wrong model without any error.
- There is no v2.1 checkpoint for the mixed pretraining set (FlyingThings3D, Sintel, ...). Use the v2.0 `seaflow3d_mixed.pth` with `--model seaflow3d`.

## Citation

If you find this project useful in your research, please consider citing our paper:

```bibtex
@InProceedings{Ling_2026_CVPR,
    author    = {Ling, Han and Sun, Quansen and Yao, Yinghua and Tsang, Ivor and Sun, Yinghui},
    title     = {SEA-Flow3D: Simplified, Efficient, and Accurate Scene Flow via Spatial Vector Sampling and Multi-scale Refinement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {36475-36484}
}
```

## Notes

- This project includes custom CUDA operators.
- Please make sure your CUDA, PyTorch, and compiler versions are compatible.
- If you encounter installation issues, first check whether your environment matches the settings listed above.
- The rigid background refinement relies on code from [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow) and [RigidMask](https://github.com/gengshan-y/rigidmask), and on masks from [MSeg](https://github.com/mseg-dataset/mseg-semantic).
  Their code and weights are not redistributed here.
