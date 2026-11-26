# How SEAFLOW3D was trained for the June 2025 KITTI submission

This page records the training setup of the network behind the KITTI 2015 scene flow submission of 2025-06-27.
It was reconstructed from the training server: the code in `code/`, the PyCharm run configurations, the shell history and the checkpoint files.
Each setting below says where it comes from. The 2026-11 retraining in the repository root (`train_flowp2p.py`) uses a different schedule. It is described in the main README.

## 1. Network

- Class `RFlow3D_MSab_v8` in [`code/model_home/RFlow3D_MSabv4_2.py`](code/model_home/RFlow3D_MSabv4_2.py). This is SEAFLOW3D under its old name. `model_home/SEAFlow3D.py` in the repository root is the same network.
- Coarse-to-fine refinement with a fixed number of iterations: 2 at 1/16 resolution, 4 at 1/8 and 3 at 1/4. The loops are hard-coded, so the `--iters`, `--itr1` and `--itr2` arguments of the trainers have no effect on this class.
- Correlation radius 6 at 1/16 and 4 at 1/8 and 1/4, using the `alt_cuda_corr` CUDA extension. Hidden and context dimension 128.
- Inputs: the two left images and the disparities of both frames (`d1`, `d2`). Outputs: optical flow and disparity change `dz`, one prediction per iteration.
- These checkpoints were trained **with the hidden-state bug** fixed in 2026-11. In this copy the update blocks return `net, ...` instead of `net16/net8/net4, ...` (three lines), so the GRU state is not carried across iterations. Apart from those lines, the file differs from `model_home/RFlow3D_MSabv4_2.py` in the root only in timing code and unused helpers.

## 2. Trainer and optimisation

The SEAFLOW3D runs used [`code/train_flow3duvd.py`](code/train_flow3duvd.py). It always builds `RFlow3D_MSab_v8`, wrapped in `nn.DataParallel`.
The ablation runs used [`code/train_flow3d_ab.py`](code/train_flow3d_ab.py), which has the same loop and loss and chooses the variant with `--ab` (`--ab 8` is SEAFLOW3D).

| Setting | `train_flow3duvd.py` default | `train_flow3d_ab.py` default |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-4, eps 1e-8 | same |
| LR schedule | OneCycleLR over `num_steps + 100`, `pct_start 0.05`, linear annealing, no momentum cycling | same |
| Peak LR | 2.5e-4 | 2.5e-4 |
| Steps | 160,000 | 100,000 |
| Batch size | 6 (`--gpus 0 1`, so 3 per GPU) | 3 (`--gpus 0`) |
| Crop | 320 × 832 | 320 × 640 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | fp32 (`--mixed_precision` not set) | fp32 |
| Loss weight γ | 0.85 | 0.85 |
| Seed | `torch.manual_seed(1234)`, `np.random.seed(1234)` | same |
| Checkpoints | `checkpoints/<step>_<name>.pth` every 10k steps, then `checkpoints/<name>.pth` | same |
| Validation | every 10k steps: `validate_kitti` and `validate_sintel` from `RFlow3Dtest.py` | same |

Because of the fixed seed, every run that starts from scratch begins from the same initial weights.

### Loss

```
flow_loss = Σ_i γ^(N-1-i) · mean(valid · |flow_i − flow_gt|)                over all N predictions
dz_loss   = Σ_{i≥1} γ^(N-1-i) · mean(valid · valid_dz · |dz_{i-1} − dz_gt|)
loss      = flow_loss + dz_loss
valid     = (gt valid ≥ 0.5) & (|flow_gt| < 400),   valid_dz = (dz_gt < 1000)
```

### Augmentation (`Dataset_home/augment3d.py`, RAFT style)

Every training set is built with `crop_size = --image_size`, `min_scale = -0.5`, `max_scale = 0.5` and `do_flip = True`:
- Random scale `2^U(-0.5, 0.5)` with probability 0.8, plus stretch `2^U(-0.2, 0.2)` with probability 0.8.
- Horizontal flip with probability 0.5 and vertical flip with probability 0.1.
- ColorJitter (0.4, 0.4, 0.4, hue 0.5/π) and random erasing with probability 0.5.
- KITTI goes through the sparse version of the augmentor.

## 3. Disparity inputs and targets per dataset

In [`code/Dataset_home/dataset_3dflowuvd.py`](code/Dataset_home/dataset_3dflowuvd.py) the network sees disparities rather than depth, and each dataset rescales them by a random factor `s`:

| Dataset | Input disparities `d1`, `d2` | Target `dz` | Random scale `s` |
|---|---|---|---|
| FlyingThings3D (clean / final), Monkaa | ground-truth disparity | ground-truth disparity change | `s ∈ [0.5, 1.5)` |
| Driving | ground-truth disparity | ground-truth disparity change | `s ∈ [0.1, 0.5)` |
| VKITTI2 | `s · f / depth`, depth in cm | from the scene flow maps | `s ∈ [40, 80)` |
| KITTI 2015 | **MonSter** disparities of both frames (`training/disp_monster_training/*_10.png`, `*_11.png`) | `disp_occ_1 − disp_occ_0` (ground truth) | `s ∈ [0.5, 1.5)`, applied to inputs and target |
| Sintel | `360 / depth` | from the warped second-frame depth | none |

At test time on KITTI the inputs are the **GA-Net** disparities (see `README.md`). So the KITTI model was trained on MonSter disparities and tested on GA-Net disparities.

## 4. Training stages

`--stage` selects the data mix in `fetch_dataloader`. `k×` means the dataset is repeated `k` times inside the `ConcatDataset`.

| Stage | Data |
|---|---|
| `pretrain` | Monkaa + FlyingThings3D final + FlyingThings3D clean |
| `driving` | VKITTI2 + 200× KITTI (all 200 training pairs) + Driving |
| `fineting` | identical to `driving` |
| `abtrain` (ablations) | Driving + 50× KITTIab (the 100 odd-index training pairs) + 3× Sintel clean + 3× Sintel final (all Sintel training scenes except `temple_2`, `market_6`, `cave_4`, `ambush_5`, `bamboo_1`) |
| `Sintel_CamLi` | FlyingThings3D clean + final (the Things-only protocol of CamLiFlow for Sintel generalisation) |

During training, `validate_kitti` runs on the 100 even-index KITTI training pairs and `validate_sintel` on Sintel. In the `driving` stage all 200 KITTI pairs are also used for training, so that KITTI number only shows convergence.

Dataset roots on the server (hard-coded in the dataset file): FlyingThings3D, Monkaa, Driving and Sintel are under `/media/lh/lh4t/flow3d_data/`, and KITTI and VKITTI2 are under `/home/lh/all_datasets/`.

## 5. The June 2025 runs

The dates are those of the first and last checkpoints. "Recorded" means the launch flags exist in the PyCharm run configurations or the shell history (see `configs/pycharm_run_configs_2025.txt`). "Inferred" means they come from the step count of the saved checkpoints and the script defaults.
The parents in the last column come from comparing each run's 10k-step checkpoint with the possible parent checkpoints. A run started from scratch looks like every other from-scratch run at 10k steps, because they all start from the same initial weights. A fine-tuned run is close to its parent.

| Run (`--name`) | Script, stage | Steps | Other flags | Dates | Started from |
|---|---|---|---|---|---|
| `Sintel_pre` | `train_flow3d_ab.py --ab 8`, `pretrain` | 200k | `--batch_size 4 --gpus 0 1 --num_steps 200000` (recorded) | 06-03 → 06-04 | scratch |
| `RFlow3D_abv8` | `train_flow3d_ab.py --ab 8`, `abtrain` | 100k | `--batch_size 4 --lr 0.0002 --gpus 0 --image_size 320 640 --gamma 0.85` (recorded) | 06-17 → 06-18 | scratch |
| **`RFlow3D_DRIVING_train_finting`** | `train_flow3duvd.py`, `driving` (inferred from the name; `fineting` has the same data) | 160k | script defaults: batch 6 on 2 GPUs, crop 320 × 832, LR 2.5e-4 (inferred: 16 checkpoints = the 160k default) | 06-17 18:40 → 06-19 00:14 | scratch, despite "finting" in the name |
| `RFlow3D_Sintel_train_PRE` | `train_flow3duvd.py`, stage not recorded | 200k | `--num_steps 200000` (inferred) | 06-19 → 06-21 | scratch |
| `RFlow3D_Sintel_train_fine` | `train_flow3duvd.py`, stage not recorded | stopped at 70k | `--restore_ckpt RFlow3D_Sintel_train_PRE.pth` (inferred) | 06-21 | `RFlow3D_Sintel_train_PRE` |
| `RFlow3D_Sintel_CamLi` | `train_flow3duvd.py`, `Sintel_CamLi` | 200k | `--stage Sintel_CamLi --validation kitti` (recorded, under the run name `RFlow3D_Sintel_train_finting`) + `--num_steps 200000` (inferred) | 06-22 → 06-23 | scratch |

`RFlow3D_all_mixed`, the conference `[all_mix]` checkpoint, was trained after the submission (it is dated 06-30). It used `train_flow3duvd.py --stage pretrain` in a later copy of the project, whose `pretrain` mix has not been kept, so it is left out of this table.

## 6. Which checkpoint produced the submission

The submission script loads a single `--model` checkpoint into `RFlow3D_MSab_v8`, and the run that wrote `submit_preRF3D627` was not logged. **`RFlow3D_DRIVING_train_finting.pth` is the most likely one**:
- It is the only June 2025 SEAFLOW3D run trained on all of KITTI. `Sintel_CamLi`, `Sintel_pre` and `Sintel_train_PRE` are Things/Sintel models, and `RFlow3D_abv8` saw only half of KITTI.
- A PyCharm run configuration of `RFlow3Dtest.py` points at it (`--model=/home/lh/CSCV_occ/checkpoints/RFlow3D_DRIVING_train_finting.pth`).
- The conference README offered two checkpoints, `[all_mix]` (`RFlow3D_all_mixed.pth`) and `[kitti]`.

This was not confirmed by inference. To confirm it, run `kitti_sceneflow_submission` with this checkpoint and compare the foreground flow (pixels the refinement does not touch) with the PNGs in the release zip. They should agree up to the 1/64 px quantisation of the KITTI format.

## Reconstructed commands

See [`configs/train_runs_2025-06.sh`](configs/train_runs_2025-06.sh).
