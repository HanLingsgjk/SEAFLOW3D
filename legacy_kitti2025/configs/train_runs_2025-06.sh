#!/usr/bin/env bash
# Reconstructed training commands of the June 2025 SEAFLOW3D runs (see ../TRAINING.md, section 5).
# They ran from the project folder /home/lh/CSCV_occ on the training server, with the files in ../code
# on the import path, and wrote checkpoints to ./checkpoints.
# Tags: RECORDED = flags found in the shell history or the PyCharm run configurations;
#       INFERRED = taken from the step count of the saved checkpoints and the script defaults.
# Not meant to be run as one script; each block is one run.
set -e

# --- The KITTI model (most likely the one behind the 2025-06-27 submission) --------------------------
# RFlow3D_DRIVING_train_finting, 2025-06-17 18:40 -> 06-19 00:14, 16 checkpoints = 160k steps, from scratch.
# INFERRED: the launch command was not kept. The stage follows the run name (`fineting` has the same data);
# the remaining flags are the train_flow3duvd.py defaults of that time, written out here.
# Data: VKITTI2 + 200x KITTI (all 200 training pairs, MonSter input disparities) + Driving.
CUDA_VISIBLE_DEVICES=0,1 python train_flow3duvd.py --name RFlow3D_DRIVING_train_finting \
    --stage driving --validation kitti \
    --gpus 0 1 --batch_size 6 --image_size 320 832 \
    --lr 0.00025 --num_steps 160000 --wdecay 0.0001 --gamma 0.85

# --- FlyingThings3D model for Sintel generalisation (CamLiFlow protocol) ------------------------------
# RFlow3D_Sintel_CamLi, 2025-06-22 -> 06-23, 200k steps, from scratch.
# RECORDED: --stage Sintel_CamLi --validation kitti (PyCharm config "RFlow3D_Sintel_train_finting", CUDA_VISIBLE_DEVICES=2,3)
# INFERRED: --num_steps 200000; other flags at their defaults.
CUDA_VISIBLE_DEVICES=2,3 python train_flow3duvd.py --name RFlow3D_Sintel_CamLi \
    --stage Sintel_CamLi --validation kitti --num_steps 200000

# --- Things/Monkaa pretraining with the ablation trainer ----------------------------------------------
# Sintel_pre, 2025-06-03 -> 06-04, 200k steps, from scratch.  RECORDED (shell history).
CUDA_VISIBLE_DEVICES=0,1 python train_flow3d_ab.py --batch_size 4 --gpus 0 1 --name Sintel_pre \
    --stage pretrain --num_steps 200000 --ab 8

# --- Ablation run of SEAFLOW3D (--ab 8) ----------------------------------------------------------------
# RFlow3D_abv8, 2025-06-17 -> 06-18, 100k steps.  RECORDED (shell history).
# Data: Driving + 50x KITTIab (100 odd-index pairs) + 3x Sintel clean + 3x Sintel final (minus 5 validation scenes).
CUDA_VISIBLE_DEVICES=1 python train_flow3d_ab.py --name RFlow3D_abv8 --ab 8 --stage abtrain \
    --batch_size 4 --lr 0.0002 --num_steps 100000 --gpus 0 --image_size 320 640 --gamma=0.85

# --- Sintel runs whose stage was not recorded ------------------------------------------------------------
# RFlow3D_Sintel_train_PRE, 2025-06-19 -> 06-21, 200k steps, from scratch.  INFERRED: --num_steps 200000.
# RFlow3D_Sintel_train_fine, 2025-06-21, stopped after 70k steps, fine-tuned from RFlow3D_Sintel_train_PRE.pth
#   (INFERRED from the weights: its 10k checkpoint is close to RFlow3D_Sintel_train_PRE.pth).
# python train_flow3duvd.py --name RFlow3D_Sintel_train_PRE  --stage <not recorded> --num_steps 200000
# python train_flow3duvd.py --name RFlow3D_Sintel_train_fine --stage <not recorded> --restore_ckpt checkpoints/RFlow3D_Sintel_train_PRE.pth
