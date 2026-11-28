#!/usr/bin/env bash
# The steps that produced submit_preRF3D627.zip on 2025-06-27, in order (see ../README.md).
# Paths are those of the training server. Not meant to be run as one script.
set -e
KITTI=/home/lh/all_datasets/kitti          # testing/image_2, testing/disp_ganet_testing, testing/calib_cam_to_cam
OUT=/home/lh/CSCV_occ_new/submit_preRF3D627

# 1-2) Network output. In RFlow3Dtest.py, call kitti_sceneflow_submission(model.module) in __main__
#      (it is commented out in this copy) and set its output folders to $OUT. Then run:
python RFlow3Dtest.py --model=/home/lh/CSCV_occ/checkpoints/RFlow3D_DRIVING_train_finting.pth   # checkpoint: most likely, see TRAINING.md
#      -> $OUT/disp_0 (GA-Net D1 x 256), $OUT/disp_1 (D1 + predicted change, x 256), $OUT/flow (KITTI PNG)

# 3) Keep the raw output for the refinement.
mv $OUT/flow   $OUT/flow_initial
mv $OUT/disp_1 $OUT/disp_1_initial

# 4) MSeg background masks (universal taxonomy, mseg-3m, 360 multi-scale), from the mseg-semantic folder:
python -u mseg_semantic/tool/universal_demo.py \
    --config=mseg_semantic/config/test/default_config_360_ms.yaml \
    model_name mseg-3m model_path /path/to/mseg-3m.pth \
    input_file $KITTI/testing/image_2
#      -> temp_files/mseg-3m_image_2_universal_ms/360/gray/*.png

# 5) Rigid background refinement, from the CamLiFlow clone (with the three refine_utils.py edits in ../README.md):
python refine_background.py            # defaults: --dataset_dir /home/lh/kitti/ --dataset_split testing (for calib_cam_to_cam)
#      reads disp_0, disp_1_initial, flow_initial; writes the refined $OUT/flow and $OUT/disp_1

# 6) Zip with disp_0/, disp_1/, flow/ at the top level and upload to KITTI.
cd $OUT && zip -r ../submit_preRF3D627.zip disp_0 disp_1 flow
