# KITTI 2015 scene flow submission of 2025-06-27 (conference version)

This folder is an archive of the pipeline that produced the SEAFLOW3D submission to the KITTI 2015 scene flow benchmark, uploaded on 2025-06-27.
That was before the 2026-11 fix and retraining described in the main README.

- **Training setup:** see [TRAINING.md](TRAINING.md).
- **Submitted results:** `submit_preRF3D627.zip` (`disp_0/`, `disp_1/`, `flow/`, 200 test pairs) is attached to the release
  [`kitti-submission-2025-06-27`](https://github.com/HanLingsgjk/SEAFLOW3D/releases/tag/kitti-submission-2025-06-27).
  It is the zip that was uploaded to KITTI (249,500,445 bytes, MD5 `49a7507c46a0a88e7e9455f8cd036215`), and its 600 PNGs are byte-identical to the result folder on the training server.

The code here was copied unchanged from the training server. The only change is a provenance header added to `refine_background.py`.
Absolute paths in the code are the original server paths. The files document what was run and will not run from this folder as is.
To run SEAFLOW3D today, use the code in the repository root (`train_flowp2p.py`, `kitti_submission.py`).

## Files

| File | Original location on the server | Last modified | Role |
|---|---|---|---|
| `code/model_home/RFlow3D_MSabv4_2.py` | `CSCV_occ/model_home/` | 2025-05-23 | the network (`RFlow3D_MSab_v8`), before the hidden-state fix |
| `code/train_flow3duvd.py` | `CSCV_occ/` | 2025-06-22 | trainer for the SEAFLOW3D runs |
| `code/train_flow3d_ab.py` | `CSCV_occ/` | 2025-06-22 | trainer for the ablations (`--ab`) |
| `code/Dataset_home/dataset_3dflowuvd.py` | `CSCV_occ/Dataset_home/` | 2025-06-22 | datasets and training stages, KITTI test loader (`kitti_testsubmit`) |
| `code/RFlow3Dtest.py` | `CSCV_occ/` | 2025-06-24 | validation and `kitti_sceneflow_submission()` |
| `code/refine_background.py` | `CamLiFlow-main/` (a local clone of CamLiFlow) | 2025-06-27 | rigid background refinement, modified from CamLiFlow |
| `configs/train_runs_2025-06.sh` | — | — | reconstructed training commands |
| `configs/submission_2025-06-27.sh` | — | — | the submission steps in order |
| `configs/pycharm_run_configs_2025.txt` | PyCharm `workspace.xml` | — | the run configurations found for these scripts |

## Submission pipeline

### 1. Disparities

Both frames use GA-Net disparities, read from `testing/disp_ganet_testing/*_10.png` and `*_11.png`.
These are the GA-Net test-set disparities released by RAFT-3D; the link is in the main README.
`kitti_testsubmit` reads them with `np.abs(...)` and applies no hole filling.

### 2. Network

`RFlow3Dtest.py::kitti_sceneflow_submission(model)` runs `RFlow3D_MSab_v8` on the 200 test pairs:
- Full resolution, padded to a multiple of 16 (`InputPadder(mode='kitti', sp=16)`), fp32, no test-time augmentation.
- `disp_0` is the GA-Net disparity of frame 1 (× 256, uint16).
- `disp_1` is the GA-Net disparity of frame 1 plus the predicted disparity change (× 256, uint16).
- `flow` is the predicted optical flow, written in the KITTI 16-bit format.

The checkpoint was most likely `RFlow3D_DRIVING_train_finting.pth` (see TRAINING.md, section 6).
In the copy kept here, the function writes to `/home/lh/CSCV_occ/submit_preRF3D619/`. For the 2025-06-27 run, the same function, with only the output folder changed, wrote to `/home/lh/CSCV_occ_new/submit_preRF3D627/`.

### 3. Keep the raw output

Rename `flow/` to `flow_initial/` and `disp_1/` to `disp_1_initial/`. `refine_background.py` reads these two folders and writes new `flow/` and `disp_1/`. `disp_0/` is not changed.

### 4. Background masks (MSeg, not included)

The background masks come from [MSeg](https://github.com/mseg-dataset/mseg-semantic). The universal-taxonomy model `mseg-3m` was run on the KITTI test images with the 360 multi-scale configuration:

```bash
# install mseg-api and mseg-semantic and download an mseg-3m checkpoint as described in the MSeg README
python -u mseg_semantic/tool/universal_demo.py \
    --config=mseg_semantic/config/test/default_config_360_ms.yaml \
    model_name mseg-3m model_path /path/to/mseg-3m.pth \
    input_file <kitti_root>/testing/image_2
# writes temp_files/mseg-3m_image_2_universal_ms/360/gray/<frame>_10.png and <frame>_11.png (one class id per pixel)
```

- The server holds both `mseg-3m-480p.pth` and `mseg-3m-720p.pth`, and which of the two produced the masks was not recorded.
- `refine_background.py` reads the `gray/` folder and uses only the `_10` frames.
- It treats 26 movable classes as foreground: the animal classes, the vehicles (car, bus, truck, train, motorcycle, bicycle, ...) and person / rider / bicyclist / motorcyclist. The `foreground` list in the script gives the exact classes, and every other pixel counts as background.

### 5. Rigid background refinement

[`code/refine_background.py`](code/refine_background.py) is a modified copy of CamLiFlow's `refine_background.py`.
For every test pair it does the following:
1. Builds the background mask from the MSeg map as described above.
2. Reads `K` from `testing/calib_cam_to_cam/<frame>.txt` (`P_rect_02`). The baseline is 0.54 m.
3. Builds a coarse occlusion mask from pixels whose flow leaves the image, plus the top 80 rows.
4. Calls `mod_flow` (CamLiFlow's `refine_utils.py`, which is adapted from RigidMask) to estimate the camera motion. The pixels used are background pixels with a positive disparity that are not in the occlusion mask. The motion is estimated with the essential matrix, then a PnP refinement on up to 50,000 best-aligned points.
5. If the mean background parallax is at least 8 px, replaces the flow and `disp_1` of **all** background pixels with the rigid solution. Otherwise the frame is left unchanged. Foreground pixels always keep the network output.
6. Clips the flow to ±500 and writes `flow/` and `disp_1/`.

The script imports `refine_utils.py` and `utils.py` from CamLiFlow. CamLiFlow has no license file, so these two files are not redistributed here.
Run the script from a clone of https://github.com/MCG-NJU/CamLiFlow.

The clone used in 2025 had three local edits to `refine_utils.py`. Apply them to upstream CamLiFlow, or the call in `refine_background.py` will fail on the extra `noc_mask` argument:
1. `def mod_flow(bg_mask, disp, disp_change, flow, K0, K1, bl, occ_mask, noc_mask, parallax_th=8)`. Inside it, the pixels for the pose fit are `valid_mask = np.logical_and(disp > 0, np.logical_and(bg_mask, noc_mask)).flatten()`; upstream uses `occ_mask` here. With upstream code, passing this script's `noc_mask` as `occ_mask` has the same effect.
2. `F_ngransac(..., cv=True, ...)` as the default, so the essential matrix comes from OpenCV's `findEssentialMat` and NG-RANSAC is not needed.
3. In the loop of `F_ngransac`, `ratios = ratios[i:i + 1, fmask, :2].cpu()` was added after the `pts2 = ...` line.

### 6. Zip

Zip `disp_0/`, `disp_1/` and `flow/` at the top level of the archive and upload it.

## Differences from `kitti_submission.py` in the repository root (2026-11)

| | June 2025 (this folder) | 2026-11 (`kitti_submission.py`) |
|---|---|---|
| Weights | trained with the hidden-state bug, `RFlow3D_DRIVING_train_finting` (most likely) | retrained, `seaflow3d_kitti.pth` |
| D1 | GA-Net disparity as read | GA-Net with holes filled like the devkit's `interpolateBackground` |
| D2 | D1 + predicted disparity change | D1 / τ |
| Background | MSeg universal map without the 26 movable classes | DDR-Net static classes (label ≤ 22) eroded by 5 px |
| Pixels excluded from the pose fit | flow leaving the image + top 80 rows | flow leaving the image (as in CamLiFlow) |
