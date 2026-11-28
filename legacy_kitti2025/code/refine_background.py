# Rigid background refinement used for the SEAFLOW3D KITTI 2015 submission of 2025-06-27.
# Modified from refine_background.py of CamLiFlow (https://github.com/MCG-NJU/CamLiFlow), which follows RigidMask.
# Changes from upstream: MSeg universal-taxonomy masks instead of DDR-Net (26 movable classes as foreground),
# a manually built mask of pixels excluded from the pose fit (flow leaving the image + top 80 rows), flow visualisation,
# and hard-coded paths of the training server. It imports refine_utils.py and utils.py from a CamLiFlow clone;
# see legacy_kitti2025/README.md for the three local edits that refine_utils.py needs.

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import torch
from refine_utils import mod_flow

from utils import load_disp_png, load_flow_png, save_flow_png, init_logging, save_disp_png, load_calib
import matplotlib.pyplot as plt
u_classes= ['backpack', 'umbrella', 'bag', 'tie', 'suitcase', 'case', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal_other', 'microwave', 'radiator', 'oven', 'toaster', 'storage_tank', 'conveyor_belt', 'sink', 'refrigerator', 'washer_dryer', 'fan', 'dishwasher', 'toilet', 'bathtub', 'shower', 'tunnel', 'bridge', 'pier_wharf', 'tent', 'building', 'ceiling', 'laptop', 'keyboard', 'mouse', 'remote', 'cell phone', 'television', 'floor', 'stage', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'fruit_other', 'food_other', 'chair_other', 'armchair', 'swivel_chair', 'stool', 'seat', 'couch', 'trash_can', 'potted_plant', 'nightstand', 'bed', 'table', 'pool_table', 'barrel', 'desk', 'ottoman', 'wardrobe', 'crib', 'basket', 'chest_of_drawers', 'bookshelf', 'counter_other', 'bathroom_counter', 'kitchen_island', 'door', 'light_other', 'lamp', 'sconce', 'chandelier', 'mirror', 'whiteboard', 'shelf', 'stairs', 'escalator', 'cabinet', 'fireplace', 'stove', 'arcade_machine', 'gravel', 'platform', 'playingfield', 'railroad', 'road', 'snow', 'sidewalk_pavement', 'runway', 'terrain', 'book', 'box', 'clock', 'vase', 'scissors', 'plaything_other', 'teddy_bear', 'hair_dryer', 'toothbrush', 'painting', 'poster', 'bulletin_board', 'bottle', 'cup', 'wine_glass', 'knife', 'fork', 'spoon', 'bowl', 'tray', 'range_hood', 'plate', 'person', 'rider_other', 'bicyclist', 'motorcyclist', 'paper', 'streetlight', 'road_barrier', 'mailbox', 'cctv_camera', 'junction_box', 'traffic_sign', 'traffic_light', 'fire_hydrant', 'parking_meter', 'bench', 'bike_rack', 'billboard', 'sky', 'pole', 'fence', 'railing_banister', 'guard_rail', 'mountain_hill', 'rock', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'net', 'base', 'sculpture', 'column', 'fountain', 'awning', 'apparel', 'banner', 'flag', 'blanket', 'curtain_other', 'shower_curtain', 'pillow', 'towel', 'rug_floormat', 'vegetation', 'bicycle', 'car', 'autorickshaw', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'trailer', 'boat_ship', 'slow_wheeled_object', 'river_lake', 'sea', 'water_other', 'swimming_pool', 'waterfall', 'wall', 'window', 'window_blind']
foreground = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal_other','bicycle', 'car', 'autorickshaw', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'trailer', 'boat_ship', 'slow_wheeled_object','person', 'rider_other', 'bicyclist', 'motorcyclist']
class_num=[]
for fore in foreground:
    class_num.append(u_classes.index(fore))

mesh_grid_cache = {}
def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image
def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def main(args):
    for i in tqdm(range(200)):
        np.random.seed(0)
        bg_mask = cv2.imread(os.path.join('/home/lh/mseg-semantic/temp_files/mseg-3m_image_2_universal_ms/360/gray/', '%06d_10.png' % i),0)
        #分离出前景
        for fore_id in class_num:
            bg_mask[bg_mask==fore_id]=240
        bg_mask = bg_mask<200

        #bg_masko = cv2.imread(os.path.join(args.dataset_dir, args.dataset_split, args.semantic_dir, '%06d_10.png' % i), 0) <= 22

        K0 = load_calib(os.path.join(args.dataset_dir, args.dataset_split, 'calib_cam_to_cam', '%06d.txt' % i))[0:3, 0:3]

        disp = load_disp_png('/home/lh/CSCV_occ_new/submit_preRF3D627/disp_0/%06d_10.png' % ( i))[0]
        disp_c = load_disp_png('/home/lh/CSCV_occ_new/submit_preRF3D627/disp_1_initial/%06d_10.png' % ( i))[0]
        flow = load_flow_png('/home/lh/CSCV_occ_new/submit_preRF3D627/flow_initial/%06d_10.png' % ( i))[0]
        #这里改成手动生成
        # generate a coarse occlusion mask for rigid background refinement
        input_h = bg_mask.shape[0]
        input_w = bg_mask.shape[1]
        grid = mesh_grid(1, input_h, input_w, device='cpu', channel_first=False)[0].numpy()
        grid_warp = grid + flow
        x_out = np.logical_or(grid_warp[..., 0] < 0, grid_warp[..., 0] > input_w)
        y_out = np.logical_or(grid_warp[..., 1] < 0, grid_warp[..., 1] > input_h)
        occ_mask = np.logical_or(x_out, y_out).astype(np.uint8)* 255
        occ_mask[:80,:]=255
        noc_mask = occ_mask==0
        occ_mask = occ_mask==255

        occupy_mask = np.logical_and(occ_mask, bg_mask).astype(np.uint8)==1

        flow_refine, disp_c_refine = mod_flow(bg_mask, disp, disp_c.copy(), flow, K0, K1=K0, bl=0.54, occ_mask=occupy_mask, noc_mask=noc_mask)
        flow_refine = np.clip(flow_refine, -500, 500)

        os.makedirs('/home/lh/CSCV_occ_new/submit_preRF3D627/flow', exist_ok=True)
        save_flow_png('/home/lh/CSCV_occ_new/submit_preRF3D627/flow/%06d_10.png' % ( i), flow_refine)

        output_dirshow = '/home/lh/CSCV_occ_new/submit_preRF3D627/anshow/flow_r'
        if not os.path.exists(output_dirshow):
            os.makedirs(output_dirshow)
        output_fileshow = os.path.join(output_dirshow, 'frame%04d.png' % (i + 1))
        flo = flow_to_image(flow_refine,convert_to_bgr=True)
        cv2.imwrite(output_fileshow, flo)
        print('frame%04d.png' % i)


        os.makedirs('/home/lh/CSCV_occ_new/submit_preRF3D627/disp_1', exist_ok=True)
        save_disp_png('/home/lh/CSCV_occ_new/submit_preRF3D627/disp_1/%06d_10.png' % ( i), disp_c_refine)
        print(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=False, default='/home/lh/kitti/')
    parser.add_argument('--dataset_split', required=False, default='testing')
    parser.add_argument('--semantic_dir', required=False, default='semantic_ddr')
    args = parser.parse_args()

    init_logging()
    main(args)
