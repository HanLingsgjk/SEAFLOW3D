"""SEA-Flow3D 的 KITTI 2015 提交包（2026-11）
默认流程（榜单口径）：
  1. 网络输入：两帧左图 + 两帧 GA-Net 视差（RAFT-3D 公开的测试集视差），原分辨率、fp32、不做 TTA
  2. D1 = GA-Net 视差（按 devkit 的 interpolateBackground 补满无效像素），D2 = D1 / tau（tau = Z2/Z1 = d1/(d1 + dz)，
     dz 是网络输出的第二帧视差减第一帧视差；换算不了的像素记 tau = 1），光流直接用网络输出
  3. 默认再做 CamLiFlow 的刚性背景优化（--no_refine 关掉）：背景 = DDR-Net 语义分割的静态类（label <= 22），
     背景掩码先向内收 --erode 个像素（默认 5，避开分割在物体边缘的错），再用 CamLiFlow 的 refine_utils.mod_flow
     拟合相机运动，把背景像素的光流和 D2 换成刚体解（D1 不动）；本质矩阵走它代码里现成的 OpenCV 分支，不需要装 NG-RANSAC
  4. 打两个 zip：kitti_sceneflow2015.zip（disp_0 / disp_1 / flow）和 kitti_flow2015.zip（flow），两个 zip 里的 flow 相同
刚性背景优化另外需要（本仓库不附带）：CamLiFlow 仓库（https://github.com/MCG-NJU/CamLiFlow，用它的 refine_utils.py）、
它 README 里给的 DDR-Net 语义分割 semantic_ddr.zip、KITTI 官方标定 data_scene_flow_calib.zip 里的 calib_cam_to_cam
用法：python kitti_submission.py --ckpt checkpoints/seaflow3d_kitti.pth --kitti_root datasets/KITTI --camliflow_dir third_party/CamLiFlow --out submission
"""
import argparse
import importlib.util
import os
import shutil
import sys
import zipfile
from types import SimpleNamespace

sys.path.append('core')

import cv2
import numpy as np
import torch
from PIL import Image

from core.utils.utils import InputPadder
from model_home.SEAFlow3D import SEAFLOW3D

NAMES = ['%06d_10.png' % j for j in range(200)]


def read_image(path):
    return torch.from_numpy(np.array(Image.open(path)).astype(np.uint8)).permute(2, 0, 1).float()


def read_disp_png(path):
    # KITTI 格式视差 PNG：uint16，值 / 256，0 = 无效
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0


def read_flow_png(path):
    # KITTI 格式光流 PNG：16 位 RGB 依次为 u、v、有效位，(值 - 2^15) / 64
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32)
    return (raw[..., :2] - 32768.0) / 64.0


def write_flow_png(path, flow):
    assert np.isfinite(flow).all()
    uv = np.clip(np.round(flow.astype(np.float64) * 64.0 + 2 ** 15), 0, 65535)
    image = np.concatenate([uv, np.ones_like(uv[..., :1])], -1).astype(np.uint16)
    cv2.imwrite(path, image[..., ::-1])


def write_disp_png(path, disp):
    assert np.isfinite(disp).all()
    cv2.imwrite(path, np.clip(np.round(disp.astype(np.float64) * 256.0), 1, 65535).astype(np.uint16))


def fill_background(disp):
    # 照 devkit io_disp.h 的 interpolateBackground：每行无效段取两端较小值，行首行尾往外延伸，全无效的行用上下最近的有效行
    d = disp.copy()
    valid = d > 0
    for v in np.nonzero(~valid.all(1))[0]:
        idx = np.nonzero(valid[v])[0]
        if len(idx) == 0:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1:
                d[v, a + 1:b] = min(d[v, a], d[v, b])
        d[v, :idx[0]] = d[v, idx[0]]
        d[v, idx[-1] + 1:] = d[v, idx[-1]]
    rows = np.nonzero((d > 0).all(1))[0]
    d[:rows[0]] = d[rows[0]]
    d[rows[-1] + 1:] = d[rows[-1]]
    assert (d > 0).all(), 'no valid disparity left in a whole row after interpolation'
    return d


def load_model(ckpt):
    # 权重是 DataParallel 存的，去掉 module. 前缀后按层名严格加载；SEAFLOW3D 的迭代次数写死在 forward 里（1/16 两轮、1/8 四轮、1/4 三轮）
    model = SEAFLOW3D(SimpleNamespace(mixed_precision=False, itr1=6, itr2=4, it1=2, it2=4, it3=2))
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}, strict=True)
    return model.cuda().eval()


@torch.no_grad()
def predict(model, image_dir, disp_dir, name):
    # 返回光流 (H, W, 2) 和 tau (H, W)；网络吃的是没补过的原始视差
    image1 = read_image(os.path.join(image_dir, name))
    image2 = read_image(os.path.join(image_dir, name.replace('_10.png', '_11.png')))
    d1 = torch.from_numpy(read_disp_png(os.path.join(disp_dir, name)))[None]
    d2 = torch.from_numpy(read_disp_png(os.path.join(disp_dir, name.replace('_10.png', '_11.png'))))[None]
    inputs = [x[None].cuda() for x in (image1, image2, d1, d2)]
    padder = InputPadder(inputs[0].shape, mode='kitti', sp=32)
    p1, p2 = padder.pad(inputs[0], inputs[1])
    q1, q2 = padder.pad(inputs[2], inputs[3])
    _, flow, dz = model(p1, p2, q1, q2, test_mode=True)
    flow = padder.unpad(flow)[0].float().permute(1, 2, 0).cpu().numpy()
    dz = padder.unpad(dz)[0, 0].float().cpu().numpy()
    disp1 = d1[0].numpy()
    convertible = (disp1 > 0) & (disp1 + dz > 0)
    tau = np.where(convertible, disp1 / np.maximum(disp1 + dz, 1e-6), 1.0).astype(np.float32)
    return flow, tau


def load_refine_utils(camliflow_dir):
    # 直接用 CamLiFlow 仓库里的 refine_utils.py（它出自 RigidMask）；pose_estimate 按模块里的名字调 F_ngransac，这里换成它现成的 OpenCV 分支（cv=True）
    spec = importlib.util.spec_from_file_location('camliflow_refine_utils', os.path.join(camliflow_dir, 'refine_utils.py'))
    refine_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refine_utils)
    ngransac = refine_utils.F_ngransac
    refine_utils.F_ngransac = lambda *a, **k: ngransac(*a, **{**k, 'cv': True})
    return refine_utils


def load_k(calib_dir, name):
    # 左彩色相机（image_2）的内参：KITTI 标定文件里 P_rect_02 的前 3x3
    with open(os.path.join(calib_dir, name.replace('_10.png', '.txt'))) as f:
        for line in f:
            if line.startswith('P_rect_02'):
                return np.array([float(x) for x in line.split()[1:]], dtype=np.float32).reshape(3, 4)[:, :3]
    raise RuntimeError('no P_rect_02 in calib of %s' % name)


def refine_frame(refine_utils, src, dst, semantic_dir, calib_dir, erode, name):
    # 照 CamLiFlow 的 refine_background.py：读写好的 PNG，背景换成刚体解；刚体解里个别不是有限值或不是正的像素保留原值
    np.random.seed(0)
    disp = read_disp_png(os.path.join(src, 'disp_0', name))
    disp_c = read_disp_png(os.path.join(src, 'disp_1', name))
    flow = read_flow_png(os.path.join(src, 'flow', name))
    height, width = disp.shape
    bg_mask = cv2.imread(os.path.join(semantic_dir, name), 0) <= 22
    if erode > 0:
        bg_mask = cv2.erode(bg_mask.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    tx, ty = xs + flow[..., 0], ys + flow[..., 1]
    occ_mask = ~((tx < 0) | (tx > width) | (ty < 0) | (ty > height))   # 光流终点没出画面的像素（同 CamLiFlow 的 kitti_submission.py）
    K0 = load_k(calib_dir, name)
    flow_r, disp_c_r = refine_utils.mod_flow(bg_mask, disp, disp_c.copy(), flow.copy(), K0, K1=K0, bl=0.54, occ_mask=occ_mask)
    flow_r = np.clip(flow_r, -500, 500)
    bad_flow = ~np.isfinite(flow_r).all(-1)
    bad_disp = ~np.isfinite(disp_c_r) | (disp_c_r <= 0)
    flow_r[bad_flow] = flow[bad_flow]
    disp_c_r[bad_disp] = disp_c[bad_disp]
    write_flow_png(os.path.join(dst, 'flow', name), flow_r)
    shutil.copyfile(os.path.join(src, 'disp_0', name), os.path.join(dst, 'disp_0', name))
    write_disp_png(os.path.join(dst, 'disp_1', name), disp_c_r)
    return float((np.linalg.norm(flow_r - flow, axis=-1) > 1e-3).mean())


def make_zip(src, path, folders):
    expected = ['%s/%s' % (folder, name) for folder in folders for name in NAMES]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
        for item in expected:
            z.write(os.path.join(src, item), item)
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == expected and z.testzip() is None
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='KITTI fine-tuned weights, e.g. checkpoints/seaflow3d_kitti.pth')
    parser.add_argument('--kitti_root', default='datasets/KITTI', help='KITTI scene flow 2015 root (uses testing/image_2)')
    parser.add_argument('--disp_dir', default=None, help='GA-Net disparities of the test set (000000_10.png, 000000_11.png, ...); default <kitti_root>/testing/disp_ganet')
    parser.add_argument('--out', default='submission')
    parser.add_argument('--no_refine', action='store_true', help='skip the CamLiFlow rigid background refinement')
    parser.add_argument('--camliflow_dir', default='third_party/CamLiFlow', help='a clone of https://github.com/MCG-NJU/CamLiFlow (for refine_utils.py)')
    parser.add_argument('--semantic_dir', default=None, help='DDR-Net segmentation of the test set; default <kitti_root>/testing/semantic_ddr')
    parser.add_argument('--calib_dir', default=None, help='KITTI calibration of the test set; default <kitti_root>/testing/calib_cam_to_cam')
    parser.add_argument('--erode', type=int, default=5, help='shrink the background mask by this many pixels before refinement; 0 = original CamLiFlow')
    args = parser.parse_args()
    test_dir = os.path.join(args.kitti_root, 'testing')
    disp_dir = args.disp_dir or os.path.join(test_dir, 'disp_ganet')
    semantic_dir = args.semantic_dir or os.path.join(test_dir, 'semantic_ddr')
    calib_dir = args.calib_dir or os.path.join(test_dir, 'calib_cam_to_cam')
    network_dir = os.path.join(args.out, 'network')                       # 网络直接输出
    refined_dir = os.path.join(args.out, 'refined_bg_e%d' % args.erode)   # 刚性背景优化之后
    final_dir = network_dir if args.no_refine else refined_dir
    for folder in ('flow', 'disp_0', 'disp_1'):
        os.makedirs(os.path.join(network_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(final_dir, folder), exist_ok=True)

    model = load_model(args.ckpt)
    for j, name in enumerate(NAMES):
        flow, tau = predict(model, os.path.join(test_dir, 'image_2'), disp_dir, name)
        d1 = fill_background(read_disp_png(os.path.join(disp_dir, name)))
        assert d1.shape == tau.shape, name
        write_flow_png(os.path.join(network_dir, 'flow', name), flow)
        write_disp_png(os.path.join(network_dir, 'disp_0', name), d1)
        write_disp_png(os.path.join(network_dir, 'disp_1', name), d1 / tau)
        if j % 20 == 0:
            print('network %d / 200  %s  mean tau %.4f' % (j, name, tau.mean()), flush=True)

    if not args.no_refine:
        refine_utils = load_refine_utils(args.camliflow_dir)
        changed = [refine_frame(refine_utils, network_dir, refined_dir, semantic_dir, calib_dir, args.erode, name) for name in NAMES]
        print('rigid background refinement: %d / 200 frames changed, median changed-pixel fraction %.3f' % (
            sum(c > 0 for c in changed), float(np.median(changed))), flush=True)

    make_zip(final_dir, os.path.join(args.out, 'kitti_sceneflow2015.zip'), ['disp_0', 'disp_1', 'flow'])
    make_zip(final_dir, os.path.join(args.out, 'kitti_flow2015.zip'), ['flow'])


if __name__ == '__main__':
    main()
