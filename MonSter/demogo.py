from __future__ import print_function, division
import sys
import os
# 当前文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 项目根目录
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)


import os
import logging
import numpy as np
import torch
import cv2

from MonSter.core.monster import Monster, autocast
from MonSter.core.utils.utils import InputPadder
from MonSter.core.utils import frame_utils

# =========================
# 配置区：按你的实际情况改这里
# =========================
CKPT_PATH = "/mnt/hdd/home/linghan/mix_all.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "encoder": "vitl",
    "hidden_dims": [128, 128, 128],
    "corr_implementation": "reg",
    "shared_backbone": False,
    "corr_levels": 2,
    "corr_radius": 4,
    "n_downsample": 2,
    "slow_fast_gru": False,
    "n_gru_layers": 3,
    "max_disp": 192,
}

# 全局缓存模型
_MODEL = None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode_array_to_uint16s(float_array):
    float_array = np.array(float_array, dtype=np.float32)
    quantized = np.clip(float_array, 0, 2 ** 24 - 1)

    highs = (quantized.astype(np.uint32) >> 16) & 0xFFFF
    lows = quantized.astype(np.uint32) & 0xFFFF

    return highs.astype(np.uint16), lows.astype(np.uint16)


def decode_array_from_uint16s(highs, lows):
    combined = (highs.astype(np.uint32) << 16) | lows.astype(np.uint32)
    restored = combined.astype(np.float32)
    return restored


def writedepth(filename, d1):
    d1 = np.asarray(d1)
    d1 = d1.clip(0, 512)
    d1 = 1024.0 * d1

    d1u, d1d = encode_array_to_uint16s(d1)
    dddc = np.concatenate(
        [
            d1u[:, :, np.newaxis],
            d1d[:, :, np.newaxis],
            d1d[:, :, np.newaxis]
        ],
        axis=-1
    ).astype(np.uint16)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, dddc)


def load_image_as_tensor(img_path):
    img = np.array(frame_utils.read_gen(img_path))

    if img.ndim < 3:
        img = np.concatenate(
            [img[:, :, np.newaxis], img[:, :, np.newaxis], img[:, :, np.newaxis]],
            axis=-1
        ).astype(np.uint8)[..., :3]
    else:
        img = img.astype(np.uint8)[..., :3]

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def _build_args():
    class Args:
        pass

    args = Args()
    args.restore_ckpt = CKPT_PATH
    args.encoder = MODEL_CONFIG["encoder"]
    args.hidden_dims = MODEL_CONFIG["hidden_dims"]
    args.corr_implementation = MODEL_CONFIG["corr_implementation"]
    args.shared_backbone = MODEL_CONFIG["shared_backbone"]
    args.corr_levels = MODEL_CONFIG["corr_levels"]
    args.corr_radius = MODEL_CONFIG["corr_radius"]
    args.n_downsample = MODEL_CONFIG["n_downsample"]
    args.slow_fast_gru = MODEL_CONFIG["slow_fast_gru"]
    args.n_gru_layers = MODEL_CONFIG["n_gru_layers"]
    args.max_disp = MODEL_CONFIG["max_disp"]
    return args


def _load_model_once():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    logging.info("Loading Monster model...")
    args = _build_args()

    if DEVICE.startswith("cuda"):
        model = torch.nn.DataParallel(Monster(args), device_ids=[0])
    else:
        model = Monster(args)

    assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    ckpt = {}
    has_module_prefix = isinstance(model, torch.nn.DataParallel)

    for key in checkpoint:
        if has_module_prefix:
            if key.startswith("module."):
                ckpt[key] = checkpoint[key]
            else:
                ckpt["module." + key] = checkpoint[key]
        else:
            if key.startswith("module."):
                ckpt[key[len("module."):]] = checkpoint[key]
            else:
                ckpt[key] = checkpoint[key]

    model.load_state_dict(ckpt, strict=True)
    model.to(DEVICE)
    model.eval()

    print(f"The model has {count_parameters(model) / 1e6:.2f}M learnable parameters.")

    _MODEL = model
    return _MODEL


@torch.no_grad()
def run_on_pair(left_path, right_path, save_path=None, iters=32, mixed_prec=None):
    """
    输入左右图路径，返回视差图 numpy 数组。

    Args:
        left_path: 左图路径
        right_path: 右图路径
        save_path: 若给定则保存结果
        iters: 迭代次数
        mixed_prec: 是否混合精度。默认自动根据 corr_implementation 判断
    Returns:
        disp_np: [H, W] numpy.ndarray
    """
    model = _load_model_once()

    if mixed_prec is None:
        mixed_prec = MODEL_CONFIG["corr_implementation"].endswith("_cuda")

    img0 = load_image_as_tensor(left_path)
    img1 = load_image_as_tensor(right_path)

    image1 = img0[None].to(DEVICE)
    image2 = img1[None].to(DEVICE)

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with autocast(enabled=mixed_prec):
        disp = model(image1, image2, iters=iters, test_mode=True)

    disp = padder.unpad(disp).cpu()

    # 常见情况：[B, 1, H, W]
    if disp.ndim == 4:
        disp_np = disp[0, 0].numpy()
    elif disp.ndim == 3:
        disp_np = disp[0].numpy()
    else:
        raise RuntimeError(f"Unexpected disp shape: {disp.shape}")

    if save_path is not None:
        writedepth(save_path, disp_np)

    return disp_np


# 可选：模块导入时就预加载
try:
    _load_model_once()
    print("Model preloaded successfully.")
except Exception as e:
    print(f"Model preload failed: {e}")
#run_on_pair('/mnt/hdd/hanling/KITTI/testing/image_2/000000_10.png','/mnt/hdd/hanling/KITTI/testing/image_3/000000_10.png')
