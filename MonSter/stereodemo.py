from __future__ import print_function, division
import sys
import os
import argparse
import logging
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(THIS_DIR, 'core')
DEPTH_ANYTHING_DIR = os.path.join(THIS_DIR, 'Depth-Anything-V2-list3')

for path in (CORE_DIR, DEPTH_ANYTHING_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.monster import Monster, autocast
import cv2
from core.utils.utils import InputPadder
import matplotlib.pyplot as plt

class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def encode_array_to_uint16s(float_array):
    # 确保输入是浮动数，并将输入转换为 numpy 数组
    float_array = np.array(float_array, dtype=np.float32)

    # 将浮动数归一化到 [0, 2^24-1] 范围
    quantized = np.clip(float_array, 0, 2 ** 24 - 1)  # 保证在 [0, 2^24) 范围内

    # 进行位拆分：高 16 位和低 16 位
    highs = (quantized.astype(np.uint32) >> 16) & 0xFFFF  # 高 16 位
    lows = quantized.astype(np.uint32) & 0xFFFF  # 低 16 位

    # 返回两个 uint16 数组
    return highs.astype(np.uint16), lows.astype(np.uint16)


def decode_array_from_uint16s(highs, lows):
    # 将两个 uint16 数组合并为一个 24 位的数
    combined = (highs.astype(np.uint32) << 16) | lows.astype(np.uint32)

    # 恢复浮动数
    restored = combined.astype(np.float32)

    return restored


def writedepth(filename, d1):
    d1 = d1.clip(0, 512)
    d1 = 1024.0 * d1  # 值域为0-32
    d1u, d1d = encode_array_to_uint16s(d1)
    dddc = np.concatenate([d1u[:, :, np.newaxis], d1d[:, :, np.newaxis], d1d[:, :, np.newaxis]], axis=-1).astype(
        np.uint16)
    cv2.imwrite(filename, dddc)


from core.utils import frame_utils
from glob import glob
import os.path as osp


@torch.no_grad()
def validate_demo(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    images1o = sorted(glob('/mnt/hdd/hanling/sceneflow_data/KITTI/testing/image_2/*'))
    images2o = sorted(glob('/mnt/hdd/hanling/sceneflow_data/KITTI/testing/image_3/*'))
    outdir = '/mnt/hdd/hanling/sceneflow_data/KITTI/testing/disp_monster_testing/'
    torch.backends.cudnn.benchmark = True
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(images1o.__len__()):
        frame_id = images1o[i].split('/')[-1].split('.')[0] + '.png'
        path_to_save = outdir + frame_id

        img0 = np.array(frame_utils.read_gen(images1o[i]))
        img1 = np.array(frame_utils.read_gen(images2o[i]))
        if img1.shape.__len__() < 3:
            img1 = \
            np.concatenate([img1[:, :, np.newaxis], img1[:, :, np.newaxis], img1[:, :, np.newaxis]], axis=-1).astype(
                np.uint8)[..., :3]
            img0 = \
            np.concatenate([img0[:, :, np.newaxis], img0[:, :, np.newaxis], img0[:, :, np.newaxis]], axis=-1).astype(
                np.uint8)[..., :3]
        else:
            img1 = img1.astype(np.uint8)[..., :3]
            img0 = img0.astype(np.uint8)[..., :3]

        img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()

        image1 = img0[None].cuda()
        image2 = img1[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            with autocast(enabled=mixed_prec):
                disp = model(image1, image2, iters=iters, test_mode=True)
        disp = padder.unpad(disp).cpu().squeeze(0)
        writedepth(path_to_save, disp[0])
        print(frame_id)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default="/data2/cjd/mono_fusion/checkpoints/sceneflow.pth")

    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow',
                        choices=["demo", "eth3d", "kitti", "sceneflow", "vkitti", "driving"] + [f"middlebury_{s}" for s
                                                                                                in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total number of parameters: {total_params:.2f}M")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of trainable parameters: {trainable_params:.2f}M")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        logging.info(args.restore_ckpt)
        assert os.path.exists(args.restore_ckpt)
        checkpoint = torch.load(args.restore_ckpt)
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            # ckpt['module.' + key] = checkpoint[key]
            if key.startswith("module."):
                ckpt[key] = checkpoint[key]  # 保持原样
            else:
                ckpt["module." + key] = checkpoint[key]  # 添加 "module."

        model.load_state_dict(ckpt, strict=True)

        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()
    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = args.corr_implementation.endswith("_cuda")
    validate_demo(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
