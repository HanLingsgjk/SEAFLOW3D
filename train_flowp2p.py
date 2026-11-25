"""SEA-Flow3D 训练入口（2026-11 起用它；发布的两份权重就是用这个脚本训的，命令见 README 的 Training 一节）
只保留 SEA-Flow3D 一个模型（model_home/SEAFlow3D.py，已修正隐状态迭代）；数据集路径都从命令行给
"""
from __future__ import print_function, division
import sys

sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_home.SEAFlow3D import SEAFLOW3D
import Dataset_home.dataset_3dflowuvd as datasets
import flowp2p_validation as evaluate
from torch.utils.tensorboard import SummaryWriter

MODEL_REGISTRY = {
    "seaflow3d": SEAFLOW3D,
}

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 50
VAL_FREQ = 200


def sequence_loss(flow_preds, dz_preds, flow_gt,dz_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_flow_predictions = len(flow_preds)
    n_dz_predictions = len(dz_preds)
    flow_loss = 0.0
    dz_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    validd = torch.isfinite(dz_gt[:, 0]) & (dz_gt[:, 0].abs() < 1000)

    for i in range(n_flow_predictions):
        i_weight = gamma ** (n_flow_predictions - i - 1)
        f_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * f_loss).mean()

    for i in range(n_dz_predictions):
        i_weight = gamma ** (n_dz_predictions - i - 1)
        d_loss = (dz_preds[i] - dz_gt).abs()
        dz_loss += i_weight * (valid[:, None] * validd[:, None] * d_loss).mean()

    epe1 = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe1 = epe1.view(-1)[valid.view(-1)]

    epe2 = torch.sum((dz_preds[-1] - dz_gt) ** 2, dim=1).sqrt()
    epe2 = epe2.view(-1)[(valid*validd).view(-1)]

    metrics = {
        '6epe': epe1.mean().item(),
        '7epe': epe2.mean().item(),
        '1px': (epe1 < 1).float().mean().item(),
        '3px': (epe1 < 3).float().mean().item(),
        '5px': (epe1 < 5).float().mean().item(),
    }
    return flow_loss+dz_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    def __init__(self, model, scheduler, log_dir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.last_time = time.time()
        self.log_dir = log_dir
    def _print_training_status(self):
        now_time = time.time()
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        time_str = ("time = %.2f, " % (now_time - self.last_time))

        self.last_time = now_time
        # print the training status
        print(training_str + metrics_str+time_str)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)
    def close(self):
        if self.writer is not None:
            self.writer.close()

def train(args):

    primary_device = torch.device('cuda:%d' % args.gpus[0])
    torch.cuda.set_device(args.gpus[0])

    model_cls = MODEL_REGISTRY[args.model]
    model = nn.DataParallel(model_cls(args).to(primary_device), device_ids=args.gpus)
    print("Model key: %s" % args.model)
    print("Model class: %s" % model_cls.__name__)
    print("Parameter Count: %d" % count_parameters(model))
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt, map_location='cpu'), strict=False)
    model.train()
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, log_dir=os.path.join(args.log_dir, args.name))
    val_freq = int(args.val_freq)
    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, d1,d2,dd1,intrinsics, valid= [
                x.to(primary_device, non_blocking=True) for x in data_blob
            ]
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn_like(image1)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn_like(image2)).clamp(0.0, 255.0)

            flow2d_predictions,dz_predictions = model(image1, image2,d1,d2, intrinsics=intrinsics, iters=args.iters)

            loss, metrics = sequence_loss(flow2d_predictions,dz_predictions, flow,dd1, valid, args.gamma)


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if val_freq > 0 and total_steps % val_freq == val_freq - 1:
                PATH = os.path.join(args.output_dir, '%d_%s.pth' % (total_steps + 1, args.name))
                torch.save(model.state_dict(), PATH)
                results = {}
                with torch.no_grad():
                    if 'sintel' in args.validation:
                        results.update(evaluate.validate_sintel(
                            model.module,
                            root=args.sintel_root,
                            device=primary_device,
                            iters=args.iters,
                        ))
                    if 'kitti' in args.validation:
                        results.update(evaluate.validate_kitti(
                            model.module,
                            root=args.kitti_root,
                            device=primary_device,
                            iters=args.iters,
                        ))
                if results:
                    logger.write_dict(results)
                model.train()
                torch.cuda.empty_cache()
            total_steps += 1

            if total_steps >= args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = os.path.join(args.output_dir, '%s.pth' % args.name)
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='abvs', help="name your experiment")
    parser.add_argument('--stage',default='kitti', help="determines which dataset to use for training")
    parser.add_argument('--model', default='seaflow3d', choices=sorted(MODEL_REGISTRY),
                        help="model variant to train")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--model_type', default='ccmr')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', default=['kitti'], choices=['sintel', 'kitti'],
                        type=str, nargs='*', help="validation datasets; pass no values to disable")
    parser.add_argument('--val_freq', type=int, default=2000,
                        help="checkpoint/validation interval; 0 disables periodic validation")
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--log_dir', default='runs')

    parser.add_argument('--things_root', default='datasets/FlyingThings3D')
    parser.add_argument('--driving_root', default='datasets/Driving')
    parser.add_argument('--sintel_root', default='datasets/Sintel')
    parser.add_argument('--kitti_root', default='datasets/KITTI')
    parser.add_argument('--kitti_ab_root', default=None,
                        help="KITTI root used by abtrain; defaults to --kitti_root")
    parser.add_argument('--vkitti_root', default='datasets/VKITTI2')

    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 640])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--data_seed', type=int, default=1234,
                        help='seed used by the DataLoader sampler')

    parser.add_argument('--iters', type=int, default=6,
                        help='legacy forward argument; current models use it1/it2/it3 or fixed loops')
    parser.add_argument('--itr1', type=int, default=6)
    parser.add_argument('--itr2', type=int, default=4)
    parser.add_argument('--it1', type=int, default=2)
    parser.add_argument('--it2', type=int, default=4)
    parser.add_argument('--it3', type=int, default=2)

    parser.add_argument('--f_iter', type=int, default=0)
    parser.add_argument('--wdecay', type=float, default=.0001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if args.num_steps < 1:
        parser.error("--num_steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if len(args.image_size) != 2 or min(args.image_size) < 1:
        parser.error("--image_size requires exactly two positive integers: HEIGHT WIDTH")
    if min(args.it1, args.it2, args.it3) < 1:
        parser.error("--it1, --it2 and --it3 must all be at least 1")
    if args.val_freq < 0:
        parser.error("--val_freq must be non-negative")
    if args.itr1 != 6 or args.itr2 != 4:
        parser.error("seaflow3d currently requires --itr1 6 --itr2 4")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)
