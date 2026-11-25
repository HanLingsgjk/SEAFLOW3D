"""Lightweight Sintel/KITTI validation for the training entry.

This module intentionally does not import the legacy RFlow3Dtest module or
MonSter. Dataset roots are explicit so importing the trainer has no external
weight-loading side effects.
"""

import inspect
import time

import numpy as np
import torch

import Dataset_home.dataset_3dflowuvd as datasets
from core.utils.utils import InputPadder


def _model_accepts_intrinsics(model):
    try:
        target = model.module if hasattr(model, "module") else model
        return "intrinsics" in inspect.signature(target.forward).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def _pad_intrinsics(intrinsics, padder, device):
    if intrinsics is None:
        return None
    intrinsics = torch.as_tensor(
        intrinsics, dtype=torch.float32, device=device
    ).reshape(1, 4).clone()
    pad_left, _, pad_top, _ = padder._pad
    intrinsics[:, 2] += float(pad_left)
    intrinsics[:, 3] += float(pad_top)
    return intrinsics


def _run_model(model, image1, image2, d1, d2, intrinsics, iters):
    kwargs = {"iters": iters, "test_mode": True}
    if intrinsics is not None and _model_accepts_intrinsics(model):
        kwargs["intrinsics"] = intrinsics
    return model(image1, image2, d1, d2, **kwargs)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(model, device):
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _validate(model, dataset, prefix, device=None, iters=6, sintel=False):
    if len(dataset) == 0:
        raise RuntimeError("%s validation dataset is empty" % prefix)

    device = _resolve_device(model, device)
    was_training = model.training
    model.eval()

    flow_epe_per_image = []
    dz_epe_per_image = []
    flow_out_sum = 0.0
    flow_pixel_count = 0
    dz_out_sum = 0.0
    dz_pixel_count = 0
    elapsed = 0.0

    try:
        for index in range(len(dataset)):
            image1, image2, flow_gt, d1, d2, dz_gt, intrinsics, valid_gt = dataset[index]

            image1 = image1[None].to(device)
            image2 = image2[None].to(device)
            d1 = d1[None].to(device)
            d2 = d2[None].to(device)

            padder = InputPadder(image1.shape, mode="kitti", sp=32)
            image1, image2 = padder.pad(image1, image2)
            d1, d2 = padder.pad(d1, d2)
            intrinsics_in = _pad_intrinsics(intrinsics, padder, device)

            _sync(device)
            started = time.time()
            _, flow_pred, dz_pred = _run_model(
                model, image1, image2, d1, d2, intrinsics_in, iters
            )
            _sync(device)
            elapsed += time.time() - started

            flow_pred = padder.unpad(flow_pred)[0].float().cpu()
            dz_pred = padder.unpad(dz_pred)[0].float().cpu()
            flow_gt = flow_gt.float().cpu()
            dz_gt = dz_gt.float().cpu()
            valid_gt = valid_gt.bool().cpu()

            flow_error = torch.linalg.vector_norm(flow_pred - flow_gt, dim=0)
            flow_magnitude = torch.linalg.vector_norm(flow_gt, dim=0)
            flow_valid = (
                valid_gt
                & torch.isfinite(flow_error)
                & torch.isfinite(flow_magnitude)
            )
            if flow_valid.any():
                flow_outlier = (
                    (flow_error > 3.0)
                    & (
                        flow_error
                        / flow_magnitude.clamp_min(1e-6)
                        > 0.05
                    )
                )
                flow_epe_per_image.append(flow_error[flow_valid].mean().item())
                flow_out_sum += flow_outlier[flow_valid].sum().item()
                flow_pixel_count += int(flow_valid.sum().item())

            dz_error = torch.linalg.vector_norm(dz_pred - dz_gt, dim=0)
            dz_magnitude = torch.linalg.vector_norm(dz_gt, dim=0)
            dz_valid = (
                torch.isfinite(dz_error)
                & torch.isfinite(dz_magnitude)
                & torch.isfinite(dz_gt[0])
                & (dz_gt[0].abs() < 1000)
            )
            if not sintel:
                dz_valid &= valid_gt
            if dz_valid.any():
                dz_outlier = (
                    (dz_error > 0.1)
                    & (
                        dz_error
                        / dz_magnitude.clamp_min(1e-6)
                        > 0.05
                    )
                )
                dz_epe_per_image.append(dz_error[dz_valid].mean().item())
                dz_out_sum += dz_outlier[dz_valid].sum().item()
                dz_pixel_count += int(dz_valid.sum().item())
    finally:
        model.train(was_training)

    flow_epe = float(np.mean(flow_epe_per_image)) if flow_epe_per_image else float("nan")
    dz_epe = float(np.mean(dz_epe_per_image)) if dz_epe_per_image else float("nan")
    flow_f1 = 100.0 * flow_out_sum / max(flow_pixel_count, 1)
    dz_f1 = 100.0 * dz_out_sum / max(dz_pixel_count, 1)
    mean_time = elapsed / max(len(dataset), 1)

    print(
        "Validation %s Flow: %.6f, %.6f; Dz: %.6f, %.6f; time_mean %.6f"
        % (prefix, flow_epe, flow_f1, dz_epe, dz_f1, mean_time)
    )
    key = prefix.lower()
    return {
        "%s-epe" % key: flow_epe,
        "%s-f1" % key: flow_f1,
        "%s-dz-epe" % key: dz_epe,
        "%s-dz-f1" % key: dz_f1,
        "%s-time" % key: mean_time,
    }


def validate_sintel(
    model,
    root,
    device=None,
    iters=6,
    split="training",
    typei="t_test",
    detype="final",
):
    dataset = datasets.Sintel_test(
        root=root,
        split=split,
        ifeval=typei,
        dstype=detype,
    )
    return _validate(
        model, dataset, "Sintel", device=device, iters=iters, sintel=True
    )


def validate_kitti(
    model,
    root,
    device=None,
    iters=6,
    split="training",
    sample_stride=2,
):
    dataset = datasets.kitti_test(
        root=root,
        split=split,
        spl=sample_stride,
    )
    return _validate(
        model, dataset, "KITTI", device=device, iters=iters, sintel=False
    )


__all__ = ["validate_sintel", "validate_kitti"]
