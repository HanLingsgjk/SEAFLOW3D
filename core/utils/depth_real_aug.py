import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_sam_instance(sam):
    """Load a SAM instance map from an array or a .npz file."""
    if isinstance(sam, np.ndarray):
        return sam

    sam_path = os.fspath(sam)
    if sam_path.endswith(".npz"):
        data = np.load(sam_path, allow_pickle=True)
        if "instance" not in data:
            raise KeyError("SAM npz must contain an 'instance' array")
        return data["instance"]

    arr = np.asarray(Image.open(sam_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def read_sintel_depth(path):
    """Read Sintel .dpt depth data."""
    with open(path, "rb") as f:
        _ = np.fromfile(f, dtype=np.float32, count=1)[0]
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        depth = np.fromfile(f, dtype=np.float32, count=width * height)
    return depth.reshape(height, width)


def depth_to_pseudo_disparity(depth, focal_baseline=1000.0, eps=1e-6):
    """Convert depth to a disparity-like map for standalone Sintel demos."""
    depth = np.asarray(depth, dtype=np.float32)
    return focal_baseline / np.maximum(depth, eps)


def read_pfm(path):
    """Read a PFM disparity/depth file."""
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError("Not a PFM file: %s" % path)

        dims = f.readline().decode("ascii").strip().split()
        width, height = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = data.reshape(shape)
        data = np.flipud(data)
    return data.astype(np.float32)


def read_disparity(path):
    """Read a common disparity format for the demo entry point."""
    path = os.fspath(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".pfm":
        disp = read_pfm(path)
    elif suffix == ".npy":
        disp = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "disp" in data:
            disp = data["disp"]
        elif "disparity" in data:
            disp = data["disparity"]
        else:
            first_key = list(data.keys())[0]
            disp = data[first_key]
    elif suffix == ".dpt":
        disp = depth_to_pseudo_disparity(read_sintel_depth(path))
    else:
        # KITTI-style png disparity is commonly stored with a /256 scale.
        disp = np.asarray(Image.open(path), dtype=np.float32) / 256.0

    if disp.ndim == 3:
        disp = disp[..., 0]
    return np.asarray(disp, dtype=np.float32)


def augment_disparity_with_sam(
    sam_instance,
    disparity,
    rgb,
    valid=None,
    distance_map=None,
    rng=None,
    num_regions=(1, 4),
    region_sample_fraction=(0.05, 0.15),
    min_region_area=64,
    gaussian_kernels=(1, 2),
    percent_range=(0.01, 0.04),
    min_abs_delta=0.25,
    max_abs_delta=5.0,
    texture_amp_range=(0.10, 0.65),
    texture_blur_sigma=3.0,
    texture_dilate_radius=5,
    reject_bright_textured=True,
    bright_enough_luma_thresh=0.35,
    rich_texture_thresh=0.55,
    edge_prob=0.05,
    scatter_prob=0.25,
    scatter_count=(1, 2),
    scatter_strength=0.15,
    edge_small_area_ratio=0.010,
    edge_thinness_thresh=0.35,
    clip_to_positive=True,
    positive_eps=1e-3,
    return_debug=False,
):
    """Apply SAM-guided synthetic real-world disparity errors.

    Parameters
    ----------
    sam_instance:
        SAM instance id map, or a path to a SAM .npz containing key "instance".
        Id 0 is treated as unlabeled background and is not sampled.
    disparity:
        HxW float disparity map. Smaller positive disparity is treated as
        farther if distance_map is not provided.
    rgb:
        HxWx3 image used to score darkness and weak texture.
    valid:
        Optional HxW validity mask. Invalid pixels are left unchanged.
    distance_map:
        Optional HxW distance/depth map. Larger values are treated as farther.
    rng:
        Optional numpy random Generator, RandomState, or seed.
    min_region_area:
        SAM regions smaller than this many valid pixels are ignored. This only
        removes tiny fragments; large background regions are kept.
    region_sample_fraction:
        Extra density control. A random fraction of the candidate SAM regions
        is selected in addition to the absolute num_regions range.
    texture_amp_range:
        Multiplicative amplitude range as (strong_texture_scale,
        weak_texture_scale). Richer texture gets the smaller scale; weak
        texture gets the larger scale.
    texture_dilate_radius:
        Pixel radius used to dilate the strong-texture response before it
        suppresses perturbation amplitude.
    reject_bright_textured:
        If True, SAM regions with enough brightness and rich texture are not
        selected as depth-error blocks.

    Returns
    -------
    disp_aug or (disp_aug, debug)
        Augmented disparity, and optional debug metadata.
    """
    rng = _make_rng(rng)
    inst = load_sam_instance(sam_instance)
    disp = np.asarray(disparity, dtype=np.float32)
    image = _as_rgb_float(rgb)

    if disp.ndim != 2:
        raise ValueError("disparity must be HxW")

    height, width = disp.shape
    inst = _resize_instance_if_needed(inst, (height, width))
    image = _resize_rgb_if_needed(image, (height, width))

    if valid is None:
        valid_mask = np.isfinite(disp) & (disp != 0)
    else:
        valid_mask = np.asarray(valid).astype(bool)
        valid_mask = _resize_bool_if_needed(valid_mask, (height, width))
        valid_mask &= np.isfinite(disp)

    gray, grad = _gray_and_gradient(image)
    amp_map, distance_weight = _build_amplitude_map(
        disp,
        valid_mask,
        distance_map,
        percent_range,
        min_abs_delta,
        max_abs_delta,
    )
    amp_map, texture_amp_factor = _apply_texture_amplitude_factor(
        amp_map,
        grad,
        valid_mask,
        texture_amp_range,
        texture_blur_sigma,
        texture_dilate_radius,
        max_abs_delta,
    )

    regions = _score_regions(
        inst,
        valid_mask,
        gray,
        grad,
        min_region_area=min_region_area,
        reject_bright_textured=reject_bright_textured,
        bright_enough_luma_thresh=bright_enough_luma_thresh,
        rich_texture_thresh=rich_texture_thresh,
    )
    selected = _weighted_sample_regions(regions, num_regions, rng, region_sample_fraction)

    delta = np.zeros_like(disp, dtype=np.float32)
    region_debug = []

    for region in selected:
        mask = (inst == region["id"]) & valid_mask
        if not np.any(mask):
            continue

        before_nonzero = np.count_nonzero(delta)
        _add_low_texture_gaussian_noise(
            delta,
            mask,
            amp_map,
            gaussian_kernels,
            rng,
        )

        edge_applied = False
        edge_p = _edge_probability(region, edge_prob, edge_small_area_ratio, edge_thinness_thresh)
        if _rand(rng) < edge_p:
            _add_edge_expand_contract(delta, mask, valid_mask, amp_map, rng)
            edge_applied = True

        scatter_applied = False
        if _rand(rng) < scatter_prob:
            _add_scatter_blobs(delta, mask, amp_map, scatter_count, scatter_strength, rng)
            scatter_applied = True

        after_nonzero = np.count_nonzero(delta)
        region_debug.append(
            {
                "id": int(region["id"]),
                "area": int(region["area"]),
                "area_ratio": float(region["area_ratio"]),
                "thinness": float(region["thinness"]),
                "score": float(region["score"]),
                "mean_luma": float(region["mean_luma"]),
                "dark_score": float(region["dark_score"]),
                "weak_texture_score": float(region["weak_texture_score"]),
                "rich_texture_score": float(region["rich_texture_score"]),
                "edge_applied": bool(edge_applied),
                "scatter_applied": bool(scatter_applied),
                "changed_pixels_after_region": int(after_nonzero - before_nonzero),
            }
        )

    delta = np.clip(delta, -max_abs_delta, max_abs_delta)
    delta *= valid_mask.astype(np.float32)
    out = disp + delta
    out[~valid_mask] = disp[~valid_mask]

    if clip_to_positive:
        pos = valid_mask & (disp > 0)
        out[pos] = np.maximum(out[pos], positive_eps)

    out = np.asarray(out, dtype=np.float32)
    if not return_debug:
        return out

    debug = {
        "delta": delta,
        "selected_ids": [int(r["id"]) for r in selected],
        "regions": region_debug,
        "candidate_region_count": len(regions),
        "distance_weight": distance_weight,
        "texture_amp_factor": texture_amp_factor,
        "amplitude_map": amp_map,
    }
    return out, debug


def demo_sintel_temple3(
    root="datasets/Sintel/training",
    scene="temple_3",
    frame="frame_0015",
    out_path="depth_real_aug_demo.png",
    seed=7,
    focal_baseline=1000.0,
):
    """Small standalone demo using the Sintel/SAM directory layout."""
    root = Path(root)
    rgb_path = root / "final" / scene / (frame + ".png")
    sam_path = root / "sam" / "final" / scene / (frame + ".npz")
    depth_path = root / "depth" / scene / (frame + ".dpt")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = read_sintel_depth(depth_path)
    disp = depth_to_pseudo_disparity(depth, focal_baseline=focal_baseline)
    sam = load_sam_instance(sam_path)

    disp_aug, debug = augment_disparity_with_sam(
        sam,
        disp,
        rgb,
        distance_map=depth,
        rng=seed,
        return_debug=True,
    )

    save_demo_image(out_path, rgb, sam, disp, disp_aug, debug)
    return disp_aug, debug


def save_demo_image(out_path, rgb, sam_instance, disp, disp_aug, debug):
    """Save a visual comparison for quick inspection."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    inst = _resize_instance_if_needed(load_sam_instance(sam_instance), disp.shape)
    selected = set(debug.get("selected_ids", []))
    overlay = _selected_overlay(rgb, inst, selected)
    shared_range = _shared_percentile_range([disp, disp_aug], percentiles=(2, 98))
    disp_vis = _heatmap(disp, value_range=shared_range)
    aug_vis = _heatmap(disp_aug, value_range=shared_range)
    delta_vis = _heatmap(disp_aug - disp, center_zero=True, center_percentile=95)

    panels = [
        ("rgb", rgb),
        ("selected sam", overlay),
        ("disp", disp_vis),
        ("disp aug", aug_vis),
        ("delta", delta_vis),
    ]

    width = max(p.shape[1] for _, p in panels)
    row_h = max(p.shape[0] for _, p in panels)
    label_h = 24
    canvas = Image.new("RGB", (width * len(panels), row_h + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i, (name, panel) in enumerate(panels):
        im = Image.fromarray(panel)
        if im.size != (width, row_h):
            im = im.resize((width, row_h), Image.BILINEAR)
        x = i * width
        draw.text((x + 6, 5), name, fill=(255, 255, 255))
        canvas.paste(im, (x, label_h))

    canvas.save(out_path)


def _make_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def _rand(rng):
    if hasattr(rng, "random"):
        return float(rng.random())
    return float(rng.rand())


def _uniform(rng, low, high):
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return float(low + (high - low) * rng.rand())


def _integers(rng, low, high=None):
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def _choice(rng, a, size=None, replace=True, p=None):
    return rng.choice(a, size=size, replace=replace, p=p)


def _as_rgb_float(rgb):
    image = np.asarray(rgb)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.5:
        image /= 255.0
    return image


def _resize_instance_if_needed(inst, shape_hw):
    inst = np.asarray(inst)
    if inst.shape[:2] == shape_hw:
        return inst
    return cv2.resize(
        inst.astype(np.int32),
        (shape_hw[1], shape_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(inst.dtype)


def _resize_rgb_if_needed(image, shape_hw):
    if image.shape[:2] == shape_hw:
        return image
    return cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def _resize_bool_if_needed(mask, shape_hw):
    if mask.shape[:2] == shape_hw:
        return mask
    return (
        cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        > 0
    )


def _gray_and_gradient(image):
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return gray.astype(np.float32), grad.astype(np.float32)


def _build_amplitude_map(
    disp,
    valid_mask,
    distance_map,
    percent_range,
    min_abs_delta,
    max_abs_delta,
):
    if distance_map is not None:
        dist = np.asarray(distance_map, dtype=np.float32)
        dist = _resize_float_if_needed(dist, disp.shape)
        vals = dist[valid_mask & np.isfinite(dist)]
        if vals.size > 0:
            lo, hi = np.percentile(vals, [5, 95])
            denom = max(float(hi - lo), 1e-6)
            weight = np.clip((dist - lo) / denom, 0.0, 1.0)
        else:
            weight = np.zeros_like(disp, dtype=np.float32)
    else:
        abs_disp = np.abs(disp)
        vals = abs_disp[valid_mask & np.isfinite(abs_disp) & (abs_disp > 1e-6)]
        if vals.size > 0:
            far_disp, near_disp = np.percentile(vals, [5, 95])
            denom = max(float(near_disp - far_disp), 1e-6)
            weight = np.clip((near_disp - abs_disp) / denom, 0.0, 1.0)
        else:
            weight = np.zeros_like(disp, dtype=np.float32)

    pct_min, pct_max = percent_range
    pct = pct_min + (pct_max - pct_min) * weight.astype(np.float32)

    base = np.abs(disp).astype(np.float32)
    valid_vals = base[valid_mask & np.isfinite(base) & (base > 1e-6)]
    fill_value = float(np.median(valid_vals)) if valid_vals.size else 1.0
    base = np.where(np.isfinite(base) & (base > 1e-6), base, fill_value)

    amp = np.clip(base * pct, min_abs_delta, max_abs_delta)
    amp[~valid_mask] = 0.0
    return amp.astype(np.float32), weight.astype(np.float32)


def _apply_texture_amplitude_factor(
    amp,
    grad,
    valid_mask,
    texture_amp_range,
    texture_blur_sigma,
    texture_dilate_radius,
    max_abs_delta,
):
    strong_scale, weak_scale = texture_amp_range
    grad_smooth = grad.astype(np.float32)
    if texture_blur_sigma and texture_blur_sigma > 0:
        grad_smooth = cv2.GaussianBlur(
            grad_smooth,
            (0, 0),
            sigmaX=float(texture_blur_sigma),
            sigmaY=float(texture_blur_sigma),
        )

    vals = grad_smooth[valid_mask & np.isfinite(grad_smooth)]
    if vals.size > 0:
        lo, hi = np.percentile(vals, [10, 90])
        denom = max(float(hi - lo), 1e-6)
        texture = np.clip((grad_smooth - lo) / denom, 0.0, 1.0)
    else:
        texture = np.zeros_like(grad_smooth, dtype=np.float32)

    if texture_dilate_radius and texture_dilate_radius > 0:
        radius = int(texture_dilate_radius)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        texture = cv2.dilate(texture.astype(np.float32), kernel, iterations=1)
        texture = np.clip(texture, 0.0, 1.0)

    weak_texture = 1.0 - texture
    factor = float(strong_scale) + (float(weak_scale) - float(strong_scale)) * weak_texture
    factor = factor.astype(np.float32)
    factor[~valid_mask] = 0.0

    amp = np.asarray(amp, dtype=np.float32) * factor
    amp = np.clip(amp, 0.0, max_abs_delta)
    amp[~valid_mask] = 0.0
    return amp.astype(np.float32), factor


def _resize_float_if_needed(arr, shape_hw):
    if arr.shape[:2] == shape_hw:
        return arr
    return cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def _score_regions(
    inst,
    valid_mask,
    gray,
    grad,
    min_region_area=64,
    reject_bright_textured=True,
    bright_enough_luma_thresh=0.35,
    rich_texture_thresh=0.55,
):
    ids = [int(x) for x in np.unique(inst) if int(x) != 0]
    valid_grad = grad[valid_mask & np.isfinite(grad)]
    grad_hi = float(np.percentile(valid_grad, 90)) if valid_grad.size else 1.0
    grad_hi = max(grad_hi, 1e-6)
    image_area = float(inst.shape[0] * inst.shape[1])
    regions = []

    for region_id in ids:
        mask = (inst == region_id) & valid_mask
        if not np.any(mask):
            continue

        area = int(np.count_nonzero(mask))
        if area < min_region_area:
            continue
        ys, xs = np.where(mask)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_area = max(float(bbox_h * bbox_w), 1.0)
        bbox_short = float(min(bbox_h, bbox_w))
        bbox_long = float(max(bbox_h, bbox_w))
        thinness = bbox_short / max(bbox_long, 1.0)
        area_ratio = float(area) / max(image_area, 1.0)
        mean_luma = float(np.mean(gray[mask]))
        mean_grad = float(np.mean(grad[mask]))
        dark_score = 1.0 - np.clip(mean_luma, 0.0, 1.0)
        weak_texture_score = 1.0 - np.clip(mean_grad / grad_hi, 0.0, 1.0)
        rich_texture_score = 1.0 - weak_texture_score
        if (
            reject_bright_textured
            and mean_luma >= bright_enough_luma_thresh
            and rich_texture_score >= rich_texture_thresh
        ):
            continue

        small_score = 1.0 - np.clip(np.sqrt(area / max(image_area, 1.0)), 0.0, 1.0)
        score = 0.55 * dark_score + 0.45 * weak_texture_score

        regions.append(
            {
                "id": region_id,
                "area": area,
                "area_ratio": float(area_ratio),
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "bbox_fill": float(area / bbox_area),
                "thinness": float(thinness),
                "mean_luma": mean_luma,
                "mean_grad": mean_grad,
                "dark_score": float(dark_score),
                "weak_texture_score": float(weak_texture_score),
                "rich_texture_score": float(rich_texture_score),
                "small_score": float(small_score),
                "score": float(max(score, 0.02)),
            }
        )

    return regions


def _weighted_sample_regions(regions, num_regions, rng, region_sample_fraction=None):
    if not regions:
        return []

    if isinstance(num_regions, (tuple, list)):
        low, high = int(num_regions[0]), int(num_regions[1])
        count = _integers(rng, low, high + 1)
    else:
        count = int(num_regions)

    if region_sample_fraction is not None:
        if isinstance(region_sample_fraction, (tuple, list)):
            frac = _uniform(rng, float(region_sample_fraction[0]), float(region_sample_fraction[1]))
        else:
            frac = float(region_sample_fraction)
        count = max(count, int(round(len(regions) * frac)))

    count = max(0, min(count, len(regions)))
    if count == 0:
        return []

    weights = np.array([r["score"] for r in regions], dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum()
    idx = _choice(rng, np.arange(len(regions)), size=count, replace=False, p=weights)
    idx = np.atleast_1d(idx)
    return [regions[int(i)] for i in idx]


def _edge_probability(region, edge_prob, edge_small_area_ratio, edge_thinness_thresh):
    is_small = region["area_ratio"] <= edge_small_area_ratio
    is_thin_small = is_small and region["thinness"] <= edge_thinness_thresh
    is_very_small = region["area_ratio"] <= edge_small_area_ratio * 0.35
    if not (is_thin_small or is_very_small):
        return 0.0

    bias = 0.65 * region["dark_score"] + 0.35 * region["small_score"]
    return float(min(1.0, edge_prob * 1.35 * (0.35 + 0.65 * bias)))


def _add_low_texture_gaussian_noise(delta, mask, amp_map, gaussian_kernels, rng):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h = max(y1 - y0, 1)
    bbox_w = max(x1 - x0, 1)
    image_area = float(mask.shape[0] * mask.shape[1])
    area_ratio = float(ys.size) / max(image_area, 1.0)
    if area_ratio >= 0.18:
        core_fraction = 0.45
    elif area_ratio >= 0.03:
        core_fraction = 0.65
    else:
        core_fraction = 1.0

    if isinstance(gaussian_kernels, (tuple, list)):
        n_kernels = _integers(rng, int(gaussian_kernels[0]), int(gaussian_kernels[1]) + 1)
    else:
        n_kernels = int(gaussian_kernels)

    n_kernels += min(5, int(ys.size / 50000.0))

    for _ in range(max(1, n_kernels)):
        pick = _integers(rng, 0, ys.size)
        cy, cx = int(ys[pick]), int(xs[pick])
        # Convert desired half-height core coverage to sigma. At the edge of
        # the requested core, gaussian value is about 0.5.
        jitter = _uniform(rng, 0.85, 1.15)
        sigma_scale = 0.425 * core_fraction * jitter
        sigma_y = max(1.5, sigma_scale * bbox_h)
        sigma_x = max(1.5, sigma_scale * bbox_w)
        margin = int(max(sigma_y, sigma_x) * 3.0) + 2
        py0, py1 = max(0, y0 - margin), min(mask.shape[0], y1 + margin)
        px0, px1 = max(0, x0 - margin), min(mask.shape[1], x1 + margin)

        patch_mask = mask[py0:py1, px0:px1]
        if not np.any(patch_mask):
            continue

        kernel = _rotated_gaussian(
            patch_mask.shape,
            center_y=cy - py0,
            center_x=cx - px0,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            angle=_uniform(rng, 0.0, np.pi),
        )
        kernel = kernel ** _uniform(rng, 0.75, 1.15)
        soft = _soft_inside_mask(patch_mask)
        sign = -1.0 if _rand(rng) < 0.5 else 1.0
        delta[py0:py1, px0:px1] += sign * amp_map[py0:py1, px0:px1] * kernel * soft


def _add_edge_expand_contract(delta, mask, valid_mask, amp_map, rng):
    area = int(np.count_nonzero(mask))
    radius = int(np.clip(round(np.sqrt(max(area, 1)) / 35.0), 2, 8))
    radius += _integers(rng, 1, 4)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = mask.astype(np.uint8)

    if _rand(rng) < 0.5:
        changed = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool) & (~mask)
    else:
        eroded = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
        changed = mask & (~eroded)

    changed &= valid_mask
    if not np.any(changed):
        return

    edge_soft = cv2.GaussianBlur(changed.astype(np.float32), (0, 0), sigmaX=max(0.6, radius / 2.0))
    edge_soft = np.clip(edge_soft, 0.0, 1.0)
    sign = -1.0 if _rand(rng) < 0.5 else 1.0
    delta += sign * amp_map * edge_soft


def _add_scatter_blobs(delta, mask, amp_map, scatter_count, scatter_strength, rng):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return

    if isinstance(scatter_count, (tuple, list)):
        count = _integers(rng, int(scatter_count[0]), int(scatter_count[1]) + 1)
    else:
        count = int(scatter_count)

    area = ys.size
    count += min(3, int(area / 80000.0))

    for _ in range(max(0, count)):
        pick = _integers(rng, 0, ys.size)
        cy, cx = int(ys[pick]), int(xs[pick])
        ry = _uniform(rng, 1.5, 6.0)
        rx = _uniform(rng, 1.5, 6.0)
        if _rand(rng) < 0.5:
            rx *= _uniform(rng, 1.2, 2.0)
        else:
            ry *= _uniform(rng, 1.2, 2.0)

        margin = int(max(rx, ry) * 3.0) + 2
        y0, y1 = max(0, cy - margin), min(mask.shape[0], cy + margin + 1)
        x0, x1 = max(0, cx - margin), min(mask.shape[1], cx + margin + 1)
        patch_mask = mask[y0:y1, x0:x1]
        if not np.any(patch_mask):
            continue

        blob = _rotated_gaussian(
            patch_mask.shape,
            center_y=cy - y0,
            center_x=cx - x0,
            sigma_y=ry,
            sigma_x=rx,
            angle=_uniform(rng, 0.0, np.pi),
        )
        sign = -1.0 if _rand(rng) < 0.5 else 1.0
        delta[y0:y1, x0:x1] += (
            sign
            * float(scatter_strength)
            * amp_map[y0:y1, x0:x1]
            * blob
            * patch_mask.astype(np.float32)
        )


def _rotated_gaussian(shape, center_y, center_x, sigma_y, sigma_x, angle):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy -= float(center_y)
    xx -= float(center_x)
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * xx + sa * yy
    yr = -sa * xx + ca * yy
    g = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return g.astype(np.float32)


def _soft_inside_mask(mask):
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max(initial=0) == 0:
        return mask_u8.astype(np.float32)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    feather = max(1.5, min(mask.shape[:2]) * 0.025)
    inside = np.clip(dist / feather, 0.0, 1.0)
    inside = np.maximum(inside, mask_u8.astype(np.float32) * 0.08)
    return inside.astype(np.float32)


def _shared_percentile_range(arrays, percentiles=(2, 98)):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float32)
        finite = np.isfinite(arr)
        if np.any(finite):
            vals.append(arr[finite].reshape(-1))
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals, axis=0)
    lo, hi = np.percentile(vals, percentiles)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _heatmap(values, center_zero=False, value_range=None, center_percentile=98):
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        norm = np.zeros(arr.shape, dtype=np.uint8)
    elif center_zero:
        vmax = float(np.percentile(np.abs(arr[finite]), center_percentile))
        vmax = max(vmax, 1e-6)
        norm = np.clip((arr / (2.0 * vmax) + 0.5) * 255.0, 0, 255).astype(np.uint8)
    else:
        if value_range is None:
            lo, hi = np.percentile(arr[finite], [2, 98])
        else:
            lo, hi = value_range
        denom = max(float(hi - lo), 1e-6)
        norm = np.clip((arr - lo) / denom * 255.0, 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _selected_overlay(rgb, inst, selected):
    out = rgb.astype(np.float32).copy()
    if not selected:
        return rgb

    selected_mask = np.zeros(inst.shape, dtype=bool)
    for region_id in selected:
        mask = inst == region_id
        selected_mask |= mask
        color = np.array(
            [
                (37 * region_id + 160) % 255,
                (83 * region_id + 60) % 255,
                (131 * region_id + 20) % 255,
            ],
            dtype=np.float32,
        )
        out[mask] = 0.55 * out[mask] + 0.45 * color

    edges = selected_mask ^ (
        np.roll(selected_mask, 1, axis=0)
        & np.roll(selected_mask, -1, axis=0)
        & np.roll(selected_mask, 1, axis=1)
        & np.roll(selected_mask, -1, axis=1)
    )
    out[edges] = np.array([255, 0, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _parse_args():
    parser = argparse.ArgumentParser(description="Standalone SAM-guided disparity augmentation demo.")
    parser.add_argument("--rgb", default=None, help="RGB image path.")
    parser.add_argument("--sam", default=None, help="SAM .npz or instance image path.")
    parser.add_argument("--disp", default=None, help="Disparity path: pfm, npy, npz, png, or dpt.")
    parser.add_argument("--depth", default=None, help="Sintel .dpt depth path, converted to pseudo disparity.")
    parser.add_argument(
        "--sintel-root",
        default="datasets/Sintel/training",
        help="Sintel training root used when rgb/sam/disp/depth are not provided.",
    )
    parser.add_argument("--scene", default="market_2")
    parser.add_argument("--frame", default="frame_0019")
    parser.add_argument("--out", default="depth_real_aug_demo.png")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--focal-baseline", type=float, default=1000.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.rgb is None and args.sam is None and args.disp is None and args.depth is None:
        demo_sintel_temple3(
            root=args.sintel_root,
            scene=args.scene,
            frame=args.frame,
            out_path=args.out,
            seed=args.seed,
            focal_baseline=args.focal_baseline,
        )
        print(args.out)
        return

    if args.rgb is None or args.sam is None:
        raise ValueError("--rgb and --sam are required when not using the default Sintel demo paths")

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    sam = load_sam_instance(args.sam)

    distance_map = None
    if args.depth is not None:
        distance_map = read_sintel_depth(args.depth)
        disp = depth_to_pseudo_disparity(distance_map, focal_baseline=args.focal_baseline)
    elif args.disp is not None:
        disp = read_disparity(args.disp)
    else:
        raise ValueError("Provide either --disp or --depth")

    disp_aug, debug = augment_disparity_with_sam(
        sam,
        disp,
        rgb,
        distance_map=distance_map,
        rng=args.seed,
        return_debug=True,
    )
    save_demo_image(args.out, rgb, sam, disp, disp_aug, debug)
    print(args.out)


if __name__ == "__main__":
    main()
