import torch
import torch.nn.functional as F


def disp_to_pointmap_batched(
        disp,
        K,
        bf=1.0,
        min_disp=1e-4,
        max_depth=1e6,
):
    """
    把双目视差图反投影成相机坐标系下的 3D 点图。

    约定：
        - disp 是正视差，深度按 Z = bf / disp 计算。
        - K 只保留 pinhole 内参的四个标量：[fx, fy, cx, cy]。
        - 相机坐标系为 x 向右、y 向下、z 向前，和图像坐标方向一致。

    disp: [B, 1, H, W]
    K:    [B, 4], each row = [fx, fy, cx, cy]
    bf:   scalar or [B] / [B, 1]，表示 baseline * focal length

    return:
        P:     [B, 3, H, W] camera-coordinate point map
        Z:     [B, 1, H, W] depth
        valid: [B, 1, H, W]
    """
    assert disp.ndim == 4 and disp.shape[1] == 1
    assert K.ndim == 2 and K.shape[1] == 4

    B, _, H, W = disp.shape
    device = disp.device
    dtype = disp.dtype

    fx = K[:, 0].view(B, 1, 1, 1).to(device=device, dtype=dtype)
    fy = K[:, 1].view(B, 1, 1, 1).to(device=device, dtype=dtype)
    cx = K[:, 2].view(B, 1, 1, 1).to(device=device, dtype=dtype)
    cy = K[:, 3].view(B, 1, 1, 1).to(device=device, dtype=dtype)

    if not torch.is_tensor(bf):
        bf = torch.tensor(bf, device=device, dtype=dtype)
    bf = bf.to(device=device, dtype=dtype)
    if bf.ndim == 0:
        bf = bf.view(1, 1, 1, 1).expand(B, 1, 1, 1)
    elif bf.ndim == 1:
        bf = bf.view(B, 1, 1, 1)
    elif bf.ndim == 2:
        bf = bf.view(B, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported bf shape: {tuple(bf.shape)}")

    # 只在有限且大于阈值的视差上计算深度，避免 0/负视差导致无穷远或反向深度。
    valid = torch.isfinite(disp) & (disp > min_disp)

    disp_safe = disp.clamp(min=min_disp)
    Z = bf / disp_safe
    valid = valid & torch.isfinite(Z) & (Z > 0) & (Z < max_depth)

    # 生成像素坐标网格，并用 pinhole 模型反投影：
    # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy。
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.view(1, 1, H, W)
    y = y.view(1, 1, H, W)

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    P = torch.cat([X, Y, Z], dim=1)
    # 无效点置零；调用方如果需要区分真实原点和无效点，应同时使用 valid mask。
    P = torch.where(valid.expand_as(P), P, torch.zeros_like(P))

    return P, Z, valid


def depth_to_pointmap_batched(
        depth,
        K,
        min_depth=1e-4,
        max_depth=1e6,
):
    """
    把 metric depth 直接反投影成相机坐标系下的 3D 点图。

    如果 d1/d2 实际已经是 metric depth，而不是 disparity，用这个函数。
    深度本身不做 bf/disp 转换，只根据 min_depth/max_depth 生成 valid mask。

    depth: [B, 1, H, W]
    K:     [B, 4] = [fx, fy, cx, cy]

    return:
        P:     [B, 3, H, W] camera-coordinate point map
        Z:     [B, 1, H, W] depth
        valid: [B, 1, H, W]
    """
    assert depth.ndim == 4 and depth.shape[1] == 1
    assert K.ndim == 2 and K.shape[1] == 4

    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    fx = K[:, 0].view(B, 1, 1, 1).to(device=device, dtype=dtype)
    fy = K[:, 1].view(B, 1, 1, 1).to(device=device, dtype=dtype)
    cx = K[:, 2].view(B, 1, 1, 1).to(device=device, dtype=dtype)
    cy = K[:, 3].view(B, 1, 1, 1).to(device=device, dtype=dtype)

    Z = depth
    valid = torch.isfinite(Z) & (Z > min_depth) & (Z < max_depth)

    # 与 disparity 分支相同，按 pinhole 模型把每个像素中心反投影到 3D。
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.view(1, 1, H, W)
    y = y.view(1, 1, H, W)

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    P = torch.cat([X, Y, Z], dim=1)
    # Z 保留原始深度，P 中无效点置零；后续计算应继续带 valid mask。
    P = torch.where(valid.expand_as(P), P, torch.zeros_like(P))

    return P, Z, valid


def pointmap_to_normal(
        P,
        valid=None,
        eps=1e-6,
        require_3x3_valid=True,
        orient_to_positive_z=True,
):
    """
    从相机坐标系点图估计每个像素的表面法线。

    核心思路：
        - 在点图上分别沿图像 x/y 方向做差分，得到局部切向量 dx/dy。
        - 用 cross(dx, dy) 得到法线方向。
        - 在 x 右、y 下、z 前的相机坐标系中，正面平面会得到约 +z 的法线。

    P:     [B, 3, H, W]
    valid: [B, 1, H, W]
    require_3x3_valid:
        True  时要求中心点周围 3x3 全部有效，能避免无效点置零污染差分；
        False 时只检查中心点有效，边界和遮挡附近更容易产生伪法线。
    orient_to_positive_z:
        True  保持当前 cross(dx, dy) 方向，默认约指向 +z；
        False 会整体翻转方向，供需要相反朝向的损失或渲染使用。

    return:
        N:      [B, 3, H, W]
        N_valid:[B, 1, H, W]
    """
    assert P.ndim == 4 and P.shape[1] == 3

    dx = torch.zeros_like(P)
    dy = torch.zeros_like(P)

    # x 方向中心差分。未除以 2 不影响单位法线方向，只会改变归一化前的长度。
    dx[:, :, :, 1:-1] = P[:, :, :, 2:] - P[:, :, :, :-2]
    # 图像左右边界没有完整邻域，用一阶差分；如果 require_3x3_valid=True，
    # 后面会因为 padding 的局部有效性检查把最外圈标成无效。
    dx[:, :, :, 0] = P[:, :, :, 1] - P[:, :, :, 0]
    dx[:, :, :, -1] = P[:, :, :, -1] - P[:, :, :, -2]

    # y 方向中心差分，同样在上下边界退化为一阶差分。
    dy[:, :, 1:-1, :] = P[:, :, 2:, :] - P[:, :, :-2, :]
    dy[:, :, 0, :] = P[:, :, 1, :] - P[:, :, 0, :]
    dy[:, :, -1, :] = P[:, :, -1, :] - P[:, :, -2, :]

    # 相机坐标：x right, y down, z forward。
    # 对 fronto-parallel 平面，cross(dx, dy) 约为 +z。
    N = torch.cross(dx, dy, dim=1)

    if not orient_to_positive_z:
        N = -N

    # 归一化前长度太小表示局部点图退化，不能形成稳定法线。
    norm = torch.linalg.norm(N, dim=1, keepdim=True)
    N_valid = norm > eps
    N = N / norm.clamp(min=eps)

    if valid is not None:
        if require_3x3_valid:
            # 要求 3x3 全有效，避免某个邻居是被置零的 invalid point。
            # padding=1 会把图像外视作无效，因此最外一圈不会通过这个检查。
            local_valid = F.avg_pool2d(
                valid.float(), kernel_size=3, stride=1, padding=1
            ) > 0.99
            N_valid = N_valid & local_valid
        else:
            N_valid = N_valid & valid.bool()

    N = torch.where(N_valid.expand_as(N), N, torch.zeros_like(N))

    return N, N_valid


def build_normals_from_disparity_pair(
        d1,
        d2,
        K,
        bf=1.0,
        min_disp=1e-4,
        max_depth=1e6,
        smooth_disp=False,
):
    """
    从一对视差图构建原始尺度的点图、深度、法线和有效性 mask。

    d1, d2: [B, 1, H, W], disparity
    K:      [B, 4] = [fx, fy, cx, cy]
    smooth_disp:
        可选的 3x3 均值滤波。它能减小视差噪声，但会模糊深度边界；
        如果输入含无效视差，最好保持 False 或改成 mask-aware smoothing。

    return dict with:
        P1, P2, Z1, Z2, N1, N2, valid1, valid2, normal_valid1, normal_valid2
    """
    if smooth_disp:
        # 可选：会减少噪声，但会模糊边界。第一版可以 False。
        d1_use = F.avg_pool2d(d1, kernel_size=3, stride=1, padding=1)
        d2_use = F.avg_pool2d(d2, kernel_size=3, stride=1, padding=1)
    else:
        d1_use, d2_use = d1, d2

    # disparity -> depth -> camera point map。
    P1, Z1, valid1 = disp_to_pointmap_batched(
        d1_use, K, bf=bf, min_disp=min_disp, max_depth=max_depth
    )
    P2, Z2, valid2 = disp_to_pointmap_batched(
        d2_use, K, bf=bf, min_disp=min_disp, max_depth=max_depth
    )

    # point map -> surface normal。normal_valid 会额外排除局部 3x3 不完整区域。
    N1, normal_valid1 = pointmap_to_normal(P1, valid1)
    N2, normal_valid2 = pointmap_to_normal(P2, valid2)

    return {
        "P1": P1,
        "P2": P2,
        "Z1": Z1,
        "Z2": Z2,
        "N1": N1,
        "N2": N2,
        "valid1": valid1,
        "valid2": valid2,
        "normal_valid1": normal_valid1,
        "normal_valid2": normal_valid2,
    }


def build_normals_from_depth_pair(
        d1,
        d2,
        K,
        min_depth=1e-4,
        max_depth=1e6,
        smooth_depth=False,
):
    """
    从一对 metric depth 图构建原始尺度的点图、深度、法线和有效性 mask。

    d1, d2: [B, 1, H, W], metric depth
    K:      [B, 4] = [fx, fy, cx, cy]
    smooth_depth:
        可选的 3x3 均值滤波。它能减小深度噪声，但会模糊深度边界；
        如果输入含无效深度，最好保持 False 或改成 mask-aware smoothing。

    return dict with:
        P1, P2, Z1, Z2, N1, N2, valid1, valid2, normal_valid1, normal_valid2
    """
    if smooth_depth:
        d1_use = F.avg_pool2d(d1, kernel_size=3, stride=1, padding=1)
        d2_use = F.avg_pool2d(d2, kernel_size=3, stride=1, padding=1)
    else:
        d1_use, d2_use = d1, d2

    # depth -> camera point map。
    P1, Z1, valid1 = depth_to_pointmap_batched(
        d1_use, K, min_depth=min_depth, max_depth=max_depth
    )
    P2, Z2, valid2 = depth_to_pointmap_batched(
        d2_use, K, min_depth=min_depth, max_depth=max_depth
    )

    # point map -> surface normal。normal_valid 会额外排除局部 3x3 不完整区域。
    N1, normal_valid1 = pointmap_to_normal(P1, valid1)
    N2, normal_valid2 = pointmap_to_normal(P2, valid2)

    return {
        "P1": P1,
        "P2": P2,
        "Z1": Z1,
        "Z2": Z2,
        "N1": N1,
        "N2": N2,
        "valid1": valid1,
        "valid2": valid2,
        "normal_valid1": normal_valid1,
        "normal_valid2": normal_valid2,
    }


def downsample_surface_maps(P, N, C, scale, eps=1e-6):
    """
    把点图/法线/置信度按整数 scale 下采样到较粗尺度。

    P 使用 C 加权平均，相当于只统计有效点的均值；
    N 先加权平均再重新归一化，避免平均后长度不为 1。
    注意 avg_pool2d 默认只保留完整窗口：如果 H/W 不能被 scale 整除，
    最右和最下侧不足一个窗口的像素会被丢弃。

    P: [B, 3, H, W]
    N: [B, 3, H, W]
    C: [B, 1, H, W], confidence / valid mask

    return:
        P_s, N_s, C_s
    """
    C = C.to(device=P.device, dtype=P.dtype)

    # C_s 是窗口内平均置信度；对于 0/1 mask，它也是有效点占比。
    C_s = F.avg_pool2d(C, kernel_size=scale, stride=scale)

    # avg(P*C) / avg(C) = sum(P*C) / sum(C)，即有效点加权均值。
    P_s = F.avg_pool2d(P * C, kernel_size=scale, stride=scale)
    P_s = P_s / C_s.clamp(min=eps)

    # 法线也按置信度平均，然后再单位化。
    N_s = F.avg_pool2d(N * C, kernel_size=scale, stride=scale)
    N_s = N_s / C_s.clamp(min=eps)

    N_norm = torch.linalg.norm(N_s, dim=1, keepdim=True)
    N_s = N_s / N_norm.clamp(min=eps)

    # 有效点比例过低或平均法线退化时，该 coarse cell 不可信。
    valid_s = (C_s > 0.5) & (N_norm > eps)

    P_s = torch.where(valid_s.expand_as(P_s), P_s, torch.zeros_like(P_s))
    N_s = torch.where(valid_s.expand_as(N_s), N_s, torch.zeros_like(N_s))
    C_s = C_s * valid_s.to(dtype=C_s.dtype)

    return P_s, N_s, C_s


def build_surface_pyramid_from_disparity_pair(
        d1,
        d2,
        K,
        bf=1.0,
        scales=(4, 8, 16),
        min_disp=1e-4,
        max_depth=1e6,
):
    """
    原始尺度先由视差构建 surface maps，再直接下采样到各个目标尺度。

    注意：
        - P/N 仍表示原相机坐标系下的 3D 几何量，不是特征网格坐标。
        - 每个尺度返回缩放后的 K，供在该尺度上把像素/特征点反投影到 3D。
        - 当前 K_s 使用简单 K / scale 规则；如果你的特征图采用 half-pixel
          resize 约定，principal point 可能需要额外的 0.5 像素修正。

    d1, d2: [B, 1, H, W]
    K:      [B, 4] = [fx, fy, cx, cy]
    """
    surf = build_normals_from_disparity_pair(
        d1, d2, K, bf=bf, min_disp=min_disp, max_depth=max_depth
    )

    C1 = surf["valid1"].float() * surf["normal_valid1"].float()
    C2 = surf["valid2"].float() * surf["normal_valid2"].float()

    pyramid = {"full": surf}

    for s in scales:
        P1_s, N1_s, C1_s = downsample_surface_maps(
            surf["P1"], surf["N1"], C1, scale=s
        )
        P2_s, N2_s, C2_s = downsample_surface_maps(
            surf["P2"], surf["N2"], C2, scale=s
        )

        # 注意：pointmap 仍然是原相机坐标系下的 3D 点；
        # 但你在对应尺度 backproject 当前 q 时，需要用缩放后的 K。
        K_s = K.to(device=surf["P1"].device, dtype=surf["P1"].dtype).clone()
        K_s[:, 0] = K_s[:, 0] / s  # fx
        K_s[:, 1] = K_s[:, 1] / s  # fy
        K_s[:, 2] = K_s[:, 2] / s  # cx
        K_s[:, 3] = K_s[:, 3] / s  # cy

        pyramid[f"1/{s}"] = {
            "P1": P1_s,
            "P2": P2_s,
            "N1": N1_s,
            "N2": N2_s,
            "C1": C1_s,
            "C2": C2_s,
            "K": K_s,
        }

    return pyramid


def build_surface_pyramid_from_depth_pair(
        d1,
        d2,
        K,
        scales=(4, 8, 16),
        min_depth=1e-4,
        max_depth=1e6,
):
    """
    原始尺度先由 metric depth 构建 surface maps，再直接下采样到各个目标尺度。

    d1, d2: [B, 1, H, W], metric depth
    K:      [B, 4] = [fx, fy, cx, cy]

    return:
        pyramid["full"]: 原始尺度 surface maps
        pyramid["1/s"]:  每个 scale 的 P1/P2/N1/N2/C1/C2/K
    """
    surf = build_normals_from_depth_pair(
        d1, d2, K, min_depth=min_depth, max_depth=max_depth
    )

    C1 = surf["valid1"].float() * surf["normal_valid1"].float()
    C2 = surf["valid2"].float() * surf["normal_valid2"].float()

    pyramid = {"full": surf}

    for s in scales:
        P1_s, N1_s, C1_s = downsample_surface_maps(
            surf["P1"], surf["N1"], C1, scale=s
        )
        P2_s, N2_s, C2_s = downsample_surface_maps(
            surf["P2"], surf["N2"], C2, scale=s
        )

        K_s = K.to(device=surf["P1"].device, dtype=surf["P1"].dtype).clone()
        K_s[:, 0] = K_s[:, 0] / s  # fx
        K_s[:, 1] = K_s[:, 1] / s  # fy
        K_s[:, 2] = K_s[:, 2] / s  # cx
        K_s[:, 3] = K_s[:, 3] / s  # cy

        pyramid[f"1/{s}"] = {
            "P1": P1_s,
            "P2": P2_s,
            "N1": N1_s,
            "N2": N2_s,
            "C1": C1_s,
            "C2": C2_s,
            "K": K_s,
        }

    return pyramid


def _sample_map_at_coords(feat, coords):
    """
    Sample a feature map at pixel coordinates.

    feat:   [B, C, H, W]
    coords: [B, N, H, W, 2], last dim is (x, y) in feat pixel coordinates

    return: [B, C, N, H, W]
    """
    B, C, H, W = feat.shape
    _, N, Hc, Wc, _ = coords.shape

    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / max(W - 1, 1) - 1
    ygrid = 2 * ygrid / max(H - 1, 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    grid = grid.reshape(B, N * Hc, Wc, 2)

    sampled = F.grid_sample(feat, grid, align_corners=True, mode="bilinear")
    return sampled.reshape(B, C, N, Hc, Wc)


def _coords_inside_image(coords, height, width):
    """coords: [B, N, H, W, 2], return [B, 1, N, H, W]."""
    x = coords[..., 0]
    y = coords[..., 1]
    inside = (x > -0.5) & (x < width - 0.5) & (y > -0.5) & (y < height - 0.5)
    return inside.unsqueeze(1)


def _backproject_candidates(coords, depth, K, eps=1e-6):
    """
    Backproject local 2D candidates with the current predicted depth.

    coords: [B, N, H, W, 2] in the current feature scale
    depth:  [B, 1, H, W]
    K:      [B, 4] = [fx, fy, cx, cy], already scaled to this feature scale

    return:
        P:       [B, 3, N, H, W]
        ray_dot: [B, 1, N, H, W], dP/ddepth projected onto a normal later
    """
    B, N, H, W, _ = coords.shape
    dtype = depth.dtype
    device = depth.device

    K = K.to(device=device, dtype=dtype)
    fx = K[:, 0].view(B, 1, 1, 1)
    fy = K[:, 1].view(B, 1, 1, 1)
    cx = K[:, 2].view(B, 1, 1, 1)
    cy = K[:, 3].view(B, 1, 1, 1)

    x = coords[..., 0]
    y = coords[..., 1]
    z = depth[:, :, None].expand(B, 1, N, H, W)

    x_over_z = (x[:, None] - cx[:, :, None]) / fx[:, :, None].clamp(min=eps)
    y_over_z = (y[:, None] - cy[:, :, None]) / fy[:, :, None].clamp(min=eps)

    X = x_over_z * z
    Y = y_over_z * z
    P = torch.cat([X, Y, z], dim=1)

    ray = torch.cat([x_over_z, y_over_z, torch.ones_like(x_over_z)], dim=1)
    return P, ray


def backproject_disparity_candidates(
        coords1,
        disp,
        K,
        radius=None,
        bf=500.0,
        min_disp=1e-4,
        eps=1e-6,
        return_aux=False,
):
    """
    Backproject the current correspondence center using the predicted disparity.

    coords1: [B, 2, H, W], current correspondence in the feature scale
    disp:    [B, 1, H, W], current predicted frame-2 disparity
    K:       [B, 4] = [fx, fy, cx, cy], already scaled to this feature scale
    radius:  accepted for backward-compatible call sites, but not used
    bf:      baseline * focal length, scalar or [B]

    return:
        P_pred: [B, 3, H, W]

    if return_aux=True:
        also return coords1 [B, 2, H, W] and Z [B, 1, H, W].
    """
    assert coords1.ndim == 4 and coords1.shape[1] == 2
    assert disp.ndim == 4 and disp.shape[1] == 1
    assert K.ndim == 2 and K.shape[1] == 4

    B, _, H, W = coords1.shape
    device = coords1.device
    dtype = coords1.dtype

    K = K.to(device=device, dtype=dtype)
    fx = K[:, 0].view(B, 1, 1, 1)
    fy = K[:, 1].view(B, 1, 1, 1)
    cx = K[:, 2].view(B, 1, 1, 1)
    cy = K[:, 3].view(B, 1, 1, 1)

    if not torch.is_tensor(bf):
        bf = torch.tensor(bf, device=device, dtype=dtype)
    bf = bf.to(device=device, dtype=dtype)
    if bf.ndim == 0:
        bf = bf.view(1, 1, 1, 1).expand(B, 1, 1, 1)
    elif bf.ndim == 1:
        bf = bf.view(B, 1, 1, 1)
    elif bf.ndim == 2:
        bf = bf.view(B, 1, 1, 1)
    elif bf.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported bf shape: {tuple(bf.shape)}")

    u = coords1[:, 0:1]
    v = coords1[:, 1:2]

    Z_center = bf / disp.clamp(min=min_disp)
    X = (u - cx) * Z_center / fx.clamp(min=eps)
    Y = (v - cy) * Z_center / fy.clamp(min=eps)
    P_pred = torch.cat([X, Y, Z_center], dim=1).contiguous()

    if return_aux:
        return P_pred, coords1, Z_center
    return P_pred


def point_to_plane_residual_volume(
        P_pred,
        P2_neighbors,
        N2_neighbors,
        coords2=None,
        disp=None,
        K=None,
        bf=500.0,
        min_disp=1e-4,
        eps=1e-6,
        residual_scale=None,
        clip=None,
        return_jacobian=False,
):
    """
    计算 point-to-plane 残差体，并可选返回残差对 (u, v, disparity) 的 Jacobian。

    这里采用的是“中心预测点到邻域目标平面”的定义，而不是每个候选点都重新反投影：

        q     = 当前估计的第二帧对应位置，即 coords2 = p + flow
        q_k   = q + delta_k，第二帧邻域候选位置
        P     = P_pred(q, disp)，当前中心预测 3D 端点
        Q_k   = P2(q_k)，第二帧邻域采样到的目标 3D 点
        n_k   = N2(q_k)，第二帧邻域采样到的目标法线

    point-to-plane 残差：

        r_k = n_k^T (P - Q_k)
            = n_x * (X - Q_x) + n_y * (Y - Q_y) + n_z * (Z - Q_z)

    对于视差输入：

        Z = bf / d
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    若把目标平面 (Q_k, n_k) 视为固定，只对当前中心状态 (u, v, d) 线性化，
    则 Jacobian 为：

        dr/du = n_x * Z / fx

        dr/dv = n_y * Z / fy

        dr/dd =
            [n_x * (u - cx) / fx + n_y * (v - cy) / fy + n_z]
            * (-bf / d^2)

    如果后续希望构造最小二乘能量 E = 0.5 * r^2 的下降方向，则：

        dE/du = r * dr/du
        dE/dv = r * dr/dv
        dE/dd = r * dr/dd

        下降方向 = -r * J

    参数形状：
        P_pred:       [B, 3, H, W]，当前中心预测 3D 端点 P
        P2_neighbors: [B, 3, N, H, W]，目标邻域 3D 点 Q_k
        N2_neighbors: [B, 3, N, H, W]，目标邻域法线 n_k
        coords2:      [B, 2, H, W]，当前第二帧对应位置 q = p + flow
        disp:         [B, 1, H, W]，当前预测第二帧 disparity d
        K:            [B, 4]，当前尺度内参 [fx, fy, cx, cy]

    返回：
        residual: [B, N, H, W]

    若 return_jacobian=True，返回：
        residual: [B, N, H, W]
        jacobian: [B, 3*N, H, W]，通道按 [dr/du, dr/dv, dr/ddisp] 拼接
    """
    assert P_pred.ndim == 4 and P_pred.shape[1] == 3
    assert P2_neighbors.ndim == 5 and P2_neighbors.shape[1] == 3
    assert N2_neighbors.ndim == 5 and N2_neighbors.shape[1] == 3
    assert P2_neighbors.shape == N2_neighbors.shape
    assert P_pred.shape[0] == P2_neighbors.shape[0]
    assert P_pred.shape[2:] == P2_neighbors.shape[3:]

    # P_pred 是中心点 [B,3,H,W]，需要插入邻域维度变成 [B,3,1,H,W]；
    # 它会广播到每个候选平面 k 上，形成 P - Q_k。
    # 逐元素乘目标法线后沿 xyz 通道求和，即 n_k^T(P - Q_k)。
    residual = torch.sum(
        N2_neighbors * (P_pred[:, :, None, :, :] - P2_neighbors),
        dim=1,
    )

    if residual_scale is not None:
        residual = residual / residual_scale

    if clip is not None:
        residual = residual.clamp(min=-clip, max=clip)

    residual = residual.contiguous()

    if not return_jacobian:
        return residual

    assert coords2 is not None and disp is not None and K is not None
    assert coords2.ndim == 4 and coords2.shape[1] == 2
    assert disp.ndim == 4 and disp.shape[1] == 1
    assert K.ndim == 2 and K.shape[1] == 4
    assert coords2.shape[0] == P_pred.shape[0]
    assert coords2.shape[2:] == P_pred.shape[2:]
    assert disp.shape[0] == P_pred.shape[0]
    assert disp.shape[2:] == P_pred.shape[2:]

    B, _, H, W = coords2.shape
    dtype = P_pred.dtype
    device = P_pred.device

    coords2 = coords2.to(device=device, dtype=dtype)
    disp = disp.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)

    fx = K[:, 0].view(B, 1, 1, 1)
    fy = K[:, 1].view(B, 1, 1, 1)
    cx = K[:, 2].view(B, 1, 1, 1)
    cy = K[:, 3].view(B, 1, 1, 1)

    if not torch.is_tensor(bf):
        bf = torch.tensor(bf, device=device, dtype=dtype)
    bf = bf.to(device=device, dtype=dtype)
    if bf.ndim == 0:
        bf = bf.view(1, 1, 1, 1).expand(B, 1, 1, 1)
    elif bf.ndim == 1:
        bf = bf.view(B, 1, 1, 1)
    elif bf.ndim == 2:
        bf = bf.view(B, 1, 1, 1)
    elif bf.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported bf shape: {tuple(bf.shape)}")

    u = coords2[:, 0:1]
    v = coords2[:, 1:2]
    disp_safe = disp.clamp(min=min_disp)

    # 视差转深度：Z = bf / d。
    # 这里 Z 的形状是 [B,1,H,W]，会自动广播到邻域维度 N。
    Z = bf / disp_safe

    # 邻域目标法线 n_k = (n_x, n_y, n_z)，形状均为 [B,N,H,W]。
    nx = N2_neighbors[:, 0]
    ny = N2_neighbors[:, 1]
    nz = N2_neighbors[:, 2]

    # 由 X = (u-cx)Z/fx 可得 dX/du = Z/fx，
    # 因此 dr/du = n_x * dX/du = n_x * Z/fx。
    dr_du = nx * (Z / fx.clamp(min=eps))

    # 同理 Y = (v-cy)Z/fy，所以 dr/dv = n_y * Z/fy。
    dr_dv = ny * (Z / fy.clamp(min=eps))

    # 对 disparity d：
    #   dZ/dd = -bf / d^2
    #   dX/dd = (u-cx)/fx * dZ/dd
    #   dY/dd = (v-cy)/fy * dZ/dd
    # 所以：
    #   dr/dd = [n_x*(u-cx)/fx + n_y*(v-cy)/fy + n_z] * (-bf/d^2)
    ray_dot_n = (
        nx * ((u - cx) / fx.clamp(min=eps))
        + ny * ((v - cy) / fy.clamp(min=eps))
        + nz
    )
    dr_ddisp = ray_dot_n * (-bf / (disp_safe ** 2))

    return residual, dr_du.contiguous(), dr_dv.contiguous(), dr_ddisp.contiguous()


def surface_aligned_p2p_features(
        surface,
        coords1,
        pred_depth2,
        radius=4,
        residual_scale=10.0,
        min_depth=1e-4,
        eps=1e-6,
):
    """
    Build local point-to-plane cost features around the current correspondence.

    The residual for a candidate q_k is:
        r_k = n2(q_k)^T (P_pred(q_k, pred_depth2) - P2(q_k))

    To make the scalar residual useful for the recurrent u/v/d update, expose
    Gauss-Newton-like components:
        r * dr/du, r * dr/dv, r * dr/ddepth

    surface:
        dict from build_surface_pyramid_from_*_pair()["1/s"], with P2/N2/C2/K.
    coords1:
        [B, 2, H, W], current frame-2 correspondence in this feature scale.
    pred_depth2:
        [B, 1, H, W], current predicted frame-2 depth value.

    return:
        surface_flow:  [B, 2*(2r+1)^2, H, W], channels are [rJ_u, rJ_v]
        surface_depth: [B,   (2r+1)^2, H, W], channels are [rJ_depth]
    """
    assert coords1.ndim == 4 and coords1.shape[1] == 2
    assert pred_depth2.ndim == 4 and pred_depth2.shape[1] == 1

    P2 = surface["P2"].to(device=coords1.device, dtype=coords1.dtype)
    N2 = surface["N2"].to(device=coords1.device, dtype=coords1.dtype)
    C2 = surface["C2"].to(device=coords1.device, dtype=coords1.dtype)
    K = surface["K"]

    B, _, H, W = coords1.shape
    device = coords1.device
    dtype = coords1.dtype

    dy, dx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    delta = torch.stack([dx, dy], dim=-1).reshape(1, -1, 1, 1, 2)
    coords = coords1.permute(0, 2, 3, 1)[:, None] + delta

    Q = _sample_map_at_coords(P2, coords)
    N = _sample_map_at_coords(N2, coords)
    C = _sample_map_at_coords(C2, coords)

    P_pred, ray = _backproject_candidates(coords, pred_depth2, K, eps=eps)
    residual = torch.sum(N * (P_pred - Q), dim=1, keepdim=True)

    # Jacobians if target sampling is treated as fixed. This follows the
    # projective point-to-plane approximation used by ICP.
    K = K.to(device=device, dtype=dtype)
    fx = K[:, 0].view(B, 1, 1, 1, 1)
    fy = K[:, 1].view(B, 1, 1, 1, 1)
    depth = pred_depth2[:, :, None].expand_as(residual)
    dr_du = N[:, 0:1] * depth / fx.clamp(min=eps)
    dr_dv = N[:, 1:2] * depth / fy.clamp(min=eps)
    dr_ddepth = torch.sum(N * ray, dim=1, keepdim=True)

    inside = _coords_inside_image(coords, H, W)
    pred_valid = torch.isfinite(pred_depth2) & (pred_depth2 > min_depth)
    pred_valid = pred_valid[:, :, None].expand_as(residual)
    valid = inside & (C > 0.01) & pred_valid

    r = (residual / residual_scale).clamp(min=-1.0, max=1.0)
    r = torch.where(valid, r, torch.zeros_like(r))

    surface_u = r * dr_du
    surface_v = r * dr_dv
    surface_d = r * dr_ddepth

    surface_u = torch.where(valid, surface_u, torch.zeros_like(surface_u))
    surface_v = torch.where(valid, surface_v, torch.zeros_like(surface_v))
    surface_d = torch.where(valid, surface_d, torch.zeros_like(surface_d))

    Nn = surface_u.shape[2]
    surface_flow = torch.cat([surface_u, surface_v], dim=2).reshape(B, 2 * Nn, H, W)
    surface_depth = surface_d.reshape(B, Nn, H, W)

    return surface_flow.contiguous(), surface_depth.contiguous()
