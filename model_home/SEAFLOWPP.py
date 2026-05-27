import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from core.utils.utils import coords_grid, upflow2,updepth2
from core.utils.xcit import XCiT
from core.utils.normal_utils import (
    backproject_disparity_candidates,
    build_surface_pyramid_from_disparity_pair,
    point_to_plane_residual_volume,
)
from model_home.broad_corr import CudaBroadCorrBlockSG,CudaBroadCorrBlockNs
from model_home.update import MotioneEncoderP2P_Flow, MotioneEncoderP2P_SVS, P2PMotionUpdateBlock
from model_home.extractor import BasicEncoder_resconv, Basic_Context_Encoder_resconvb
import matplotlib.pyplot as plt
#做一个结构代价的机制用轻量化提取器提取深度特征，双路一起来，新版本先做一个单目的版本。
#其实如果有了背景的相机运动完全可以顺便把前景分离出来啊，然后又多了一个运动先验，可以加到光流里面做辅助，你看看最近有没有相关的工作，我觉得应该不少，看看我们相比于他们有什么核心的创新
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class SEAFLOW3DPP(nn.Module):
    def __init__(self, args):
        super(SEAFLOW3DPP, self).__init__()
        self.args = args
        args.dim = 128

        self.correlation_depth = 162
        self.hidden_dim = hdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.fnet = BasicEncoder_resconv(input_dim=3,output_dim=256, norm_fn="group",model_type="ccmr")
        self.cnet = Basic_Context_Encoder_resconvb(input_dim=8,output_dim=2 * self.args.dim, norm_fn="group",model_type="ccmr+")
        self.update_block = P2PMotionUpdateBlock(hidden_dim=hdim)#后续的迭代模块
        self.update_blockx16 = P2PMotionUpdateBlock(hidden_dim=hdim)


        self.motionenc_flow16 = MotioneEncoderP2P_Flow(cor_planes=(2 * args.itr1 + 1) ** 2)
        self.motionenc_flow = MotioneEncoderP2P_Flow(cor_planes=(2 * args.itr2 + 1) ** 2)
        self.motionenc_svs16 = MotioneEncoderP2P_SVS(cor_planes=(2 * args.itr1 + 1) ** 2)
        self.motionenc_svs = MotioneEncoderP2P_SVS(cor_planes=(2 * args.itr2 + 1) ** 2)



        self.mask2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 * 9, 1, padding=0))
        self.mask4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 * 9, 1, padding=0))
        self.mask8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 * 9, 1, padding=0))
        #TODO  这里还得搞一个新的迭代器，xcit(corr)+unet   后续接上一个计算RT的+一个分离前景的功能

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

    def upsample_dz(self, flow, mask, scale=8):
        """ Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold( flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, scale * H, scale * W)


    def initialize_flow16x(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 16, W // 16).to(img.device)
        coords1 = coords_grid(N, H // 16, W // 16).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def initialize_flowNx(self, img,Nc):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // Nc, W // Nc).to(img.device)
        coords1 = coords_grid(N, H // Nc, W // Nc).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def encode_p2p_motion(self, coords1, disp2, surface, corr_p2_fn, corr_n2_fn, corr_flow,
                          flow2d,depthc, radius, flow_encoder, svs_encoder):
        corrp2 = corr_p2_fn(coords1)
        corrn2 = corr_n2_fn(coords1)
        bf = getattr(self.args, "bf", 500.0)
        Ppred = backproject_disparity_candidates(
            coords1, disp2, surface["K"], radius=radius, bf=bf)
        p2p_r, dru, drv, drd = point_to_plane_residual_volume(
            Ppred, corrp2, corrn2, coords2=coords1, disp=disp2,
            K=surface["K"], bf=bf,
            residual_scale=getattr(self.args, "svs_residual_scale", 10.0),
            clip=getattr(self.args, "svs_residual_clip", 1.0),
            return_jacobian=True)

        flow_motion = flow_encoder(corr_flow, flow2d)
        svs_motion = svs_encoder(p2p_r, dru, drv, drd,depthc)
        return torch.cat([flow_motion, svs_motion], dim=1)

    def forward(self, image1, image2, d1,d2,intrinsics,iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        #搞一个参考RAFT3D版本的，u,v,d
        #在之前的实验中初始化部件搞三次是最好的，这部分要单独分出来
        image1 = 2*(image1 / 255.0)-1.0
        image2 = 2*(image2 / 255.0)-1.0

        # d1/d2 是 disparity：先用 Z = bf / disp 反投影到相机坐标系点图，
        # 再基于 point/normal/valid 做 mask-aware 下采样。
        bf = getattr(self.args, "bf", 500.0)
        surface_pyramid = build_surface_pyramid_from_disparity_pair(
            d1, d2, intrinsics, bf=bf, scales=(16, 8, 4)
        )
        surface_16x = surface_pyramid["1/16"]
        surface_8x = surface_pyramid["1/8"]
        surface_4x = surface_pyramid["1/4"]

        d1_4x = F.interpolate(d1, scale_factor=1 / 4, mode='bilinear')
        d1_8x = F.interpolate(d1, scale_factor=1 / 8, mode='bilinear')
        d1_16x = F.interpolate(d1, scale_factor=1 / 16, mode='bilinear')

        d1 = 2*(d1/ 128)-1.0
        d2 = 2*(d2/ 128)-1.0

        imaged1 =torch.cat([image1,d1],dim=1)
        imaged2 =torch.cat([image2,d2],dim=1)

        imaged1 = imaged1.contiguous()
        imaged2 = imaged2.contiguous()
        time_x16=[]
        time_x8 = []
        time_x4 = []
        time_feature = []
        # run the context network
        time_e_s = time.time()
        # run the feature network,这里这个特征提取就要全出了
        with autocast(enabled=self.args.mixed_precision):
            fnet_pyramid = self.fnet([image1, image2])

        zeroP16 = torch.zeros_like(surface_16x["P2"])
        zeroP8 = torch.zeros_like(surface_8x["P2"])
        zeroP4 = torch.zeros_like(surface_4x["P2"])

        corr_fp2x16 = CudaBroadCorrBlockNs(zeroP16, surface_16x["P2"], radius=6)
        corr_fp2x8 = CudaBroadCorrBlockNs(zeroP8, surface_8x["P2"], radius=4)
        corr_fp2x4 = CudaBroadCorrBlockNs(zeroP4, surface_4x["P2"], radius=4)

        zero16 = torch.zeros_like(surface_16x["N2"])
        zero8 = torch.zeros_like(surface_8x["N2"])
        zero4 = torch.zeros_like(surface_4x["N2"])

        corr_fnnx16 =CudaBroadCorrBlockNs(zero16, surface_16x["N2"],radius=6)
        corr_fnnx8 = CudaBroadCorrBlockNs(zero8, surface_8x["N2"], radius=4)
        corr_fnnx4 = CudaBroadCorrBlockNs(zero4, surface_4x["N2"], radius=4)

        corr_fn16 = CudaBroadCorrBlockSG(fnet_pyramid[0][0].float(), fnet_pyramid[0][1].float(), radius=6)
        corr_fn8 = CudaBroadCorrBlockSG(fnet_pyramid[1][0].float(), fnet_pyramid[1][1].float(), radius=4)
        corr_fn4 = CudaBroadCorrBlockSG(fnet_pyramid[2][0].float(), fnet_pyramid[2][1].float(), radius=4)

        with autocast(enabled=self.args.mixed_precision):
            cnet_pyramid = self.cnet(torch.cat([imaged1, imaged2], dim=1))
            net16, inp16 = torch.split(cnet_pyramid[0], [self.args.dim, self.args.dim], dim=1)
            net16 = torch.tanh(net16)
            inp16 = torch.relu(inp16)
            net8, inp8 = torch.split(cnet_pyramid[1], [self.args.dim, self.args.dim], dim=1)
            net8 = torch.tanh(net8)
            inp8 = torch.relu(inp8)
            net4, inp4 = torch.split(cnet_pyramid[2], [self.args.dim, self.args.dim], dim=1)
            net4 = torch.tanh(net4)
            inp4 = torch.relu(inp4)

            inp2 = torch.relu(cnet_pyramid[3])
            up_mask2 = self.mask2(inp2)
            up_mask4 = self.mask4(inp4)
            up_mask8 = self.mask8(inp8)
        time_e_e = time.time()
        time_feature.append(time_e_e-time_e_s)


        x16coords0, x16coords1 = self.initialize_flow16x(image1)
        d1_4xu = d1_4x.detach().clone()
        d1_8xu = d1_8x.detach().clone()
        d1_16xu = d1_16x.detach().clone()
        d2_16xu = d1_16x.detach().clone()
        flow2d_predictions = []
        dz_predictions = []
        #首先在1/16的尺度上进行初始化，迭代三次
        for itr in range(2):
            #todo 首先拿更新的参数量
            x16coords1 = x16coords1.detach()
            d2_16xu = d2_16xu.detach()

            time_c_16s = time.time()
            #todo 然后索引相关性代价和三维代价,融合代价（这个代价如果直接搞在二维平面上会方便很多，而且本质上似乎没什么区别）

            corr = corr_fn16(x16coords1)
            flow2d = x16coords1-x16coords0
            depthc = d2_16xu-d1_16xu
            motion_features = self.encode_p2p_motion(
                x16coords1, d2_16xu, surface_16x, corr_fp2x16, corr_fnnx16,
                corr, flow2d,depthc, radius=6,
                flow_encoder=self.motionenc_flow16, svs_encoder=self.motionenc_svs16)
            time_c_16e = time.time()


            #应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            #todo 迭代优化器（Unet版本）

            time_u_16s = time.time()
            with autocast(enabled=self.args.mixed_precision):
                net16, up_mask, delta_flow,delta_dz = self.update_blockx16(net16, inp16, motion_features)
            time_u_16e = time.time()

            x16coords1 = x16coords1+delta_flow
            d2_16xu = d2_16xu+delta_dz

            time_up_16s = time.time()
            flow_up8 = self.upsample_flow(x16coords1 - x16coords0, up_mask, scale=2)
            dz_up8 = self.upsample_dz(d2_16xu-d1_16xu, up_mask, scale=2)
            flow_up4 = self.upsample_flow(flow_up8, up_mask8, scale=2)
            dz_up4 = self.upsample_dz(dz_up8, up_mask8, scale=2)
            flow_up2 = self.upsample_flow(flow_up4, up_mask4, scale=2)
            dz_up2 = self.upsample_dz(dz_up4, up_mask4, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)
            time_up_16e = time.time()

            time_x16.append([time_c_16e-time_c_16s,time_u_16e-time_u_16s,time_up_16e-time_up_16s])


            flow2d_predictions.append(flow_up)#RAFT3D监督的深度变化量的差值 d确实是与第一帧的差值，f就是光流
            dz_predictions.append(dz_up)


        #todo 准备初始化的衔接数据 1/8尺度
        flow_up8 = self.upsample_flow(x16coords1 - x16coords0, up_mask, scale=2)
        dz_up8 = self.upsample_dz(d2_16xu - d1_16xu, up_mask, scale=2)
        x8coords0, x8coords1 = self.initialize_flowNx(image1,8)
        x8coords1 = x8coords1+flow_up8
        d2_8xu =  d1_8xu + dz_up8

        for itr in range(4):
            #todo 首先拿更新的参数量
            x8coords1 = x8coords1.detach()
            d2_8xu = d2_8xu.detach()

            time_c_8s = time.time()
            #todo 然后索引相关性代价和三维代价,融合代价（这个代价如果直接搞在二维平面上会方便很多，而且本质上似乎没什么区别）
            corr = corr_fn8(x8coords1)#3d索引向量
            flow2d = x8coords1-x8coords0
            depthc = d2_8xu-d1_8xu
            motion_features = self.encode_p2p_motion(
                x8coords1, d2_8xu, surface_8x, corr_fp2x8, corr_fnnx8,
                corr, flow2d, depthc, radius=4,
                flow_encoder=self.motionenc_flow, svs_encoder=self.motionenc_svs)
            time_c_8e = time.time()

            #应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）

            time_u_8s = time.time()
            #todo 迭代优化器（Unet版本）
            with autocast(enabled=self.args.mixed_precision):
                net8, up_mask, delta_flow,delta_dz = self.update_block(net8, inp8, motion_features)
            x8coords1 = x8coords1+delta_flow
            d2_8xu = d2_8xu+delta_dz
            time_u_8e = time.time()

            time_up_8s = time.time()
            flow_up4 = self.upsample_flow(x8coords1 - x8coords0, up_mask, scale=2)
            dz_up4 = self.upsample_dz(d2_8xu-d1_8xu, up_mask, scale=2)
            flow_up2 = self.upsample_flow(flow_up4, up_mask4, scale=2)
            dz_up2 = self.upsample_dz(dz_up4, up_mask4, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)
            time_up_8e = time.time()

            time_x8.append([time_c_8e - time_c_8s, time_u_8e - time_u_8s, time_up_8e - time_up_8s])

            flow2d_predictions.append(flow_up)#RAFT3D监督的深度变化量的差值 d确实是与第一帧的差值，f就是光流
            dz_predictions.append(dz_up)

        #todo 准备初始化的衔接数据 1/4尺度
        flow_up4 = self.upsample_flow(x8coords1 - x8coords0, up_mask, scale=2)
        dz_up4 = self.upsample_dz(d2_8xu - d1_8xu, up_mask, scale=2)
        x4coords0, x4coords1 = self.initialize_flowNx(image1,4)
        x4coords1 = x4coords1+flow_up4
        d2_4xu =  d1_4xu + dz_up4

        for itr in range(3):
            # todo 首先拿更新的参数量
            x4coords1 = x4coords1.detach()
            d2_4xu = d2_4xu.detach()

            time_c_4s = time.time()
            # todo 然后索引相关性代价和三维代价,融合代价（这个代价如果直接搞在二维平面上会方便很多，而且本质上似乎没什么区别）
            corr = corr_fn4(x4coords1)  # 3d索引向量
            flow2d = x4coords1 - x4coords0
            depthc = d2_4xu - d1_4xu
            motion_features = self.encode_p2p_motion(
                x4coords1, d2_4xu, surface_4x, corr_fp2x4, corr_fnnx4,
                corr, flow2d, depthc, radius=4,
                flow_encoder=self.motionenc_flow, svs_encoder=self.motionenc_svs)
            time_c_4e = time.time()

            # 应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            # todo 迭代优化器（Unet版本）

            time_u_4s = time.time()
            with autocast(enabled=self.args.mixed_precision):
                net4, up_mask, delta_flow, delta_dz = self.update_block(net4, inp4, motion_features)
            time_u_4e = time.time()
            x4coords1 = x4coords1 + delta_flow
            d2_4xu = d2_4xu + delta_dz

            time_up_4s = time.time()
            flow_up2 = self.upsample_flow(x4coords1 - x4coords0, up_mask, scale=2)
            dz_up2 = self.upsample_dz(d2_4xu - d1_4xu, up_mask, scale=2)

            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)
            time_up_4e = time.time()

            time_x4.append([time_c_4e - time_c_4s, time_u_4e - time_u_4s, time_up_4e - time_up_4s])
            flow2d_predictions.append(flow_up)  # RAFT3D监督的深度变化量的差值 d确实是与第一帧的差值，f就是光流
            dz_predictions.append(dz_up)
        if test_mode:
            return flow_up, flow_up, dz_up
        return flow2d_predictions,dz_predictions
