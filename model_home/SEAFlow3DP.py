import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from core.utils.utils import coords_grid, upflow2,updepth2
from core.utils.xcit import XCiT
from model_home.broad_corr import CudaBroadCorrBlock3d,CudaBroadCorrBlock
from model_home.update import RF3DuvdUpdateBlockwdv3ab,RF3DuvdUpdateBlock_16dv3ab
from model_home.extractor import BasicEncoder_resconv, Basic_Context_Encoder_resconvb
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

class BasicCorrEncoder_catconv(nn.Module):
    def __init__(self, corr_radius=6):
        super(BasicCorrEncoder_catconv, self).__init__()
        cor_planes = (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 168, 1, padding=0)
        self.convc2 = nn.Conv2d(cor_planes, 168, 1, padding=0)
        self.xcitcorr1 = XCiT(embed_dim=168, depth=1, num_heads=8, mlp_ratio=1, separate=False)
        self.xcitcorr2 = XCiT(embed_dim=168, depth=1, num_heads=8, mlp_ratio=1, separate=False)
        self.convc3 = nn.Conv2d(168*2, 128, 1, padding=0)
    def forward(self, corr1,corr2):
        cor1 = F.relu(self.convc1(corr1))
        cor2 = F.relu(self.convc2(corr2))
        cor1 = self.xcitcorr1(cor1)
        cor2 = self.xcitcorr1(cor2)
        cor3 = F.relu(self.convc3(torch.cat([cor1,cor2],dim=1)))
        return cor3


# 这里加上一个快速初始化，xict互相关特征,采样相关性，全局卷积，迭代两次到三次
class SEAFLOW3DP(nn.Module):
    def __init__(self, args):
        super(SEAFLOW3DP, self).__init__()
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
        self.update_block = RF3DuvdUpdateBlockwdv3ab(self.args, hidden_dim=hdim)#后续的迭代模块
        self.update_blockx16 = RF3DuvdUpdateBlock_16dv3ab(self.args, hidden_dim=hdim)
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

    def forward(self, image1, image2, d1,d2,iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        #搞一个参考RAFT3D版本的，u,v,d
        #在之前的实验中初始化部件搞三次是最好的，这部分要单独分出来
        image1 = 2*(image1 / 255.0)-1.0
        image2 = 2*(image2 / 255.0)-1.0

        d1_4x = F.interpolate(d1, scale_factor=1 / 4, mode='bilinear')
        d2_4x = F.interpolate(d2, scale_factor=1 / 4, mode='bilinear')

        d1_8x = F.interpolate(d1, scale_factor=1 / 8, mode='bilinear')
        d2_8x = F.interpolate(d2, scale_factor=1 / 8, mode='bilinear')

        d1_16x = F.interpolate(d1, scale_factor=1 / 16, mode='bilinear')
        d2_16x = F.interpolate(d2, scale_factor=1 / 16, mode='bilinear')

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

        X1_4x = torch.cat([d1_4x,d1_4x,torch.zeros_like(d1_4x)],dim=1)
        X2_4x = torch.cat([d1_4x,d2_4x, d2_4x], dim=1)

        X1_8x = torch.cat([d1_8x,d1_8x,torch.zeros_like(d1_8x)],dim=1)
        X2_8x = torch.cat([d1_8x,d2_8x, d2_8x], dim=1)

        X1_16x = torch.cat([d1_16x,d1_16x,torch.zeros_like(d1_16x)],dim=1)
        X2_16x = torch.cat([d1_16x,d2_16x, d2_16x], dim=1)
        # 用于初始化的超大核卷积

        corr_fndx16 =CudaBroadCorrBlock3d(X1_16x,X2_16x,radius=6)
        corr_fndx8 = CudaBroadCorrBlock3d(X1_8x, X2_8x, radius=4)
        corr_fndx4 = CudaBroadCorrBlock3d(X1_4x, X2_4x, radius=4)

        corr_fn16 = CudaBroadCorrBlock(fnet_pyramid[0][0].float(), fnet_pyramid[0][1].float(), radius=6)
        corr_fn8 = CudaBroadCorrBlock(fnet_pyramid[1][0].float(), fnet_pyramid[1][1].float(), radius=4)
        corr_fn4 = CudaBroadCorrBlock(fnet_pyramid[2][0].float(), fnet_pyramid[2][1].float(), radius=4)
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
            corr3d = corr_fndx16(x16coords1) # index correlation volume
            corr = corr_fn16(x16coords1)#3d索引向量
            flow2d = x16coords1-x16coords0
            depthc = d2_16xu-d1_16xu
            time_c_16e = time.time()


            corrd = corr3d[:,1,:,:,:]
            #应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            #todo 迭代优化器（Unet版本）

            time_u_16s = time.time()
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow,delta_dz = self.update_blockx16(net16, inp16,corrd,corr, flow2d,depthc)
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
            corr3d = corr_fndx8(x8coords1) # index correlation volume
            corr = corr_fn8(x8coords1)#3d索引向量
            flow2d = x8coords1-x8coords0
            depthc = d2_8xu-d1_8xu
            time_c_8e = time.time()

            corrd = corr3d[:,1, :, :, :]
            #应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）

            time_u_8s = time.time()
            #todo 迭代优化器（Unet版本）
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow,delta_dz = self.update_block(net8, inp8,corrd ,corr, flow2d,depthc)
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
            corr3d = corr_fndx4(x4coords1) # index correlation volume
            corr = corr_fn4(x4coords1)  # 3d索引向量
            flow2d = x4coords1 - x4coords0
            depthc = d2_4xu - d1_4xu
            time_c_4e = time.time()

            corrd = corr3d[:,1, :, :, :]
            # 应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            # todo 迭代优化器（Unet版本）

            time_u_4s = time.time()
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, delta_dz = self.update_block(net4, inp4,corrd, corr, flow2d, depthc)
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
