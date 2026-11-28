#corr 还要把深度转换为x,y,z， 然后把x,y,z转为u,v,d 来采样
#这个方法的核心特点是，采样相关性特征的时候同时采样空间特性，搞一个x,y,z向量+v（像素相关性），然后将其投影到2D空间做训练
#todo 把整个网络放到1/4尺度上运行,我担心的事情是这个缺乏一个全局的视野；核心思想减少采样次数/多尺度多级计算
#todo 现在这个版本还是先在相机坐标空间中搞吧（u,v,z）还有一个版本是（u,v,1/z）话说回来那还上啥真深度啊，直接上视差不就完事了
#todo 搞一个估计深度比例变化的版本，现在这个直接估计深度变化量，感觉结果很糊很糊，是不是因为这个量级太小了，网络很难精细化估计
#todo 读取一下vkitti2 搞一个评估吧
#如果要评估这个效果必须得有哥评估标准
#将d放大后感觉估计的效果好了很多，后面试试在1/16和1/4尺度上估计
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.update_raft import BasicUpdateBlock
from core.extractor import ResNetFPN
from core.corr import CorrBlock,CorrBlock3D
from core.utils.flow_viz import flow2rgb
from core.utils.utils import coords_grid, upflow2,updepth2
from core.utils.xcit import XCiT
from model_home.broad_corr import CudaBroadCorrBlock3d,CudaBroadCorrBlock
from model_home.update import RF3DuvdUpdateBlockwdv3ab,RF3DuvdUpdateBlock_16dv3ab
from core.utils.submodule import Tiny_Unetlikev7
from model_home.extractor import BasicEncoder_resconv, Basic_Context_Encoder_resconvb
#首先是1/16尺度上的光流，然后上采样到1/8尺度，然后上采样到1/4尺度，然后再上采样到1/2尺度？？？？
'''
CUDA_VISIBLE_DEVICES=0,1 python train_gm_occ.py --name raft_nerf_8k --stage nerf --validation kitti --restore_ckpt /home/lh/CSCV_occ/checkpoints/80000_raft_sintel_conf.pth --gpus 0 1 --num_steps 80000 --batch_size 12 --lr 0.0005 --image_size 256 768 --wdecay 0.0001 --gamma=0.85 --traneval=0
Validation clean Sintel: 3.463586, 8.519480
Validation final Sintel: 6.909781, 13.919200
'''
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

#我在考虑这个fx和这个fy干脆固定下来得了，cx,cy，还有向量的边缘区域全部归0
def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:, None, None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    #d = 1.0 / Z
    d =  Z
    coords = torch.stack([x, y, d], dim=-1)
    return coords


def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """
    depths = depths[:,0,:,:]
    ht, wd = depths.shape[-2:]

    fx, fy, cx, cy = \
        intrinsics[:, None, None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(),
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)

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
class RFlow3D_MSab_v8(nn.Module):
    def __init__(self, args):
        super(RFlow3D_MSab_v8, self).__init__()
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
        self.update_blockx16 = RF3DuvdUpdateBlock_16dv3ab(self.args, hidden_dim=hdim)#1/16尺度的当作第一个，用于初始化
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

        corr_fndx16 = CudaBroadCorrBlock3d(X1_16x, X2_16x, radius=6)
        corr_fndx8 = CudaBroadCorrBlock3d(X1_8x, X2_8x, radius=4)
        corr_fndx4 = CudaBroadCorrBlock3d(X1_4x, X2_4x, radius=4)

        corr_fn16 = CudaBroadCorrBlock(fnet_pyramid[0][0].float(), fnet_pyramid[0][1].float(), radius=6)
        corr_fn8 = CudaBroadCorrBlock(fnet_pyramid[1][0].float(), fnet_pyramid[1][1].float(), radius=4)
        corr_fn4 = CudaBroadCorrBlock(fnet_pyramid[2][0].float(), fnet_pyramid[2][1].float(), radius=4)

        # run the context network
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
            #todo 然后索引相关性代价和三维代价,融合代价（这个代价如果直接搞在二维平面上会方便很多，而且本质上似乎没什么区别）
            corr3d = corr_fndx16(x16coords1) # index correlation volume
            corr = corr_fn16(x16coords1)#3d索引向量
            flow2d = x16coords1-x16coords0
            depthc = d2_16xu-d1_16xu

            corrd = corr3d[:,1,:,:,:]
            #应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            #todo 迭代优化器（Unet版本）
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow,delta_dz = self.update_blockx16(net16, inp16,corrd,corr, flow2d,depthc)
            x16coords1 = x16coords1+delta_flow
            d2_16xu = d2_16xu+delta_dz
            flow_up8 = self.upsample_flow(x16coords1 - x16coords0, up_mask, scale=2)
            dz_up8 = self.upsample_dz(d2_16xu-d1_16xu, up_mask, scale=2)
            flow_up4 = self.upsample_flow(flow_up8, up_mask8, scale=2)
            dz_up4 = self.upsample_dz(dz_up8, up_mask8, scale=2)
            flow_up2 = self.upsample_flow(flow_up4, up_mask4, scale=2)
            dz_up2 = self.upsample_dz(dz_up4, up_mask4, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)

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
            #todo 然后索引相关性代价和三维代价,融合代价（这个代价如果直接搞在二维平面上会方便很多，而且本质上似乎没什么区别）
            corr3d = corr_fndx8(x8coords1) # index correlation volume
            corr = corr_fn8(x8coords1)#3d索引向量
            flow2d = x8coords1-x8coords0
            depthc = d2_8xu-d1_8xu

            corrd = corr3d[:,1, :, :, :]
            #应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            #todo 迭代优化器（Unet版本）
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow,delta_dz = self.update_block(net8, inp8,corrd ,corr, flow2d,depthc)
            x8coords1 = x8coords1+delta_flow
            d2_8xu = d2_8xu+delta_dz

            flow_up4 = self.upsample_flow(x8coords1 - x8coords0, up_mask, scale=2)
            dz_up4 = self.upsample_dz(d2_8xu-d1_8xu, up_mask, scale=2)
            flow_up2 = self.upsample_flow(flow_up4, up_mask4, scale=2)
            dz_up2 = self.upsample_dz(dz_up4, up_mask4, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)

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
            # todo 然后索引相关性代价和三维代价,融合代价（这个代价如果直接搞在二维平面上会方便很多，而且本质上似乎没什么区别）
            corr3d = corr_fndx4(x4coords1) # index correlation volume
            corr = corr_fn4(x4coords1)  # 3d索引向量
            flow2d = x4coords1 - x4coords0
            depthc = d2_4xu - d1_4xu

            corrd = corr3d[:,1, :, :, :]
            # 应该是采样的第二帧（映射深度）   减去    估计的第二帧（映射深度）
            # todo 迭代优化器（Unet版本）
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, delta_dz = self.update_block(net4, inp4,corrd, corr, flow2d, depthc)
            x4coords1 = x4coords1 + delta_flow
            d2_4xu = d2_4xu + delta_dz

            flow_up2 = self.upsample_flow(x4coords1 - x4coords0, up_mask, scale=2)
            dz_up2 = self.upsample_dz(d2_4xu - d1_4xu, up_mask, scale=2)

            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)


            flow2d_predictions.append(flow_up)  # RAFT3D监督的深度变化量的差值 d确实是与第一帧的差值，f就是光流
            dz_predictions.append(dz_up)

        if test_mode:
            return flow_up, flow_up, dz_up
        return flow2d_predictions,dz_predictions
