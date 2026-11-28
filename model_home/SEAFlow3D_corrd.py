"""SEA-Flow3D with the depth-change residual (v2.1, 2026-11; `--model seaflow3d_cd`, weights of the v2.1 release).

Same network as `SEAFLOW3D` in model_home/SEAFlow3D.py (same layers and layer names, hidden state carried across
iterations, iterations 2 / 4 / 3 at 1/16, 1/8 and 1/4 resolution). The only change is the depth-change cue that the
update block receives at all three scales:

    SEAFLOW3D       corrd = corr3d[:, 1]
    SEAFLOW3DCorrd  corrd = corr3d[:, 1] - depthc

corr3d is the spatial vector correlation of X1 = [d1, d1, 0] and X2 = [d1, d2, d2] (channel-wise difference, not scaled),
so corr3d[:, 1] = d2(candidate) - d1, the disparity change implied by each candidate match. depthc = d2u - d1 is the
current estimate of the disparity change. Both are in full-resolution disparity units, so the difference is the residual
d2(candidate) - d2u of each candidate with respect to the current estimate.

Because the layer names are the same, v2.0 checkpoints (trained for SEAFLOW3D) also load into this class with
strict=True, and v2.1 checkpoints load into SEAFLOW3D: always pair each checkpoint with the model it was trained for.
"""
import torch

from model_home.SEAFlow3D import SEAFLOW3D, autocast
from model_home.broad_corr import CudaBroadCorrBlock3d, CudaBroadCorrBlock
import torch.nn.functional as F


class SEAFLOW3DCorrd(SEAFLOW3D):
    corrd_scale = 1.0

    def forward(self, image1, image2, d1, d2, intrinsics=None,
                iters=12, flow_init=None, upsample=True, test_mode=False):
        del intrinsics
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        d1_4x = F.interpolate(d1, scale_factor=1 / 4, mode='bilinear')
        d2_4x = F.interpolate(d2, scale_factor=1 / 4, mode='bilinear')
        d1_8x = F.interpolate(d1, scale_factor=1 / 8, mode='bilinear')
        d2_8x = F.interpolate(d2, scale_factor=1 / 8, mode='bilinear')
        d1_16x = F.interpolate(d1, scale_factor=1 / 16, mode='bilinear')
        d2_16x = F.interpolate(d2, scale_factor=1 / 16, mode='bilinear')

        d1 = 2 * (d1 / 128) - 1.0
        d2 = 2 * (d2 / 128) - 1.0
        imaged1 = torch.cat([image1, d1], dim=1).contiguous()
        imaged2 = torch.cat([image2, d2], dim=1).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            fnet_pyramid = self.fnet([image1, image2])

        X1_4x = torch.cat([d1_4x, d1_4x, torch.zeros_like(d1_4x)], dim=1)
        X2_4x = torch.cat([d1_4x, d2_4x, d2_4x], dim=1)
        X1_8x = torch.cat([d1_8x, d1_8x, torch.zeros_like(d1_8x)], dim=1)
        X2_8x = torch.cat([d1_8x, d2_8x, d2_8x], dim=1)
        X1_16x = torch.cat([d1_16x, d1_16x, torch.zeros_like(d1_16x)], dim=1)
        X2_16x = torch.cat([d1_16x, d2_16x, d2_16x], dim=1)

        corr_fndx16 = CudaBroadCorrBlock3d(X1_16x, X2_16x, radius=6)
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

        x16coords0, x16coords1 = self.initialize_flow16x(image1)
        d1_4xu = d1_4x.detach().clone()
        d1_8xu = d1_8x.detach().clone()
        d1_16xu = d1_16x.detach().clone()
        d2_16xu = d1_16x.detach().clone()
        flow2d_predictions = []
        dz_predictions = []

        # 1/16 resolution, 2 iterations
        for itr in range(2):
            x16coords1 = x16coords1.detach()
            d2_16xu = d2_16xu.detach()
            corr3d = corr_fndx16(x16coords1)
            corr = corr_fn16(x16coords1)
            flow2d = x16coords1 - x16coords0
            depthc = d2_16xu - d1_16xu
            corrd = corr3d[:, 1, :, :, :] * self.corrd_scale - depthc   # residual: disparity change of each candidate minus the current estimate
            with autocast(enabled=self.args.mixed_precision):
                net16, up_mask, delta_flow, delta_dz = self.update_blockx16(net16, inp16, corrd, corr, flow2d, depthc)   # hidden state carried to the next iteration
            x16coords1 = x16coords1 + delta_flow
            d2_16xu = d2_16xu + delta_dz

            flow_up8 = self.upsample_flow(x16coords1 - x16coords0, up_mask, scale=2)
            dz_up8 = self.upsample_dz(d2_16xu - d1_16xu, up_mask, scale=2)
            flow_up4 = self.upsample_flow(flow_up8, up_mask8, scale=2)
            dz_up4 = self.upsample_dz(dz_up8, up_mask8, scale=2)
            flow_up2 = self.upsample_flow(flow_up4, up_mask4, scale=2)
            dz_up2 = self.upsample_dz(dz_up4, up_mask4, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)
            flow2d_predictions.append(flow_up)
            dz_predictions.append(dz_up)

        # 1/8 resolution, 4 iterations
        flow_up8 = self.upsample_flow(x16coords1 - x16coords0, up_mask, scale=2)
        dz_up8 = self.upsample_dz(d2_16xu - d1_16xu, up_mask, scale=2)
        x8coords0, x8coords1 = self.initialize_flowNx(image1, 8)
        x8coords1 = x8coords1 + flow_up8
        d2_8xu = d1_8xu + dz_up8

        for itr in range(4):
            x8coords1 = x8coords1.detach()
            d2_8xu = d2_8xu.detach()
            corr3d = corr_fndx8(x8coords1)
            corr = corr_fn8(x8coords1)
            flow2d = x8coords1 - x8coords0
            depthc = d2_8xu - d1_8xu
            corrd = corr3d[:, 1, :, :, :] * self.corrd_scale - depthc   # same residual (1/8 and 1/4 share update_block)
            with autocast(enabled=self.args.mixed_precision):
                net8, up_mask, delta_flow, delta_dz = self.update_block(net8, inp8, corrd, corr, flow2d, depthc)   # hidden state carried to the next iteration
            x8coords1 = x8coords1 + delta_flow
            d2_8xu = d2_8xu + delta_dz

            flow_up4 = self.upsample_flow(x8coords1 - x8coords0, up_mask, scale=2)
            dz_up4 = self.upsample_dz(d2_8xu - d1_8xu, up_mask, scale=2)
            flow_up2 = self.upsample_flow(flow_up4, up_mask4, scale=2)
            dz_up2 = self.upsample_dz(dz_up4, up_mask4, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)
            flow2d_predictions.append(flow_up)
            dz_predictions.append(dz_up)

        # 1/4 resolution, 3 iterations
        flow_up4 = self.upsample_flow(x8coords1 - x8coords0, up_mask, scale=2)
        dz_up4 = self.upsample_dz(d2_8xu - d1_8xu, up_mask, scale=2)
        x4coords0, x4coords1 = self.initialize_flowNx(image1, 4)
        x4coords1 = x4coords1 + flow_up4
        d2_4xu = d1_4xu + dz_up4

        for itr in range(3):
            x4coords1 = x4coords1.detach()
            d2_4xu = d2_4xu.detach()
            corr3d = corr_fndx4(x4coords1)
            corr = corr_fn4(x4coords1)
            flow2d = x4coords1 - x4coords0
            depthc = d2_4xu - d1_4xu
            corrd = corr3d[:, 1, :, :, :] * self.corrd_scale - depthc   # same residual
            with autocast(enabled=self.args.mixed_precision):
                net4, up_mask, delta_flow, delta_dz = self.update_block(net4, inp4, corrd, corr, flow2d, depthc)   # hidden state carried to the next iteration
            x4coords1 = x4coords1 + delta_flow
            d2_4xu = d2_4xu + delta_dz

            flow_up2 = self.upsample_flow(x4coords1 - x4coords0, up_mask, scale=2)
            dz_up2 = self.upsample_dz(d2_4xu - d1_4xu, up_mask, scale=2)
            flow_up = self.upsample_flow(flow_up2, up_mask2, scale=2)
            dz_up = self.upsample_dz(dz_up2, up_mask2, scale=2)
            flow2d_predictions.append(flow_up)
            dz_predictions.append(dz_up)
        if test_mode:
            return flow_up, flow_up, dz_up
        return flow2d_predictions, dz_predictions
