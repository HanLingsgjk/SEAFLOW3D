import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.xcit import XCiT
import sys
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class RefineHead(nn.Module):
    def __init__(self, input_dim=4*128, hidden_dim=128):
        super(RefineHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

#todo 跨尺度状态更新，在跨越尺度时延续Net(???)
from core.layer import ConvNeXtV2tiny
class NextConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(NextConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 5, padding=2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 5, padding=2)
        self.convq = ConvNeXtV2tiny(hidden_dim+input_dim, hidden_dim, kszie=5, bl=2)
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h



class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256+128):  # input dim=256 + 128, hdim=64
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))

    # h:net:128, x = 128 + 128 = 256
    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, correlation_depth, stack_coords=False):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(correlation_depth, 256, 1, padding=0)
        self.stack_coords = stack_coords
        # self.convc1 = nn.Conv2d(324, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        if not stack_coords:
            self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

        else:
            self.conv_coords_x_1 = nn.Conv2d(correlation_depth, 64, 3, padding=1)
            self.conv_coords_x_2 = nn.Conv2d(64, 64, 3, padding=1)

            self.conv_coords_y_1 = nn.Conv2d(correlation_depth, 64, 3, padding=1)
            self.conv_coords_y_2 = nn.Conv2d(64, 64, 3, padding=1)

            self.conv_corr_coords = nn.Conv2d(192 + 128, 256, 3, padding=1)
            self.conv_corr_coords_flow_1 = nn.Conv2d(256 + 64, 256, 3, padding=1)
            self.conv_corr_coords_flow_2 = nn.Conv2d(256, 128 - 2, 1, padding=0)

    def forward(self, flow, corr, coords_x=None, coords_y=None):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        if not self.stack_coords:
            cor_flo = torch.cat([cor, flo], dim=1)
            out = F.relu(self.conv(cor_flo))
        else:
            conved_coords_x_1 = F.relu(self.conv_coords_x_1(coords_x))
            conved2_coords_x = F.relu(self.conv_coords_x_2(conved_coords_x_1))

            conved_coords_y_1 = F.relu(self.conv_coords_y_1(coords_y))
            conved2_coords_y = F.relu(self.conv_coords_y_2(conved_coords_y_1))

            conved_corr_coords = F.relu(self.conv_corr_coords(torch.cat([conved2_coords_x, conved2_coords_y, cor], dim=1)))
            conved_corr_coords_flow_1 = F.relu(self.conv_corr_coords_flow_1(torch.cat([conved_corr_coords, flo], dim=1)))
            out = F.relu(self.conv_corr_coords_flow_2(conved_corr_coords_flow_1))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, correlation_depth, stack_coords=False, hidden_dim=128, input_dim=256, scale=8):
        super(BasicUpdateBlock, self).__init__()
        # hidden_dim = 64
        # input_dim = 192
        self.args = args
        self.encoder = BasicMotionEncoder(correlation_depth, stack_coords=stack_coords)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale*scale*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, coords_x=None, coords_y=None, level_index=0):
        # net: 128
        # inp depth: 128
        motion_features = self.encoder(flow, corr, coords_x, coords_y)  # motion feature depth: 128
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)  # output:net.depth:128

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
class BasicUpdateBlockCCMR(nn.Module):
    def __init__(self, args, correlation_depth, hidden_dim=128, input_dim=256, scale=8, num_heads=8, depth=1,
                 mlp_ratio=1, num_scales=4):
        super(BasicUpdateBlockCCMR, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(correlation_depth)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + 2*hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale * scale * 9, 1, padding=0))

        self.aggregator = nn.ModuleList(
            [XCiT(embed_dim=128, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, separate=True)])
        for i in range(num_scales - 1):
            self.aggregator.extend(
                [XCiT(embed_dim=128, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, separate=True)])

    def forward(self, net, inp, corr, flow, global_context, level_index=0):
        motion_features = self.encoder(flow, corr)  # motion feature depth: 128
        motion_features_global = self.aggregator[level_index](global_context, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
#运动编码，corr3d用xcit编码，然后三维运动普通编码
class MotioneEncoderRF3D(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3D, self).__init__()
        cor_planes = 162*4#343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 216, 3, padding=1)
        self.xcitx16 = XCiT(embed_dim=216, depth=1, num_heads=8, mlp_ratio=1, separate=False)

        self.convf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = ConvNeXtV2tiny(216+64, 128-3, kszie=7, bl=2)

    def forward(self, flow3d, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor3d = self.xcitx16(cor)

        flo = F.relu(self.convf1(flow3d))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor3d, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow3d], dim=1)
#运动编码，corr3d用xcit编码，然后三维运动普通编码
class MotioneEncoderRF3Duvd(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd, self).__init__()
        cor_planes = 162#343#411
        self.convc1 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.convc2 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.convc3 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.xcitc1 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)
        self.xcitc2 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)
        self.xcitc3 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(96+192+64+32, 128-3, kszie=7, bl=2)

    def forward(self, corrdf,corr3d,corr, flow2d,depthc):

        cord = self.xcitc1(F.relu(self.convc1(corr3d)))
        corc = self.xcitc2(F.relu(self.convc2(corr)))
        corrdfo = self.xcitc3(F.relu(self.convc3(corrdf)))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corrdfo,cord,corc,flo,dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d,depthc], dim=1)
#运动编码，corr3d用xcit编码，然后三维运动普通编码
class MotioneEncoderRF3Duvd_wod(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wod, self).__init__()
        cor_planes = 162#343#411
        self.convc1 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.xcitc1 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(96+64+32, 128-3, kszie=7, bl=2)

    def forward(self, corr, flow2d,depthc):

        cord = self.xcitc1(F.relu(self.convc1(corr)))
        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([cord,flo,dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_16d(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_16d, self).__init__()
        cor_planes = 338#343#411
        self.convc1 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.xcitc1 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(96+64+32, 128-3, kszie=7, bl=2)

    def forward(self, corr, flow2d,depthc):

        cord = self.xcitc1(F.relu(self.convc1(corr)))
        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([cord,flo,dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_16(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_16, self).__init__()
        cor_planes = 162#343#411
        self.convc1 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.xcitc1 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(96+64+32, 128-3, kszie=7, bl=2)

    def forward(self, corr, flow2d,depthc):

        cord = self.xcitc1(F.relu(self.convc1(corr)))
        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([cord,flo,dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_wod16(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wod16, self).__init__()
        cor_planes = 338#343#411
        self.convc1 = nn.Conv2d(cor_planes, 128, 3, padding=1)
        self.xcitc1 = XCiT(embed_dim=128, depth=1, num_heads=8, mlp_ratio=1, separate=False)
        self.convc2 = nn.Conv2d(cor_planes, 96, 3, padding=1)
        self.xcitc2 = XCiT(embed_dim=96, depth=1, num_heads=8, mlp_ratio=1, separate=False)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(128+96+64+32, 192-3, kszie=7, bl=2)

    def forward(self,corrd, corr, flow2d,depthc):

        cord = self.xcitc1(F.relu(self.convc1(corr)))
        corr = self.xcitc2(F.relu(self.convc2(corrd)))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corr,cord,flo,dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_wod16v1(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wod16v1, self).__init__()
        cor_planes = ((args.itr1 * 2 + 1) ** 2) * 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(128+96+64+32, 192-3, kszie=7, bl=2)

    def forward(self,corrd, corr, flow2d,depthc):

        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))


        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corrdo,corro,flo,dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_wod16v3(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wod16v3, self).__init__()
        cor_planes = ((args.itr1 * 2 + 1) ** 2) * 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_flo = ConvNeXtV2tiny(128+32, 128-2, kszie=7, bl=2)
        self.conv_dpz = ConvNeXtV2tiny(96+32, 64-1, kszie=7, bl=2)

    def forward(self,corrd, corr, flow2d,depthc):

        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corro,flo], dim=1)
        cor_dpz = torch.cat([corrdo,dzo], dim=1)

        out_flow = self.conv_flo(cor_flo)
        out_dpz = self.conv_dpz(cor_dpz)
        return torch.cat([out_flow,out_dpz, flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_wod16v7(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wod16v7, self).__init__()
        cor_planes = 338#343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(4, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 32, 3, padding=1)


        self.convz1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_flo = ConvNeXtV2tiny(128+32, 128-4, kszie=7, bl=2)
        self.conv_dpz = ConvNeXtV2tiny(96+32, 64-2, kszie=7, bl=2)

    def forward(self,corrd, corr, flow2d,depthc):

        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corro,flo], dim=1)
        cor_dpz = torch.cat([corrdo,dzo], dim=1)

        out_flow = self.conv_flo(cor_flo)
        out_dpz = self.conv_dpz(cor_dpz)
        return torch.cat([out_flow,out_dpz,flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_wod16v6(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wod16v6, self).__init__()
        cor_planes = 338#343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convaggf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convaggf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_flo = ConvNeXtV2tiny(128+32+32, 128-4, kszie=7, bl=2)
        self.conv_dpz = ConvNeXtV2tiny(96+32, 64-1, kszie=7, bl=2)

    def forward(self,corrd, corr, aggflow2d, flow2d,depthc):

        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        floa = F.relu(self.convaggf1(aggflow2d))
        floa = F.relu(self.convaggf2(floa))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corro,flo,floa], dim=1)
        cor_dpz = torch.cat([corrdo,dzo], dim=1)

        out_flow = self.conv_flo(cor_flo)
        out_dpz = self.conv_dpz(cor_dpz)
        return torch.cat([out_flow,out_dpz,aggflow2d,flow2d,depthc], dim=1)
class MotioneEncoderRF3Duvd_wd(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wd, self).__init__()
        cor_planes = 162  # 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 3, padding=1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 3, padding=1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = ConvNeXtV2tiny(128+96 + 64 + 32, 192 - 3, kszie=7, bl=2)

    def forward(self, corrd, corr, flow2d, depthc):
        cord = self.xcitc1(F.relu(self.convc1(corr)))
        corr = self.xcitc2(F.relu(self.convc2(corrd)))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corr, cord, flo, dzo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, flow2d, depthc], dim=1)


class MotioneEncoderRF3Duvd_wdv1(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wdv1, self).__init__()
        cor_planes  = ((args.itr2 * 2 + 1) ** 2) * 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = ConvNeXtV2tiny(128+96 + 64 + 32, 192 - 3, kszie=7, bl=2)

    def forward(self, corrd, corr, flow2d, depthc):
        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))
        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))
        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))
        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))
        cor_flo = torch.cat([corro, corrdo, flo, dzo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow2d, depthc], dim=1)

#这一版运动编码，强化光流估计能力
class MotioneEncoderRF3D_itr2616(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3D_itr2616, self).__init__()
        cor_planes = ((args.itr2 * 2 + 1) ** 2) * 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)
        self.convdf = ConvNeXtV2tiny(128 + 96 + 64 + 32, 128 - 3, kszie=7, bl=2)
        self.convf = ConvNeXtV2tiny(128 + 64, 64, kszie=7, bl=2)

    def forward(self, corrd, corr, flow2d, depthc):
        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))
        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))
        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))
        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))
        cor_flod = torch.cat([corro, corrdo, flo, dzo], dim=1)
        cor_flo = torch.cat([corro, flo], dim=1)
        outdf = self.convdf(cor_flod)
        outf = self.convf(cor_flo)

        return torch.cat([outdf,outf, flow2d, depthc], dim=1)
#这一版运动编码，强化光流估计能力
class MotioneEncoderRF3D_itr1616(nn.Module):
    def __init__(self, args):
        super(MotioneEncoderRF3D_itr1616, self).__init__()
        cor_planes = ((args.itr1 * 2 + 1) ** 2) * 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)
        self.convdf = ConvNeXtV2tiny(128 + 96 + 64 + 32, 128 - 3, kszie=7, bl=2)
        self.convf = ConvNeXtV2tiny(128 + 64, 64, kszie=7, bl=2)

    def forward(self, corrd, corr, flow2d, depthc):
        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))
        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))
        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))
        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))
        cor_flod = torch.cat([corro, corrdo, flo, dzo], dim=1)
        cor_flo = torch.cat([corro, flo], dim=1)
        outdf = self.convdf(cor_flod)
        outf = self.convf(cor_flo)

        return torch.cat([outdf,outf, flow2d, depthc], dim=1)

class MotioneEncoderRF3Duvd_wdv3(nn.Module):#todo这个版本把深度编码单独摘出来
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wdv3, self).__init__()
        cor_planes  = ((args.itr2 * 2 + 1) ** 2) * 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_flo = ConvNeXtV2tiny(128 + 32, 128 - 2, kszie=7, bl=2)
        self.conv_dpz = ConvNeXtV2tiny(96 + 32, 64 - 1, kszie=7, bl=2)

    def forward(self, corrd, corr, flow2d, depthc):
        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corro, flo], dim=1)
        cor_dpz = torch.cat([corrdo, dzo], dim=1)

        out_flow = self.conv_flo(cor_flo)
        out_dpz = self.conv_dpz(cor_dpz)
        return torch.cat([out_flow, out_dpz, flow2d, depthc], dim=1)
class MotioneEncoderRF3Duvd_wdv6(nn.Module):#todo这个版本把深度编码单独摘出来
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wdv6, self).__init__()
        cor_planes = 162  # 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convaf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convaf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convz1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_flo = ConvNeXtV2tiny(128 + 32 +32, 128 - 4, kszie=7, bl=2)
        self.conv_dpz = ConvNeXtV2tiny(96 + 32, 64 - 1, kszie=7, bl=2)

    def forward(self, corrd, corr, aflow2d,flow2d, depthc):
        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        floa = F.relu(self.convaf1(aflow2d))
        floa = F.relu(self.convaf2(floa))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corro, flo , floa], dim=1)
        cor_dpz = torch.cat([corrdo, dzo], dim=1)

        out_flow = self.conv_flo(cor_flo)
        out_dpz = self.conv_dpz(cor_dpz)
        return torch.cat([out_flow, out_dpz, aflow2d, flow2d, depthc], dim=1)
class MotioneEncoderRF3Duvd_wdv7(nn.Module):#todo这个版本把深度编码单独摘出来
    def __init__(self, args):
        super(MotioneEncoderRF3Duvd_wdv7, self).__init__()
        cor_planes = 162  # 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        self.convcd1 = nn.Conv2d(cor_planes, 256, 1)
        self.convcd2 = nn.Conv2d(256, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(4, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convz1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convz2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_flo = ConvNeXtV2tiny(128 + 32, 128 - 4, kszie=7, bl=2)
        self.conv_dpz = ConvNeXtV2tiny(96 + 32, 64 - 2, kszie=7, bl=2)

    def forward(self, corrd, corr,flow2d, depthc):
        corro = F.relu(self.convc1(corr))
        corro = F.relu(self.convc2(corro))

        flo = F.relu(self.convf1(flow2d))
        flo = F.relu(self.convf2(flo))

        corrdo = F.relu(self.convcd1(corrd))
        corrdo = F.relu(self.convcd2(corrdo))

        dzo = F.relu(self.convz1(depthc))
        dzo = F.relu(self.convz2(dzo))

        cor_flo = torch.cat([corro, flo], dim=1)
        cor_dpz = torch.cat([corrdo, dzo], dim=1)

        out_flow = self.conv_flo(cor_flo)
        out_dpz = self.conv_dpz(cor_dpz)
        return torch.cat([out_flow, out_dpz, flow2d, depthc], dim=1)
from core.utils.submodule import  Tiny_Unetlikev7,Tiny_Unetlikev7t

class dzhead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(dzhead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class RF3DuvdUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corrdf,corr3d,corr, flow2d,depthc):
        motion_features = self.encoder(corrdf,corr3d,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlock_16(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_16d(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp,corr, flow2d,depthc):
        motion_features = self.encoder(corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz
class RF3DuvdUpdateBlock_16d(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16d, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp,corrd,corr, flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz
class RF3DuvdUpdateBlock_16dv1(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv1, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v1(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp,corrd,corr, flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlock_16dv0(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv0, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v1(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=192 +  2*hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp,corrd,corr, flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz
class RF3DuvdUpdateBlock_16dv2(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv2, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v1(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)


    def forward(self, net, inp,corrd,corr, flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz
class RF3DuvdUpdateBlock_16dv3(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv3, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v3(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)


    def forward(self, net, inp,corrd,corr, flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz

class RF3DuvdUpdateBlock_16dv3ab(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv3ab, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v3(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp,corrd,corr, flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)
        mask = .25 * self.mask(net)
        return net,mask, delta_flow,delta_dz

class RF3DuvdUpdateBlock_16dv6(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv6, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v6(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)


    def forward(self, net, inp,corrd,corr, aggflow2d,flow2d,depthc):
        motion_features = self.encoder(corrd,corr,aggflow2d, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz
class RF3DuvdUpdateBlock_16dv7(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock_16dv7, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod16v7(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)


    def forward(self, net, inp,corrd,corr,flow2d,depthc):
        motion_features = self.encoder(corrd,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz
class RF3DuvdUpdateBlockwodepth(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwodepth, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow2d,depthc):
        motion_features = self.encoder(corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlockwodepth84(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwodepth84, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wod(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow2d,depthc):
        motion_features = self.encoder(corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlockwd(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwd, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wd(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz



class RF3DuvdUpdateBlockwdv0(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv0, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv1(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=192 + 2*hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlockwdv1(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv1, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv1(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlock16_itr2(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock16_itr2, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3D_itr2616(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz

class RF3DuvdUpdateBlock16_itr1(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlock16_itr1, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3D_itr1616(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow,delta_dz




class RF3DuvdUpdateBlockwdv2(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv2, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv1(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz

class RF3DuvdUpdateBlockwdv3(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv3, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv3(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz
class RF3DuvdUpdateBlockwdv3ab(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv3ab, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv3(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)
        #上采样可以新加一个任务，就是把真实深度准确上采样
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 9, 1, padding=0))
    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)
        mask = .25 * self.mask(net)
        return net,mask, delta_flow,delta_dz

class RF3DuvdUpdateBlockwdv6(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv6, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv6(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, coord,corr,aggflow2d, flow2d,depthc):
        motion_features = self.encoder(coord,corr,aggflow2d, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz

class RF3DuvdUpdateBlockwdv7(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(RF3DuvdUpdateBlockwdv7, self).__init__()
        self.args = args
        self.encoder = MotioneEncoderRF3Duvd_wdv7(args)
        self.TinyU = Tiny_Unetlikev7(192 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.dz_head = dzhead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, coord,corr, flow2d,depthc):
        motion_features = self.encoder(coord,corr, flow2d,depthc)

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.TinyU(torch.cat([net, inp], dim=1))

        delta_flow = self.flow_head(net)
        delta_dz = self.dz_head(net)

        return net, delta_flow,delta_dz