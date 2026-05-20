import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.submodule import  bfmodule,Final_UnetV1
from core.utils.gma import Aggregate
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class FlowHeadconf(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHeadconf, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 6, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class expHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(expHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv2(self.relu(self.conv1(x))))

class expHeadv2conf(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(expHeadv2conf, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class expHeadv2(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(expHeadv2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class occHead(nn.Module):#这个放在普通的层里面
    def __init__(self, input_dim=128, hidden_dim=256):
        super(occHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv2(self.relu(self.conv1(x))))
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
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


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
class ConvGRUd(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, dilation=4):
        super(ConvGRUd, self).__init__()
        self.hidden_dim = hidden_dim
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz1(hx) + self.convz2(hx))
        r = torch.sigmoid(self.convr1(hx) + self.convr2(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)) + self.convq2(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
from utils.grad_mask import GradMask
class BasicMotioneEncoder_occ_one(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder_occ_one, self).__init__()
        cor_planes = 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192+64, 128-3, 3, padding=1)
        self.gradmask = GradMask.apply
    def forward(self, flow, corr, exp,occ=None):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        if occ!=None:
            cor = self.gradmask(cor, occ)
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo,expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
class BasicMotioneEncoder_exp_pre(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder_exp_pre, self).__init__()
        cor_planes = 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 32, 3, padding=1)

        self.convexp1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convexp2 = nn.Conv2d(128, 32, 3, padding=1)

        self.conv = nn.Conv2d(64+192+32+32, 128-4, 3, padding=1)
    def forward(self, flow, corr, exp,exp2):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        expo2 = F.relu(self.convexp1(exp2))
        expo2 = F.relu(self.convexp2(expo2))

        cor_flo = torch.cat([cor, flo, expo, expo2], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp, exp2], dim=1)
#这个放到最终的细化层中
class occ_pre(nn.Module):
    def __init__(self):
        super(occ_pre, self).__init__()
        cor_planes = 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)

        self.conv1l1 = nn.Conv2d(49, 64, 7, padding=3)
        self.conv1l2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv2l1 = nn.Conv2d(49, 64, 7, padding=3)
        self.conv2l2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv3l1 = nn.Conv2d(49, 64, 7, padding=3)
        self.conv3l2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv4l1 = nn.Conv2d(49, 64, 7, padding=3)
        self.conv4l2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv5l1 = nn.Conv2d(49, 64, 7, padding=3)
        self.conv5l2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = nn.Conv2d(49*5+192+192, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, corr,corrliner5,net):
        corr = corr.detach()
        corrliner5 = corrliner5.detach()

        cor1, cor2 ,cor3, cor4 ,cor5 = torch.split(corrliner5, [49, 49,49, 49,49], dim=1)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        cor1 = F.relu(self.convc1(cor1))
        cor1 = F.relu(self.convc2(cor1))

        cor2 = F.relu(self.convc1(cor2))
        cor2 = F.relu(self.convc2(cor2))

        cor3 = F.relu(self.convc1(cor3))
        cor3 = F.relu(self.convc2(cor3))

        cor4 = F.relu(self.convc1(cor4))
        cor4 = F.relu(self.convc2(cor4))

        cor5 = F.relu(self.convc1(cor5))
        cor5 = F.relu(self.convc2(cor5))

        cor_flo = torch.cat([cor,cor1, cor2 ,cor3, cor4 ,cor5,net], dim=1)
        out = F.relu(self.conv(cor_flo))
        out = self.sigmoid(self.conv2(out))


        return out
class BasicMotioneEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder, self).__init__()
        cor_planes = 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192+64, 128-3, 3, padding=1)

    def forward(self, flow, corr, exp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo, expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
from core.layer import ConvNeXtV2tiny
'''
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = ConvNeXtV2tiny(64+192+64, 128-3, kszie=7, bl=2)
        
        
        self.convc1 = nn.Conv2d(cor_planes, 192, 7, padding=3)
        self.convc2 = nn.Conv2d(192, 128, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = ConvNeXtV2tiny(64+128+64, 128-3, kszie=5, bl=2)
'''
class BasicMotioneEncoderNextv2(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoderNextv2, self).__init__()
        cor_planes = 343#343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = ConvNeXtV2tiny(64+192+64, 128-3, kszie=7, bl=2)
    def forward(self, flow, corr, exp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo, expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
class BasicMotioneEncoderResRAFT(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoderResRAFT, self).__init__()
        cor_planes = 324#343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = ConvNeXtV2tiny(192+64, 128-2, kszie=7, bl=2)
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
class BasicMotioneEncoderNextv6(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoderNextv6, self).__init__()
        self.convc1 = nn.Conv2d(196, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)

        self.convcs1 = nn.Conv2d(147, 128, 1, padding=0)
        self.convcs2 = nn.Conv2d(128, 96, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 16, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conve2 = nn.Conv2d(32, 16, 3, padding=1)
        #self.conv = ConvNeXtV2tiny(16+192+16, 128-3, kszie=7, bl=2)
        self.conv = nn.Conv2d(16+192+16, 128 - 3, 3, padding=1)
    def forward(self, flow, corr,corrScale, exp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        corS = F.relu(self.convcs1(corrScale))
        corS = F.relu(self.convcs2(corS))

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, corS, flo, expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
class BasicMotioneEncoderOCC(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoderOCC, self).__init__()
        cor_planes = 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convl1 = nn.Conv2d(21, 128, 7, padding=3)
        self.convl2 = nn.Conv2d(128, 64, 3, padding=1)


        self.conv = nn.Conv2d(64+192+64+64, 128-3-1, 3, padding=1)

    def forward(self, flow, corr, exp,pl,pe):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))
        plo = F.relu(self.convl1(pl))
        plo = F.relu(self.convl2(plo))

        cor_flo = torch.cat([cor, flo,expo,plo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp,pe], dim=1)
class CSCV343OCCUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(CSCV343OCCUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoderOCC(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        # self.gru = ConvGRUd(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp, pl,pe, upsample=True):
        motion_features = self.encoder(flow, corr, exp,pl,pe)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class UpdateBlock_occ_linear(nn.Module):
    def __init__(self, args,hidim):
        super(UpdateBlock_occ_linear, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoderOCC(args)

        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(128+hidim*2, 1)
    def forward(self, net, inp, corr, flow,exp,pl,pe):
        motion_features = self.encoder(flow, corr, exp,pl,pe)
        inp = torch.cat([inp, motion_features,net], dim=1)

        exp_flow = self.exp_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,exp_flow[0]
import time
class ScaleflowUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow

from layer import ConvNeXtV2Block
class ResFlowUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ResFlowUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        self.RNN1 = ConvNeXtV2Block(128 + 2*hidden_dim,hidden_dim)
        self.RNN2 = ConvNeXtV2Block(128 + 2*hidden_dim,hidden_dim)

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128

        net = self.RNN1(torch.cat([net, inp],dim=1))
        net = self.RNN2(torch.cat([net, inp],dim=1))

        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow

from core.utils.submodule import  Tiny_Unetlike,Tiny_Unetlikev7,Tiny_Unetlikev3,Tiny_Unetlikev6,Tiny_Unetlikev8,Tiny_Unetlikev9

class ResFlowUpdateBlockUnet(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ResFlowUpdateBlockUnet, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        #self.TinyU = Tiny_Unetlikev7(128 + 2 * hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.TinyU=Tiny_Unetlikev9(hidden_dim=hidden_dim, input_dim=128 + 128)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadv2(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp,cx16=None):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128
        net = self.TinyU(net, inp)
        #net = self.TinyU(torch.cat([net, inp], dim=1))
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow

class ResFlowUpdateBlockUnetL(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ResFlowUpdateBlockUnetL, self).__init__()
        self.args = args
        #self.encoder = BasicMotioneEncoder(args)
        self.encoder = BasicMotioneEncoderNextv2(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        #self.TinyU=Tiny_Unetlikev9(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadv2(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp,cx16=None):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128
        #net = self.TinyU(net, inp)
        net = self.TinyU(torch.cat([net, inp], dim=1))
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ResFlowUpdateBlockUnetLConf(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ResFlowUpdateBlockUnetLConf, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoderNextv2(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHeadconf(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadv2conf(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128
        net = self.TinyU(torch.cat([net, inp], dim=1))
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ResRAFTUpdateBlockUnetL(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ResRAFTUpdateBlockUnetL, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoderResRAFT(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow,cx16=None):
        motion_features = self.encoder(flow, corr)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128
        net = self.TinyU(torch.cat([net, inp], dim=1))
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
from layer import ConvNextBlock
class ResFlowUpdateBlockUnet2RNN(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ResFlowUpdateBlockUnet2RNN, self).__init__()
        self.args = args
        #self.encoder = BasicMotioneEncoder(args)
        self.encoder = BasicMotioneEncoderNextv2(args)
        self.TinyU = nn.Sequential(
            # convex combination of 3x3 patches
            ConvNextBlock(128 +2*hidden_dim, hidden_dim),
            ConvNextBlock(hidden_dim, hidden_dim)
        )
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadv2(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp,cx16=None):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128
        #net = self.TinyU(net, inp)
        net = self.TinyU(torch.cat([net, inp], dim=1))
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class BlockUnetCorrScale(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(BlockUnetCorrScale, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoderNextv6(args)
        self.TinyU = Tiny_Unetlikev7(128 +2*hidden_dim, hidden_dim, mid_channle=hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadv2(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, corrScale, flow, exp,cx16=None):
        motion_features = self.encoder(flow, corr,corrScale, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+128
        net = self.TinyU(torch.cat([net, inp], dim=1))
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ScaleflowUpdateBlockocc_one(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlockocc_one, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_occ_one(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp,occ=None, upsample=True):
        motion_features = self.encoder(flow, corr, exp,occ)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ScaleflowUpdateBlockocc_pre(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlockocc_pre, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_occ_one(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp,corrlinear=None,occ=None, upsample=True):
        motion_features = self.encoder(flow, corr, exp,occ)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ScaleflowUpdateBlockocc_pre(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlockocc_pre, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_occ_one(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp,corrlinear=None,occ=None, upsample=True):
        motion_features = self.encoder(flow, corr, exp,occ)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ScaleflowUpdateBlockpreocc(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlockpreocc, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        self.occ_head = occHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.maskocc = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192
        net = self.gru(net, inp)
        occ_out = self.occ_head(net)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        maskocc = .25 * self.maskocc(net)
        return net, mask,masks,maskocc, delta_flow, exp_flow,occ_out


class DCUpdateBlock(nn.Module):
    def __init__(self, args,hidim):
        super(DCUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(128+hidim*2, 1)
    def forward(self, net, inp, corr, flow,exp):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)
        exp_update = self.exp_head(inp)[0]
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())

        return mask ,exp_update

class DCFlowUpdateBlock(nn.Module):
    def __init__(self, args,hidim):
        super(DCFlowUpdateBlock, self).__init__()
        self.args = args
        #self.encoder = BasicMotioneEncoder(args)
        self.encoder = BasicMotioneEncoderNextv2(args)
        self.maske = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = Tiny_Unetlikev7(128+2*hidim, 1,mid_channle=192)
    def forward(self, net, inp, corr, flow,exp,cx16):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)
        exp_update = self.exp_head(inp)
        # scale mask to balence gradients
        maske = .25 * self.maske(net.float())
        return maske ,exp_update
class DCFlowUpdateBlockV2(nn.Module):
    def __init__(self, args,hidim):
        super(DCFlowUpdateBlockV2, self).__init__()
        self.args = args
        #self.encoder = BasicMotioneEncoder(args)
        self.encoder = BasicMotioneEncoderNextv6(args)
        self.maske = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = Tiny_Unetlikev7(128+2*hidim, 1,mid_channle=192)
    def forward(self, net, inp, corr,coorS, flow,exp,cx16):
        motion_features = self.encoder(flow, corr,coorS, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)
        exp_update = self.exp_head(inp)
        # scale mask to balence gradients
        maske = .25 * self.maske(net.float())
        return maske ,exp_update