import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import ConvNextBlock
from core.utils.submodule import  bfmodule

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=4):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, dim=128):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(471, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim , 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conve1 = nn.Conv2d(1, dim, 7, padding=3)
        self.conve2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 3, 3, padding=1)

    def forward(self, flow,exp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))
        cor_flo = torch.cat([cor, flo, expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hdim=192, cdim=192):
        # net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(2):
            self.refine.append(ConvNextBlock(2 * cdim + hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow,exp, upsample=True):
        motion_features = self.encoder(flow, exp,corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net
#这个的基本思路是先用一次超大的采样率进行一次具有全局视野的初始化更新flow和exp，后续再慢慢迭代
class BasicInitBlock(nn.Module):
    def __init__(self, args, hdim=192, cdim=192):
        # net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(2):
            self.refine.append(ConvNextBlock(2 * cdim + hdim, hdim))
        self.refine = nn.ModuleList(self.refine)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.upsample_weightdc = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 2, 3, padding=1)
        )
        self.exp_head = nn.Sequential(
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 1, 3, padding=1)
        )
    def forward(self, net, inp, corr, flow,exp, upsample=True):
        motion_features = self.encoder(flow, exp,corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        flowd = self.flow_head(net)
        expd = self.exp_head(net)
        upmask = self.upsample_weight(net)
        upmaskdc = self.upsample_weightdc(net)
        return net,flowd,upmask,expd,upmaskdc
class FinalUpdateBlock(nn.Module):
    def __init__(self, args, hdim=192, cdim=192):
        # net: hdim, inp: cdim
        super(FinalUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        self.refine=ConvNextBlock(2 * cdim + hdim, hdim)
        self.exphead = ConvNextBlock(2 * cdim + hdim, 1)

    def forward(self, net, inp, corr, flow,exp, upsample=True):
        motion_features = self.encoder(flow, exp,corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.refine(torch.cat([net, inp], dim=1))
        exp_delta = self.exphead(net)
        return exp_delta
#单纯更新一次
class DCUpdateBlock(nn.Module):
    def __init__(self, args):
        super(DCUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args,dim=args.dim)
        self.mask = nn.Sequential(
            nn.Conv2d(args.dim*3, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.bfhead = bfmodule(args.dim*3, 1)

    def forward(self, net, inp, corr, flow,exp):
        motion_features = self.encoder(flow, exp,corr)
        inp = torch.cat([inp, motion_features,net], dim=1)
        expd = self.bfhead(inp)[0]
        maskup = self.mask(inp)
        return maskup,expd