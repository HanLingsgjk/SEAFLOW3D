import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_fn, num_channels):
    if norm_fn == 'group':
        groups = 8 if num_channels % 8 == 0 else 1
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    elif norm_fn == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif norm_fn == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif norm_fn == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_fn: {norm_fn}")


class DWResidualBlock(nn.Module):
    """
    Depthwise-separable residual block:
    DW(kxk) -> PW(1x1) -> Norm -> ReLU -> DW(kxk) -> PW(1x1) -> Norm + skip
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2

        self.dw1 = nn.Conv2d(
            in_planes, in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=False
        )
        self.pw1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = _make_norm(norm_fn, planes)
        self.relu = nn.ReLU(inplace=True)

        self.dw2 = nn.Conv2d(
            planes, planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=planes,
            bias=False
        )
        self.pw2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.norm2 = _make_norm(norm_fn, planes)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_fn, planes)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.dw1(x)
        out = self.pw1(out)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.dw2(out)
        out = self.pw2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.relu(out)
        return out




class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        if x.shape[1] != y.shape[1]:  # for the uplayers.
            return y

        return self.relu(x+y)
class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group'):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, output_dim, kernel_size=1)

        self.in_planes = 256 + 128
        self.up_layer2 = self._make_layer(128, stride=1)
        self.in_planes = 128 + 96
        self.up_layer1 = self._make_layer(96, stride=1)
        self.in_planes = 96 + 64
        self.up_layer0 = self._make_layer(64, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.up_layer2(up2layer_input)

        # uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.up_layer1(up1layer_input)

        # uplayer0:
        cur_h, cur_w = list(enc_out1.size())[-2:]
        up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
        up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
        up0_out = self.up_layer0(up0layer_input)

        enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
        up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
        up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
        up0_out = torch.split(up0_out, [batch_dim, batch_dim], dim=0)

        return [enc_out4, up2_out, up1_out, up0_out]
class Basic_Context_Encoder(nn.Module):
    def __init__(self, output_dim=256,input_dim=3, norm_fn='group'):
        super(Basic_Context_Encoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.layer4 = self._make_layer(160, stride=2)           # 160 for group norm instead of 164
        self.conv2 = nn.Conv2d(160, output_dim, kernel_size=1)  # 160 for group norm instead of 164
        self.in_planes = 256 + 128
        self.up_layer2 = self._make_layer(output_dim, stride=1)
        self.in_planes = 256 + 96
        self.up_layer1 = self._make_layer(output_dim, stride=1)
        self.in_planes = 256 + 64
        self.up_layer0 = self._make_layer(output_dim, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def _make_modest_layer(self, dim, intermediate_channels, stride=1):
        layer1 = ResidualBlock(self.in_planes, intermediate_channels, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(intermediate_channels, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.up_layer2(up2layer_input)

        # uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.up_layer1(up1layer_input)

        # uplayer 0:
        cur_h, cur_w = list(enc_out1.size())[-2:]
        up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
        up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
        up0_out = self.up_layer0(up0layer_input)

        return [enc_out4, up2_out, up1_out, up0_out]
class BasicEncoder_resconv(nn.Module):
    def __init__(self,input_dim=3, output_dim=128, norm_fn='group', model_type="CCMR+"):
        super(BasicEncoder_resconv, self).__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Top-down feature extraction

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, 160, kernel_size=1)

        # Bottom-up feature consolidation

        self.in_planes = 160 + 128
        self.up_layer2 = self._make_layer(128, stride=1)
        self.after_up_layer2_conv = nn.Conv2d(128, 128, kernel_size=1)
        self.in_planes = 128 + 96
        self.up_layer1 = self._make_layer(96, stride=1)
        self.after_up_layer1_conv = nn.Conv2d(96, 96, kernel_size=1)

        if self.model_type.casefold() == "ccmr+":
            self.in_planes = 96 + 64
            self.up_layer0 = self._make_layer(64, stride=1)
            self.after_up_layer0_conv = nn.Conv2d(64, 64, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))

        # uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))

        if self.model_type.casefold() == "ccmr+":
            # uplayer0:
            cur_h, cur_w = list(enc_out1.size())[-2:]
            up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
            up0_out = torch.split(up0_out, [batch_dim, batch_dim], dim=0)

            return [enc_out4, up2_out, up1_out, up0_out]
        elif self.model_type.casefold() == "ccmr":
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)

            return [enc_out4, up2_out, up1_out]
class Basic_Context_Encoder_resconv(nn.Module):
    def __init__(self,input_dim=3, output_dim=128, norm_fn='group', model_type="CCMR+"):
        super(Basic_Context_Encoder_resconv, self).__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Top-down feature extraction

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, output_dim, kernel_size=1)

        # Bottom-up feature consolidation
        self.in_planes = 256 + 128
        self.up_layer2 = self._make_layer(128, stride=1)
        self.after_up_layer2_conv = nn.Conv2d(128, output_dim, kernel_size=1)
        self.in_planes = 256 + 96
        self.up_layer1 = self._make_layer(96, stride=1)
        self.after_up_layer1_conv = nn.Conv2d(96, output_dim, kernel_size=1)
        if self.model_type.casefold() == "ccmr+":
            self.in_planes = 256 + 64
            self.up_layer0 = self._make_layer(64, stride=1)
            self.after_up_layer0_conv = nn.Conv2d(64, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))
        # uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))
        if self.model_type.casefold() == "ccmr+":
            # uplayer0:
            cur_h, cur_w = list(enc_out1.size())[-2:]
            up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
            return [enc_out4, up2_out, up1_out, up0_out]

        elif self.model_type.casefold() == "ccmr":
            return [enc_out4, up2_out, up1_out]
class Basic_Context_Encoder_resconvb(nn.Module):
    def __init__(self,input_dim=3, output_dim=256, norm_fn='group', model_type="CCMR+"):
        super(Basic_Context_Encoder_resconvb, self).__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Top-down feature extraction

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, output_dim, kernel_size=1)

        # Bottom-up feature consolidation
        self.in_planes = output_dim + 128
        self.up_layer2 = self._make_layer(128, stride=1)
        self.after_up_layer2_conv = nn.Conv2d(128, output_dim, kernel_size=1)
        self.in_planes = output_dim + 96
        self.up_layer1 = self._make_layer(96, stride=1)
        self.after_up_layer1_conv = nn.Conv2d(96, output_dim, kernel_size=1)
        if self.model_type.casefold() == "ccmr+":
            self.in_planes = output_dim + 64
            self.up_layer0 = self._make_layer(64, stride=1)
            self.after_up_layer0_conv = nn.Conv2d(64, 128, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))
        # uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))
        if self.model_type.casefold() == "ccmr+":
            # uplayer0:
            cur_h, cur_w = list(enc_out1.size())[-2:]
            up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
            return [enc_out4, up2_out, up1_out, up0_out]

        elif self.model_type.casefold() == "ccmr":
            return [enc_out4, up2_out, up1_out]
def _dw_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) // 2) * dilation


class HybridDilatedDWResidualBlock(nn.Module):
    """
    轻量 Hybrid block:
    pre 1x1 -> [DW local, DW dilated] -> concat -> fuse 1x1 -> residual
    """
    def __init__(
        self,
        in_planes,
        planes,
        norm_fn='group',
        stride=1,
        kernel_size=5,
        dilations=(1, 3),
    ):
        super().__init__()
        d1, d2 = dilations

        self.pre = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            _make_norm(norm_fn, planes),
            nn.ReLU(inplace=True),
        )

        self.branch_local = nn.Sequential(
            nn.Conv2d(
                planes, planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=_dw_padding(kernel_size, d1),
                dilation=d1,
                groups=planes,
                bias=False,
            ),
            _make_norm(norm_fn, planes),
            nn.ReLU(inplace=True),
        )

        self.branch_dil = nn.Sequential(
            nn.Conv2d(
                planes, planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=_dw_padding(kernel_size, d2),
                dilation=d2,
                groups=planes,
                bias=False,
            ),
            _make_norm(norm_fn, planes),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
            _make_norm(norm_fn, planes),
        )

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_fn, planes),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        x = self.pre(x)
        x1 = self.branch_local(x)
        x2 = self.branch_dil(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)

        x = x + identity
        x = self.relu(x)
        return x


class DWBasicEncoder_resconv_mixed(nn.Module):
    """
    你要的折中版：
    - stem: 只保留第一层全分辨率聚合，第二个 stem 分支挪到下采样后
    - layer1 / layer2: Hybrid Dilated DW
    - layer3 / layer4 / up: 原 DWResidualBlock
    """
    def __init__(
        self,
        input_dim=3,
        output_dim=128,   # 保留接口
        norm_fn='group',
        model_type="CCMR+",
        stem_kernel1=7,
        stem_kernel2=5,
        hybrid_kernels=(5, 5),
        hybrid_dilations=((1, 2), (1, 3)),
        stage_kernels=(3, 3),   # layer3, layer4
        up_kernel=3,
        base_dims=(48, 80, 112, 144),
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type

        c1, c2, c3, c4 = base_dims

        # 输出维度保持和原版一致
        out4_dim = 160
        up2_dim = 128
        up1_dim = 96
        up0_dim = 64

        self.out4_dim = out4_dim
        self.up2_dim = up2_dim
        self.up1_dim = up1_dim
        self.up0_dim = up0_dim

        # -------------------------
        # Stem part 1: 全分辨率聚合，不下采样
        # -------------------------
        self.stem_full = nn.Sequential(
            nn.Conv2d(
                input_dim, c1,
                kernel_size=stem_kernel1,
                stride=1,
                padding=stem_kernel1 // 2,
                bias=False
            ),
            _make_norm(self.norm_fn, c1),
            nn.ReLU(inplace=True),
        )
        # 下采样后移
        self.stem_down = nn.AvgPool2d(kernel_size=2, stride=2)

        # -------------------------
        # Stem part 2: 挪到下采样后
        # -------------------------
        self.stem_post = nn.Sequential(
            nn.Conv2d(
                c1, c1,
                kernel_size=stem_kernel2,
                stride=1,
                padding=_dw_padding(stem_kernel2, 2),
                dilation=2,
                groups=c1,
                bias=False
            ),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            _make_norm(norm_fn, c1),
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Encoder
        # layer1/layer2: Hybrid
        # layer3/layer4: 原 DWResidualBlock
        # -------------------------
        self.in_planes = c1

        self.layer1 = self._make_hybrid_layer(
            c1,
            stride=1,
            kernel_size=hybrid_kernels[0],
            dilations=hybrid_dilations[0]
        )
        self.layer2 = self._make_hybrid_layer(
            c2,
            stride=2,
            kernel_size=hybrid_kernels[1],
            dilations=hybrid_dilations[1]
        )

        self.layer3 = self._make_dw_layer(
            c3,
            stride=2,
            kernel_size=stage_kernels[0]
        )
        self.layer4 = self._make_dw_layer(
            c4,
            stride=2,
            kernel_size=stage_kernels[1]
        )

        self.conv2 = nn.Conv2d(c4, out4_dim, kernel_size=1, bias=False)

        # -------------------------
        # Bottom-up feature consolidation
        # up 全部用原 DWResidualBlock
        # -------------------------
        self.in_planes = out4_dim + c3
        self.up_layer2 = self._make_dw_layer(up2_dim, stride=1, kernel_size=up_kernel)
        self.after_up_layer2_conv = nn.Conv2d(up2_dim, up2_dim, kernel_size=1, bias=False)

        self.in_planes = up2_dim + c2
        self.up_layer1 = self._make_dw_layer(up1_dim, stride=1, kernel_size=up_kernel)
        self.after_up_layer1_conv = nn.Conv2d(up1_dim, up1_dim, kernel_size=1, bias=False)

        if self.model_type.casefold() == "ccmr+":
            self.in_planes = up1_dim + c1
            self.up_layer0 = self._make_dw_layer(up0_dim, stride=1, kernel_size=up_kernel)
            self.after_up_layer0_conv = nn.Conv2d(up0_dim, up0_dim, kernel_size=1, bias=False)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_hybrid_layer(self, dim, stride=1, kernel_size=5, dilations=(1, 3)):
        layer1 = HybridDilatedDWResidualBlock(
            self.in_planes, dim,
            norm_fn=self.norm_fn,
            stride=stride,
            kernel_size=kernel_size,
            dilations=dilations
        )
        layer2 = HybridDilatedDWResidualBlock(
            dim, dim,
            norm_fn=self.norm_fn,
            stride=1,
            kernel_size=kernel_size,
            dilations=dilations
        )
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def _make_dw_layer(self, dim, stride=1, kernel_size=3):
        layer1 = DWResidualBlock(
            self.in_planes, dim,
            norm_fn=self.norm_fn,
            stride=stride,
            kernel_size=kernel_size
        )
        layer2 = DWResidualBlock(
            dim, dim,
            norm_fn=self.norm_fn,
            stride=1,
            kernel_size=kernel_size
        )
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        batch_dim = None
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # stem
        x = self.stem_full(x)   # full-res aggregation
        x = self.stem_down(x)   # downsample later
        x = self.stem_post(x)   # second stem branch after downsample

        # encoder
        x = self.layer1(x)
        enc_out1 = x

        x = self.layer2(x)
        enc_out2 = x

        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # up layer2
        enc_out4_resized = F.interpolate(
            enc_out4, size=enc_out3.shape[-2:],
            mode='bilinear', align_corners=False
        )
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))

        # up layer1
        up2_out_resized = F.interpolate(
            up2_out, size=enc_out2.shape[-2:],
            mode='bilinear', align_corners=False
        )
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))

        if self.model_type.casefold() == "ccmr+":
            # up layer0
            up1_out_resized = F.interpolate(
                up1_out, size=enc_out1.shape[-2:],
                mode='bilinear', align_corners=False
            )
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))

            if is_list:
                enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
                up2_out  = torch.split(up2_out,  [batch_dim, batch_dim], dim=0)
                up1_out  = torch.split(up1_out,  [batch_dim, batch_dim], dim=0)
                up0_out  = torch.split(up0_out,  [batch_dim, batch_dim], dim=0)

            return [enc_out4, up2_out, up1_out, up0_out]

        elif self.model_type.casefold() == "ccmr":
            if is_list:
                enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
                up2_out  = torch.split(up2_out,  [batch_dim, batch_dim], dim=0)
                up1_out  = torch.split(up1_out,  [batch_dim, batch_dim], dim=0)

            return [enc_out4, up2_out, up1_out]

class DepthGeomAug(nn.Module):
    """
    输入:  Bx1xHxW
           假设输入已经大致在 [-1, 1]
    输出:  Bx6xHxW
           各分支尽量保持在 [-1, 1] 左右，但不做硬边界截断
    """
    def __init__(self, blur_kernel=5, log_scale=10.0, resid_gain=2.0, use_softsign=False):
        super().__init__()
        self.blur_kernel = blur_kernel
        self.log_scale = log_scale
        self.resid_gain = resid_gain
        self.use_softsign = use_softsign

    def _soft_range(self, x):
        # 不做硬截断，只做轻微软压缩
        if self.use_softsign:
            return x / (1.0 + x.abs())
        else:
            return torch.tanh(x)

    def forward(self, d):
        # d: already roughly in [-1, 1]
        # centered version，保留和 RGB 对齐的分布风格
        d_c = d

        # 1) 主信号：保持 centered
        main = d_c

        # 2) log 分支：
        # 先把 [-1,1] 映到 [0,1]，做 log 压缩，再映回 [-1,1]
        d_01 = 0.5 * (d_c + 1.0)
        log_base = torch.log(
            torch.tensor(1.0 + self.log_scale, device=d.device, dtype=d.dtype)
        )
        d_log01 = torch.log1p(self.log_scale * d_01) / log_base
        d_log = 2.0 * d_log01 - 1.0

        # 3) 低频分支：仍保持 centered
        blur = F.avg_pool2d(
            d_c,
            kernel_size=self.blur_kernel,
            stride=1,
            padding=self.blur_kernel // 2
        )

        # 4) 高频残差：建议做一个软压缩，避免幅值太尖
        resid_raw = d_c - blur
        resid = self._soft_range(self.resid_gain * resid_raw)

        # 5) 坐标调制：天然就比较接近 [-1,1]
        B, _, H, W = d.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=d.device, dtype=d.dtype),
            torch.linspace(-1, 1, W, device=d.device, dtype=d.dtype),
            indexing='ij'
        )
        xx = xx.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        dx = xx * d_c
        dy = yy * d_c

        feat = torch.cat([main, d_log, blur, resid, dx, dy], dim=1)
        return feat
class DWBasic_Context_Encoder_resconvb_mixed(nn.Module):
    """
    Context encoder 的 mixed 版：
    - stem: 第一层全分辨率聚合；第二个 stem 分支放到下采样后
    - layer1/layer2: Hybrid Dilated DW
    - layer3/layer4/up: 原 DWResidualBlock
    - 输出接口保持 context 版语义：
        enc_out4 : output_dim
        up2_out  : output_dim
        up1_out  : output_dim
        up0_out  : 128   (only for CCMR+)
    """
    def __init__(
        self,
        input_dim=3,
        output_dim=256,
        norm_fn='group',
        model_type="CCMR+",
        stem_kernel1=7,
        stem_kernel2=5,
        stem_post_dilation=2,
        hybrid_kernels=(5, 5),
        hybrid_dilations=((1, 2), (1, 3)),
        stage_kernels=(3, 3),   # layer3, layer4
        up_kernel=3,
        base_dims=(48, 80, 112, 144),
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type

        c1, c2, c3, c4 = base_dims

        # -------------------------
        # Stem part 1: full-res aggregation
        # -------------------------
        self.stem_full = nn.Sequential(
            nn.Conv2d(
                input_dim, c1,
                kernel_size=stem_kernel1,
                stride=1,
                padding=stem_kernel1 // 2,
                bias=False
            ),
            _make_norm(self.norm_fn, c1),
            nn.ReLU(inplace=True),
        )
        # 下采样后移
        self.stem_down = nn.AvgPool2d(kernel_size=2, stride=2)

        # -------------------------
        # Stem part 2: moved after downsample
        # -------------------------
        self.stem_post = nn.Sequential(
            nn.Conv2d(
                c1, c1,
                kernel_size=stem_kernel2,
                stride=1,
                padding=_dw_padding(stem_kernel2, stem_post_dilation),
                dilation=stem_post_dilation,
                groups=c1,
                bias=False
            ),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            _make_norm(self.norm_fn, c1),
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Top-down feature extraction
        # layer1/layer2: Hybrid
        # layer3/layer4: original DWResidualBlock
        # -------------------------
        self.in_planes = c1

        self.layer1 = self._make_hybrid_layer(
            c1,
            stride=1,
            kernel_size=hybrid_kernels[0],
            dilations=hybrid_dilations[0]
        )
        self.layer2 = self._make_hybrid_layer(
            c2,
            stride=2,
            kernel_size=hybrid_kernels[1],
            dilations=hybrid_dilations[1]
        )

        self.layer3 = self._make_dw_layer(
            c3,
            stride=2,
            kernel_size=stage_kernels[0]
        )
        self.layer4 = self._make_dw_layer(
            c4,
            stride=2,
            kernel_size=stage_kernels[1]
        )

        self.conv2 = nn.Conv2d(c4, output_dim, kernel_size=1, bias=False)

        # -------------------------
        # Bottom-up feature consolidation
        # up 全部保持原 DWResidualBlock
        # -------------------------
        self.in_planes = output_dim + c3
        self.up_layer2 = self._make_dw_layer(128, stride=1, kernel_size=up_kernel)
        self.after_up_layer2_conv = nn.Conv2d(128, output_dim, kernel_size=1, bias=False)

        self.in_planes = output_dim + c2
        self.up_layer1 = self._make_dw_layer(96, stride=1, kernel_size=up_kernel)
        self.after_up_layer1_conv = nn.Conv2d(96, output_dim, kernel_size=1, bias=False)

        if self.model_type.casefold() == "ccmr+":
            self.in_planes = output_dim + c1
            self.up_layer0 = self._make_dw_layer(64, stride=1, kernel_size=up_kernel)
            self.after_up_layer0_conv = nn.Conv2d(64, 128, kernel_size=1, bias=False)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_hybrid_layer(self, dim, stride=1, kernel_size=5, dilations=(1, 3)):
        layer1 = HybridDilatedDWResidualBlock(
            self.in_planes, dim,
            norm_fn=self.norm_fn,
            stride=stride,
            kernel_size=kernel_size,
            dilations=dilations
        )
        layer2 = HybridDilatedDWResidualBlock(
            dim, dim,
            norm_fn=self.norm_fn,
            stride=1,
            kernel_size=kernel_size,
            dilations=dilations
        )
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def _make_dw_layer(self, dim, stride=1, kernel_size=3):
        layer1 = DWResidualBlock(
            self.in_planes, dim,
            self.norm_fn,
            stride=stride,
            kernel_size=kernel_size
        )
        layer2 = DWResidualBlock(
            dim, dim,
            self.norm_fn,
            stride=1,
            kernel_size=kernel_size
        )
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            x = torch.cat(x, dim=0)

        # stem
        x = self.stem_full(x)   # full-resolution aggregation
        x = self.stem_down(x)   # delayed downsampling
        x = self.stem_post(x)   # second stem branch after downsampling

        # encoder
        x = self.layer1(x)
        enc_out1 = x

        x = self.layer2(x)
        enc_out2 = x

        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x

        # up layer2
        enc_out4_resized = F.interpolate(
            enc_out4, size=enc_out3.shape[-2:], mode='bilinear', align_corners=False
        )
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))

        # up layer1
        up2_out_resized = F.interpolate(
            up2_out, size=enc_out2.shape[-2:], mode='bilinear', align_corners=False
        )
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))

        if self.model_type.casefold() == "ccmr+":
            # up layer0
            up1_out_resized = F.interpolate(
                up1_out, size=enc_out1.shape[-2:], mode='bilinear', align_corners=False
            )
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
            return [enc_out4, up2_out, up1_out, up0_out]

        elif self.model_type.casefold() == "ccmr":
            return [enc_out4, up2_out, up1_out]
