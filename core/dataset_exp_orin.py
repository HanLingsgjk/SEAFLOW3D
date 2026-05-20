# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
#原汁原味的DATASETS
from utils.utils import  coords_grid
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import os
import pickle
import math
import random
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib
from core.utils.flow_viz import flow2rgb,getvis
from core.utils.bezier import mybezier
from core.utils.rectangle_noise import retangle
from core.utils import frame_utils
from matplotlib import cm
import  cv2

from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentorm,DFFAugmentorm
def coords_grid(ht, wd):
    coords = torch.meshgrid(torch.arange(wd), torch.arange(ht))
    coords = torch.stack(coords[::-1], dim=2).float()
    return coords

class FlowDatasetflow(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.bezier = mybezier()
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.maskall = []
        self.occmask = []
        self.isnerf = False
        self.rand = True
        self.o3dmap_list = sorted(glob(osp.join("/new_data/kitti3d/training/instance_2/", "*.png")))
        self.image3d_list = sorted(glob(osp.join("/new_data/kitti3d/training/image_2/", "*.png")))
    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        indx = torch.randint(0, 7481, (1, 1))
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        objmask = frame_utils.read_gen(self.o3dmap_list[indx[0, 0]])
        objmask = np.array(objmask).astype(np.uint8)
        imgmask = cv2.imread(self.image3d_list[indx])

        if self.isnerf:
            ssim,d2d,aphla = frame_utils.readNerfMask(self.maskall[index])
            valid_mask = frame_utils.read_gen(self.occmask[index])
            valid_mask = np.array(valid_mask).astype(np.float32)/255
            dmask = np.array(d2d<0.1).astype(np.float32)
            amask = np.array(aphla > 0.7).astype(np.float32)
            smask = np.array(ssim < 0.1).astype(np.float32)
            valid = valid_mask

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        if self.rand ==True:
            for i in range(2):
                imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,imgmask)
                if flag>1 :
                    img1[imgb1 != 0] = imgb1[imgb1 != 0]
                    img2[imgb2 != 0] = imgb2[imgb2 != 0]
                    flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                    if self.sparse:
                        valid[imgb1[:,:,0]!=0] = 1
        self.last_image = img2

        '''
        plt.imshow(flow[:,:,0])
        plt.show()
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()

        plt.imshow(d2d)
        plt.show()
        plt.imshow(ssim)
        plt.show()
        plt.imshow(aphla)
        plt.show()
        plt.imshow(valid)
        plt.show()

        '''
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.maskall = v*self.maskall
        self.occmask = v*self.occmask
        return self

    def __len__(self):
        return len(self.image_list)
def bilinear_sampler(img,coords , mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    #img  C W H
    #coord W H 2
    img = img.unsqueeze(0).unsqueeze(0)
    coords =coords.unsqueeze(0)
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True,mode=mode)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    if len(x.shape) != len(w.shape):
        w = np.broadcast_to(w[..., None], x.shape)
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)



def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth
def readPFM(file):
    import re
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:
        return readPFM(path)[0]
def get_grid_np(B,H,W):
    meshgrid_base = np.meshgrid(range(0, W), range(0, H))[::-1]
    basey = np.reshape(meshgrid_base[0], [1, 1, 1, H, W])
    basex = np.reshape(meshgrid_base[1], [1, 1, 1, H, W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).float()
    return grid.view( H, W, 2)
class AnythingDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DFFAugmentorm(**aug_params)

        self.Scale_list = []
        self.flow_list = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0
        self.last_image = np.random.randn(320,960,3)

    def __getitem__(self, index):
        self.kit = self.kit +1
        index = index % len(self.image_list)
        flow, mask = frame_utils.readFlowFFD(self.flow_list[index])
        dc_change = frame_utils.readDC(self.Scale_list[index])
        #mask = np.ones_like(mask)
        conf = mask
        valid = mask>0
        dc_change = np.concatenate((dc_change[:, :, np.newaxis], valid[:, :, np.newaxis]), axis=2)
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)

        if self.augmentor is not None:
            img1, img2, flow,dc_change,conf, valid = self.augmentor(img1, img2, flow,dc_change,conf, valid)
        for i in range(1):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            kit1 = (imgb1 != 0).sum()
            kit2 = (imgb2 != 0).sum()
            flag2 = abs(kit2-kit1)/(kit2+kit1+1)
            if flag>1 and flag2<0.5:

                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                    conf[imgb1[:, :, 0] != 0] = 0.95
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1
        self.last_image = img2


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        conf = torch.from_numpy(conf).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, dc_change, valid.float(),conf

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.Scale_list = v * self.Scale_list
        return self

    def __len__(self):
        return len(self.image_list)


class FlowTODepthDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DFFAugmentorm(**aug_params)

        self.Scale_list = []
        self.flow_list = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.fflow_list =[]
        self.fmask_DPS =[]
        self.fmask_PPF =[]
        self.fGSddc =[]
        self.image_listf = []
        self.fimage12 = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0
        self.last_image = np.random.randn(320,960,3)

    def resize_sparse_exp(self, exp, valid, fx=1.0, fy=1.0):
            ht, wd = exp.shape[:2]
            coords = np.meshgrid(np.arange(wd), np.arange(ht))
            coords = np.stack(coords, axis=-1)

            coords = coords.reshape(-1, 2).astype(np.float32)
            exp = exp.reshape(-1).astype(np.float32)
            valid = valid.reshape(-1).astype(np.float32)

            coords0 = coords[valid >= 1]
            exp0 = exp[valid >= 1]

            ht1 = int(round(ht * fy))
            wd1 = int(round(wd * fx))

            coords1 = coords0 * [fx, fy]
            exp1 = exp0

            xx = np.round(coords1[:, 0]).astype(np.int32)
            yy = np.round(coords1[:, 1]).astype(np.int32)

            v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
            xx = xx[v]
            yy = yy[v]
            exp1 = exp1[v]

            exp_img = np.zeros([ht1, wd1], dtype=np.float32)
            valid_img = np.zeros([ht1, wd1], dtype=np.int32)

            exp_img[yy, xx] = exp1
            valid_img[yy, xx] = 1

            return exp_img, valid_img

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
            ht, wd = flow.shape[:2]
            coords = np.meshgrid(np.arange(wd), np.arange(ht))
            coords = np.stack(coords, axis=-1)

            coords = coords.reshape(-1, 2).astype(np.float32)
            flow = flow.reshape(-1, 2).astype(np.float32)
            valid = valid.reshape(-1).astype(np.float32)

            coords0 = coords[valid >= 1]
            flow0 = flow[valid >= 1]

            ht1 = int(round(ht * fy))
            wd1 = int(round(wd * fx))

            coords1 = coords0 * [fx, fy]
            flow1 = flow0 * [fx, fy]

            xx = np.round(coords1[:, 0]).astype(np.int32)
            yy = np.round(coords1[:, 1]).astype(np.int32)

            v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
            xx = xx[v]
            yy = yy[v]
            flow1 = flow1[v]

            flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
            valid_img = np.zeros([ht1, wd1], dtype=np.int32)

            flow_img[yy, xx] = flow1
            valid_img[yy, xx] = 1

            return flow_img, valid_img

    def addfore(self, im1in, im2in, flowin, dcchangein, validin,confin):
            # TODO 添加随机前景

            indexf = np.random.randint(0, self.image_listf.__len__())

            d1, d2, dc_change = frame_utils.readNerfddc(self.fGSddc[indexf])
            flow, valid = frame_utils.readFlowKITTI(self.fflow_list[indexf])
            pds, rdds, ssim = frame_utils.readNerfMask(self.fmask_DPS[indexf])
            pah1, pah2, occall = frame_utils.readNerfMask(self.fmask_PPF[indexf])
            img01f = frame_utils.read_gen(self.image_listf[indexf][0])
            img02f = frame_utils.read_gen(self.image_listf[indexf][1])
            img01f = np.array(img01f).astype(np.uint8)
            img02f = np.array(img02f).astype(np.uint8)

            pmask = np.array(pds < 2).astype(np.float32)
            amask = np.array(pah1 < 0.1).astype(np.float32)
            valid_mask = occall < 0.02
            dmask = np.array(rdds < 0.01).astype(np.float32)
            smask = np.array(ssim > 0.9).astype(np.float32)
            valid_mask = np.array(valid_mask).astype(np.float32)

            forevalid1 = img01f[:, :, 0] > 0
            forevalid2 = img02f[:, :, 0] > 0
            forevalid = forevalid1 + forevalid2
            # 试一下没有amask的
            validf = dmask * valid_mask * pmask * amask

            coordsf = np.where(forevalid > 0)
            minydf = coordsf[0].min()
            maxydf = coordsf[0].max()
            minxdf = coordsf[1].min()
            maxxdf = coordsf[1].max()

            img01f = img01f[minydf:maxydf, minxdf:maxxdf, :]
            img02f = img02f[minydf:maxydf, minxdf:maxxdf, :]
            validf = validf[minydf:maxydf, minxdf:maxxdf]
            flowf = flow[minydf:maxydf, minxdf:maxxdf, :]
            dcf = dc_change[minydf:maxydf, minxdf:maxxdf]

            # 随机前景增强，大小缩放，随机位置叠加，随机放射变化
            scale = 2 ** np.random.uniform(-0.4, 0.4)  # !!!!!!!!!!!!!!!!!!0.4
            img1 = cv2.resize(img01f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img02f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flowf, validf, fx=scale, fy=scale)
            dc_out, _ = self.resize_sparse_exp(dcf, validf, fx=scale, fy=scale)
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                valid = valid[:, ::-1]
                dc_out = dc_out[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < 0.1:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                dc_out = dc_out[::-1, :]
                valid = valid[::-1, :]

            if (im1in.shape[0] - img1.shape[0]) > 0 and (im1in.shape[1] - img1.shape[1]) > 0:
                y0 = np.random.randint(0, im1in.shape[0] - img1.shape[0])
                x0 = np.random.randint(0, im1in.shape[1] - img1.shape[1])
                im1inm = np.zeros_like(im1in)
                im2inm = np.zeros_like(im2in)
                flowinm = np.zeros_like(flowin)
                dcchangeinm = np.zeros_like(dcchangein)
                validinm = np.zeros_like(validin)

                im1inm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = img1
                im2inm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = img2
                flowinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = flow
                dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 0] = dc_out
                dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 1] = valid
                validinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = valid

                im1in[im1inm[:, :, 0] > 0] = im1inm[im1inm[:, :, 0] > 0]
                im2in[im2inm[:, :, 0] > 0] = im2inm[im2inm[:, :, 0] > 0]
                flowin[im1inm[:, :, 0] > 0] = flowinm[im1inm[:, :, 0] > 0]
                dcchangein[im1inm[:, :, 0] > 0] = dcchangeinm[im1inm[:, :, 0] > 0]
                validin[im1inm[:, :, 0] > 0] = validinm[im1inm[:, :, 0] > 0]
                confin[im1inm[:, :, 0] > 0] = 1
            return im1in, im2in, flowin, dcchangein, validin,confin
    def __getitem__(self, index):
        self.kit = self.kit +1
        index = index % len(self.image_list)
        flow, mask = frame_utils.readFlowFFD(self.flow_list[index])
        dc_change = frame_utils.readDC(self.Scale_list[index])
        conf = mask
        valid = mask>0

        dc_change = np.concatenate((dc_change[:, :, np.newaxis], valid[:, :, np.newaxis]), axis=2)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)

        '''
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow2rgb(flow))
        plt.show()
        plt.imshow(dc_change[:,:,0])
        plt.show()
        plt.imshow(conf)
        plt.show()
        plt.imshow(valid)
        plt.show()
        '''
        if self.augmentor is not None:
            img1, img2, flow,dc_change,conf, valid = self.augmentor(img1, img2, flow,dc_change,conf, valid)
            #img1, img2, flow, dc_change, valid,conf = self.addfore(img1, img2, flow, dc_change, valid,conf)
        for i in range(0):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            kit1 = (imgb1 != 0).sum()
            kit2 = (imgb2 != 0).sum()
            flag2 = abs(kit2-kit1)/(kit2+kit1+1)
            if flag>1 and flag2<0.5:

                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                    conf[imgb1[:, :, 0] != 0] = 0.95
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1
        self.last_image = img2
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        conf = torch.from_numpy(conf).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, dc_change, valid.float(),conf

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.Scale_list = v * self.Scale_list
        return self

    def __len__(self):
        return len(self.image_list)
class HotGSDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.isnerf = False
        self.flow_list = []
        self.mask_DPS = []
        self.mask_PPF = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.GSddc = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0
        self.fflow_list =[]
        self.fmask_DPS =[]
        self.fmask_PPF =[]
        self.fGSddc =[]
        self.image_listf = []
        self.fimage12 = []
        self.last_image = np.random.randn(320,960,3)
    def resize_sparse_exp(self, exp, valid, fx=1.0, fy=1.0):
            ht, wd = exp.shape[:2]
            coords = np.meshgrid(np.arange(wd), np.arange(ht))
            coords = np.stack(coords, axis=-1)

            coords = coords.reshape(-1, 2).astype(np.float32)
            exp = exp.reshape(-1).astype(np.float32)
            valid = valid.reshape(-1).astype(np.float32)

            coords0 = coords[valid >= 1]
            exp0 = exp[valid >= 1]

            ht1 = int(round(ht * fy))
            wd1 = int(round(wd * fx))

            coords1 = coords0 * [fx, fy]
            exp1 = exp0

            xx = np.round(coords1[:, 0]).astype(np.int32)
            yy = np.round(coords1[:, 1]).astype(np.int32)

            v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
            xx = xx[v]
            yy = yy[v]
            exp1 = exp1[v]

            exp_img = np.zeros([ht1, wd1], dtype=np.float32)
            valid_img = np.zeros([ht1, wd1], dtype=np.int32)

            exp_img[yy, xx] = exp1
            valid_img[yy, xx] = 1

            return exp_img, valid_img
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
            ht, wd = flow.shape[:2]
            coords = np.meshgrid(np.arange(wd), np.arange(ht))
            coords = np.stack(coords, axis=-1)
            coords = coords.reshape(-1, 2).astype(np.float32)
            flow = flow.reshape(-1, 2).astype(np.float32)
            valid = valid.reshape(-1).astype(np.float32)
            coords0 = coords[valid >= 1]
            flow0 = flow[valid >= 1]
            ht1 = int(round(ht * fy))
            wd1 = int(round(wd * fx))
            coords1 = coords0 * [fx, fy]
            flow1 = flow0 * [fx, fy]
            xx = np.round(coords1[:, 0]).astype(np.int32)
            yy = np.round(coords1[:, 1]).astype(np.int32)
            v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
            xx = xx[v]
            yy = yy[v]
            flow1 = flow1[v]
            flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
            valid_img = np.zeros([ht1, wd1], dtype=np.int32)
            flow_img[yy, xx] = flow1
            valid_img[yy, xx] = 1
            return flow_img, valid_img
    def addfore(self, im1in, im2in, flowin, dcchangein, validin):
            # TODO 添加随机前景
            indexf = np.random.randint(0, self.image_listf.__len__())
            d1, d2, dc_change = frame_utils.readNerfddc(self.fGSddc[indexf])
            flow, valid = frame_utils.readFlowKITTI(self.fflow_list[indexf])
            pds, rdds, ssim = frame_utils.readNerfMask(self.fmask_DPS[indexf])
            pah1, pah2, occall = frame_utils.readNerfMask(self.fmask_PPF[indexf])
            img01f = frame_utils.read_gen(self.image_listf[indexf][0])
            img02f = frame_utils.read_gen(self.image_listf[indexf][1])
            img01f = np.array(img01f).astype(np.uint8)
            img02f = np.array(img02f).astype(np.uint8)

            pmask = np.array(pds < 2).astype(np.float32)
            amask = np.array(pah1 < 0.1).astype(np.float32)
            valid_mask = occall < 0.02
            dmask = np.array(rdds < 0.01).astype(np.float32)
            smask = np.array(ssim > 0.9).astype(np.float32)
            valid_mask = np.array(valid_mask).astype(np.float32)

            forevalid1 = img01f[:, :, 0] > 0
            forevalid2 = img02f[:, :, 0] > 0
            forevalid = forevalid1 + forevalid2
            # 试一下没有amask的
            validf = dmask * valid_mask * pmask * amask
            coordsf = np.where(forevalid > 0)
            minydf = coordsf[0].min()
            maxydf = coordsf[0].max()
            minxdf = coordsf[1].min()
            maxxdf = coordsf[1].max()
            img01f = img01f[minydf:maxydf, minxdf:maxxdf, :]
            img02f = img02f[minydf:maxydf, minxdf:maxxdf, :]
            validf = validf[minydf:maxydf, minxdf:maxxdf]
            flowf = flow[minydf:maxydf, minxdf:maxxdf, :]
            dcf = dc_change[minydf:maxydf, minxdf:maxxdf]
            # 随机前景增强，大小缩放，随机位置叠加，随机放射变化
            scale = 2 ** np.random.uniform(-0.8, 0.0)  # !!!!!!!!!!!!!!!!!!0.4
            img1 = cv2.resize(img01f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img02f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flowf, validf, fx=scale, fy=scale)
            dc_out, _ = self.resize_sparse_exp(dcf, validf, fx=scale, fy=scale)
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                valid = valid[:, ::-1]
                dc_out = dc_out[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < 0.1:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                dc_out = dc_out[::-1, :]
                valid = valid[::-1, :]

            if (im1in.shape[0] - img1.shape[0]) > 0 and (im1in.shape[1] - img1.shape[1]) > 0:
                y0 = np.random.randint(0, im1in.shape[0] - img1.shape[0])
                x0 = np.random.randint(0, im1in.shape[1] - img1.shape[1])
                im1inm = np.zeros_like(im1in)
                im2inm = np.zeros_like(im2in)
                flowinm = np.zeros_like(flowin)
                dcchangeinm = np.zeros_like(dcchangein)
                validinm = np.zeros_like(validin)

                im1inm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = img1
                im2inm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = img2
                flowinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = flow
                dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 0] = dc_out
                dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 1] = valid
                validinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1]] = valid

                im1in[im1inm[:, :, 0] > 0] = im1inm[im1inm[:, :, 0] > 0]
                im2in[im2inm[:, :, 0] > 0] = im2inm[im2inm[:, :, 0] > 0]
                flowin[im1inm[:, :, 0] > 0] = flowinm[im1inm[:, :, 0] > 0]
                dcchangein[im1inm[:, :, 0] > 0] = dcchangeinm[im1inm[:, :, 0] > 0]
                validin[im1inm[:, :, 0] > 0] = validinm[im1inm[:, :, 0] > 0]
            return im1in, im2in, flowin, dcchangein, validin
    def __getitem__(self, index):
        self.kit = self.kit +1
        index = index % len(self.image_list)
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        pds, rdds, ssim = frame_utils.readNerfMask(self.mask_DPS[index])
        pah1, pah2, occall = frame_utils.readNerfMask(self.mask_PPF[index])

        pmask = np.array(pds < 1).astype(np.float32)
        amask = np.array(pah1 < 0.2).astype(np.float32)
        valid_mask = occall<0.02
        dmask = np.array(rdds < 0.005).astype(np.float32)
        smask = np.array(ssim >0.9).astype(np.float32)

        valid_mask = np.array(valid_mask).astype(np.float32)
        #试一下没有amask的
        valid = amask*dmask*valid_mask
        #valid = np.ones_like(valid)
        '''
        plt.imshow(amask)
        plt.show()
        plt.imshow(dmask)
        plt.show()
        plt.imshow(valid_mask)
        plt.show()
        plt.imshow(valid)
        plt.show()
        '''
        d1, d2, dc_change = frame_utils.readNerfddc(self.GSddc[index])
        #ao = frame_utils.readAO(self.AOmask[index])
        dc_change = np.concatenate((dc_change[:, :, np.newaxis], valid[:, :, np.newaxis]), axis=2)
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)
        depth1 = np.array(d1).astype(np.float32)
        #如果要纯静态的输出
        if np.random.random(1)<0.01:
            img2  =  img1
            flow = flow*0
            valid = np.ones_like(valid)
            dc_change = np.ones_like(dc_change)

        '''
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow2rgb(flow))
        plt.show()
        plt.imshow(dc_change[:,:,0])
        plt.show()
        plt.imshow(valid)
        plt.show()
        '''
        if self.augmentor is not None:
            img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
            img1, img2, flow, dc_change, valid = self.addfore(img1, img2, flow, dc_change, valid)
            img1, img2, flow, dc_change, valid = self.addfore(img1, img2, flow, dc_change, valid)
            img1, img2, flow, dc_change, valid = self.addfore(img1, img2, flow, dc_change, valid)
        for i in range(0):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            if flag>1 :
                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1
        self.last_image = img2
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        conf = valid.clone()
        batch,xo,yo = img1.shape
        #if xo<320 or yo<720:
        #    print('get it')
        return img1, img2, flow, dc_change, valid.float(),conf.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.mask_DPS = v*self.mask_DPS
        self.mask_PPF = v*self.mask_PPF
        self.depth_list = v * self.depth_list
        self.GSddc = v * self.GSddc
        return self

    def __len__(self):
        return len(self.image_list)
class GSDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.isnerf = False
        self.flow_list = []
        self.mask_DPS = []
        self.mask_PPF = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.GSddc = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0
        self.fflow_list =[]
        self.fmask_DPS =[]
        self.fmask_PPF =[]
        self.fGSddc =[]
        self.image_listf = []
        self.fimage12 = []
        self.last_image = np.random.randn(320,960,3)
    def resize_sparse_exp(self, exp, valid, fx=1.0, fy=1.0):
        ht, wd = exp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        exp = exp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        exp0 = exp[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        exp1 = exp0

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        exp1 = exp1[v]

        exp_img = np.zeros([ht1, wd1], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        exp_img[yy, xx] = exp1
        valid_img[yy, xx] = 1

        return exp_img, valid_img
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def addfore(self,im1in,im2in,flowin,dcchangein,validin):
        #TODO 添加随机前景

        indexf = np.random.randint(0, self.image_listf.__len__())

        d1, d2, dc_change = frame_utils.readNerfddc(self.fGSddc[indexf])
        flow, valid = frame_utils.readFlowKITTI(self.fflow_list[indexf])
        pds, rdds, ssim = frame_utils.readNerfMask(self.fmask_DPS[indexf])
        pah1, pah2, occall = frame_utils.readNerfMask(self.fmask_PPF[indexf])
        img01f = frame_utils.read_gen(self.image_listf[indexf][0])
        img02f = frame_utils.read_gen(self.image_listf[indexf][1])
        img01f = np.array(img01f).astype(np.uint8)
        img02f = np.array(img02f).astype(np.uint8)

        pmask = np.array(pds < 2).astype(np.float32)
        amask = np.array(pah1 < 0.1).astype(np.float32)
        valid_mask = occall<0.02
        dmask = np.array(rdds < 0.01).astype(np.float32)
        smask = np.array(ssim >0.9).astype(np.float32)
        valid_mask = np.array(valid_mask).astype(np.float32)

        forevalid1 = img01f[:,:,0]>0
        forevalid2  = img02f[:, :, 0] > 0
        forevalid = forevalid1 + forevalid2
        #试一下没有amask的
        validf = dmask *valid_mask*pmask*amask

        coordsf = np.where(forevalid > 0)
        minydf = coordsf[0].min()
        maxydf = coordsf[0].max()
        minxdf = coordsf[1].min()
        maxxdf = coordsf[1].max()

        img01f = img01f[minydf:maxydf, minxdf:maxxdf,:]
        img02f = img02f[minydf:maxydf, minxdf:maxxdf,:]
        validf = validf[minydf:maxydf, minxdf:maxxdf]
        flowf =  flow[minydf:maxydf, minxdf:maxxdf,:]
        dcf = dc_change[minydf:maxydf, minxdf:maxxdf]
        '''
        plt.imshow(img01f)
        plt.show()
        plt.imshow(img02f)
        plt.show()

        validf[(img01f<1)[:,:,0]] = 0
        plt.imshow(validf)
        plt.show()
        flows = np.zeros_like(flow2rgb(flowf))
        flows[(validf>0),:] = flow2rgb(flowf)[(validf>0),:]
        plt.imshow(flows)
        plt.show()
        '''
        #随机前景增强，大小缩放，随机位置叠加，随机放射变化
        scale = 2 ** np.random.uniform(-0.4, 0.4)#!!!!!!!!!!!!!!!!!!0.4
        img1 = cv2.resize(img01f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img02f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        flow, valid = self.resize_sparse_flow_map(flowf, validf, fx=scale, fy=scale)
        dc_out, _ = self.resize_sparse_exp(dcf,validf, fx=scale, fy=scale)
        if np.random.rand() < 0.5: # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            valid = valid[:,::-1]
            dc_out = dc_out[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
        if np.random.rand() < 0.1: # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]
            dc_out = dc_out[::-1, :]
            valid = valid[::-1, :]

        if (im1in.shape[0] - img1.shape[0])>0 and (im1in.shape[1] - img1.shape[1])>0:
            y0 = np.random.randint(0, im1in.shape[0] - img1.shape[0])
            x0 = np.random.randint(0, im1in.shape[1] - img1.shape[1])
            im1inm = np.zeros_like(im1in)
            im2inm = np.zeros_like(im2in)
            flowinm = np.zeros_like(flowin)
            dcchangeinm = np.zeros_like(dcchangein)
            validinm = np.zeros_like(validin)

            im1inm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img1
            im2inm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img2
            flowinm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = flow
            dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1],0] = dc_out
            dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 1] = valid
            validinm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = valid


            im1in[im1inm[:,:,0]>0] = im1inm[im1inm[:,:,0]>0]
            im2in[im2inm[:,:,0]>0]  = im2inm[im2inm[:,:,0]>0]
            flowin[im1inm[:,:,0]>0]  = flowinm[im1inm[:,:,0]>0]
            dcchangein[im1inm[:, :, 0] > 0] = dcchangeinm[im1inm[:, :, 0] > 0]
            validin[im1inm[:,:,0]>0]  = validinm[im1inm[:,:,0]>0]

        '''
        plt.imshow(im1in)
        plt.show()
        plt.imshow(im2in)
        plt.show()
        plt.imshow(flowin[:,:,0])
        plt.show()
        plt.imshow(dcchangein[:,:,0])
        plt.show()
        plt.imshow(validin)
        plt.show()
        '''
        return im1in,im2in,flowin,dcchangein,validin
    def __getitem__(self, index):
        self.kit = self.kit +1
        index = index % len(self.image_list)
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])

        pds, rdds, ssim = frame_utils.readNerfMask(self.mask_DPS[index])
        pah1, pah2, occall = frame_utils.readNerfMask(self.mask_PPF[index])

        pmask = np.array(pds < 1).astype(np.float32)
        amask = np.array(pah1 < 0.06).astype(np.float32)
        valid_mask = occall<0.02
        dmask = np.array(rdds < 0.005).astype(np.float32)
        smask = np.array(ssim >0.9).astype(np.float32)

        valid_mask = np.array(valid_mask).astype(np.float32)
        #试一下没有amask的
        valid = amask*dmask*valid_mask

        d1, d2, dc_change = frame_utils.readNerfddc(self.GSddc[index])
        #ao = frame_utils.readAO(self.AOmask[index])
        dc_change = np.concatenate((dc_change[:, :, np.newaxis], valid[:, :, np.newaxis]), axis=2)


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])


        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)
        depth1 = np.array(d1).astype(np.float32)
        #如果要纯静态的输出

        if np.random.random(1)<0.01:
            img2  =  img1
            flow = flow*0
            valid = np.ones_like(valid)
            dc_change = np.ones_like(dc_change)

        '''
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow2rgb(flow))
        plt.show()
        plt.imshow(getvis(dc_change[:,:,0],colorbar='turbo',lo=0.8,hi=100-0.8))
        plt.show()
        plt.imshow(pmask)
        plt.show()
        plt.imshow(valid)
        plt.show()
        '''
        if self.augmentor is not None:
            img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
            img1, img2, flow, dc_change, valid = self.addfore(img1, img2, flow,dc_change, valid)
        for i in range(1):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            if flag>1 :
                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1
        self.last_image = img2
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        conf = valid.clone()
        batch,xo,yo = img1.shape
        if xo<320 or yo<720:
            print('get it')
        return img1, img2, flow, dc_change, valid.float(),conf.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.mask_DPS = v*self.mask_DPS
        self.mask_PPF = v*self.mask_PPF
        self.depth_list = v * self.depth_list
        self.GSddc = v * self.GSddc
        return self

    def __len__(self):
        return len(self.image_list)
class NerfDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.isnerf = False
        self.flow_list = []
        self.maskall = []
        self.occmask = []
        self.AOmask = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.NERFddc = []
        self.fflow_list =[]
        self.fmask_DPS =[]
        self.fmask_PPF =[]
        self.fGSddc =[]
        self.image_listf = []
        self.fimage12 = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0

        self.last_image = np.random.randn(320,960,3)
    def resize_sparse_exp(self, exp, valid, fx=1.0, fy=1.0):
        ht, wd = exp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        exp = exp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        exp0 = exp[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        exp1 = exp0

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        exp1 = exp1[v]

        exp_img = np.zeros([ht1, wd1], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        exp_img[yy, xx] = exp1
        valid_img[yy, xx] = 1

        return exp_img, valid_img
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img
    def addfore(self,im1in,im2in,flowin,dcchangein,validin):
        #TODO 添加随机前景

        indexf = np.random.randint(0, self.image_listf.__len__())

        d1, d2, dc_change = frame_utils.readNerfddc(self.fGSddc[indexf])
        flow, valid = frame_utils.readFlowKITTI(self.fflow_list[indexf])
        pds, rdds, ssim = frame_utils.readNerfMask(self.fmask_DPS[indexf])
        pah1, pah2, occall = frame_utils.readNerfMask(self.fmask_PPF[indexf])
        img01f = frame_utils.read_gen(self.image_listf[indexf][0])
        img02f = frame_utils.read_gen(self.image_listf[indexf][1])
        img01f = np.array(img01f).astype(np.uint8)
        img02f = np.array(img02f).astype(np.uint8)

        pmask = np.array(pds < 2).astype(np.float32)
        amask = np.array(pah1 < 0.1).astype(np.float32)
        valid_mask = occall<0.02
        dmask = np.array(rdds < 0.01).astype(np.float32)
        smask = np.array(ssim >0.9).astype(np.float32)
        valid_mask = np.array(valid_mask).astype(np.float32)

        forevalid1 = img01f[:,:,0]>0
        forevalid2  = img02f[:, :, 0] > 0
        forevalid = forevalid1 + forevalid2
        #试一下没有amask的
        validf = dmask *valid_mask*pmask*amask

        coordsf = np.where(forevalid > 0)
        minydf = coordsf[0].min()
        maxydf = coordsf[0].max()
        minxdf = coordsf[1].min()
        maxxdf = coordsf[1].max()

        img01f = img01f[minydf:maxydf, minxdf:maxxdf,:]
        img02f = img02f[minydf:maxydf, minxdf:maxxdf,:]
        validf = validf[minydf:maxydf, minxdf:maxxdf]
        flowf =  flow[minydf:maxydf, minxdf:maxxdf,:]
        dcf = dc_change[minydf:maxydf, minxdf:maxxdf]
        '''
        plt.imshow(img01f)
        plt.show()
        plt.imshow(img02f)
        plt.show()

        validf[(img01f<1)[:,:,0]] = 0
        plt.imshow(validf)
        plt.show()
        flows = np.zeros_like(flow2rgb(flowf))
        flows[(validf>0),:] = flow2rgb(flowf)[(validf>0),:]
        plt.imshow(flows)
        plt.show()
        '''
        #随机前景增强，大小缩放，随机位置叠加，随机放射变化
        scale = 2 ** np.random.uniform(-0.4, 0.4)#!!!!!!!!!!!!!!!!!!0.4
        img1 = cv2.resize(img01f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img02f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        flow, valid = self.resize_sparse_flow_map(flowf, validf, fx=scale, fy=scale)
        dc_out, _ = self.resize_sparse_exp(dcf,validf, fx=scale, fy=scale)
        if np.random.rand() < 0.5: # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            valid = valid[:,::-1]
            dc_out = dc_out[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
        if np.random.rand() < 0.1: # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]
            dc_out = dc_out[::-1, :]
            valid = valid[::-1, :]

        if (im1in.shape[0] - img1.shape[0])>0 and (im1in.shape[1] - img1.shape[1])>0:
            y0 = np.random.randint(0, im1in.shape[0] - img1.shape[0])
            x0 = np.random.randint(0, im1in.shape[1] - img1.shape[1])
            im1inm = np.zeros_like(im1in)
            im2inm = np.zeros_like(im2in)
            flowinm = np.zeros_like(flowin)
            dcchangeinm = np.zeros_like(dcchangein)
            validinm = np.zeros_like(validin)

            im1inm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img1
            im2inm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img2
            flowinm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = flow
            dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1],0] = dc_out
            dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 1] = valid
            validinm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = valid


            im1in[im1inm[:,:,0]>0] = im1inm[im1inm[:,:,0]>0]
            im2in[im2inm[:,:,0]>0]  = im2inm[im2inm[:,:,0]>0]
            flowin[im1inm[:,:,0]>0]  = flowinm[im1inm[:,:,0]>0]
            dcchangein[im1inm[:, :, 0] > 0] = dcchangeinm[im1inm[:, :, 0] > 0]
            validin[im1inm[:,:,0]>0]  = validinm[im1inm[:,:,0]>0]

        '''
        plt.imshow(im1in)
        plt.show()
        plt.imshow(im2in)
        plt.show()
        plt.imshow(flowin[:,:,0])
        plt.show()
        plt.imshow(dcchangein[:,:,0])
        plt.show()
        plt.imshow(validin)
        plt.show()
        '''
        return im1in,im2in,flowin,dcchangein,validin
    def __getitem__(self, index):
        self.kit = self.kit +1
        index = index % len(self.image_list)
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])


        ssim, d2d, aphla = frame_utils.readNerfMask(self.maskall[index])
        valid_mask = frame_utils.read_gen(self.occmask[index])
        valid_mask = np.array(valid_mask).astype(np.float32) / 255
        dmask = np.array(d2d < 0.1).astype(np.float32)
        amask = np.array(aphla > 0.7).astype(np.float32)
        smask = np.array(ssim < 0.1).astype(np.float32)
        valid = dmask * amask * smask*valid_mask

        d1, d2, dc_change = frame_utils.readNerfddc(self.NERFddc[index])
        #ao = frame_utils.readAO(self.AOmask[index])
        dc_change = np.concatenate((dc_change[:, :, np.newaxis], valid[:, :, np.newaxis]), axis=2)


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])


        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)

        occshow = np.concatenate(
            [valid_mask[:, :, np.newaxis], valid_mask[:, :, np.newaxis], valid_mask[:, :, np.newaxis]], axis=-1)
        finalshow = np.concatenate(
            [valid[:, :, np.newaxis], valid[:, :, np.newaxis], valid[:, :, np.newaxis]], axis=-1)

        '''
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()

        plt.imshow(flow2rgb(flow))
        plt.show()
        plt.imshow(finalshow)
        plt.show()
        plt.imshow(occshow)
        plt.show()
        logmid = dc_change[:, :, 0]
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid)
        datamax = np.max(logmid)
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        plt.imshow(heatmap)
        plt.show()
        '''

        if self.augmentor is not None:
            img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
            img1, img2, flow, dc_change, valid = self.addfore(img1, img2, flow, dc_change, valid)
        flowadd = np.zeros_like(flow)
        for i in range(1):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            if flag>1 :
                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                flowadd[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1

        self.last_image = img2
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, dc_change, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.maskall = v*self.maskall
        self.occmask = v*self.occmask
        self.AOmask = v * self.AOmask
        self.NERFddc = v * self.NERFddc
        return self

    def __len__(self):
        return len(self.image_list)
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.driving = False
        self.sintel = False
        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.dispnet =[]

        self.instance = []
        self.sematic = []

        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0
        self.k = 1
        self.kr = 0
        self.get_depth = 0
        self.kitti_test = 0
        self.sintel_test = 0
        self.fflow_list =[]
        self.fmask_DPS =[]
        self.fmask_PPF =[]
        self.fGSddc =[]
        self.image_listf = []
        self.fimage12 = []
        self.ifocc = 1
        self.last_image = np.random.randn(320,720,3)
    def resize_sparse_exp(self, exp, valid, fx=1.0, fy=1.0):
        ht, wd = exp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        exp = exp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        exp0 = exp[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        exp1 = exp0

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        exp1 = exp1[v]

        exp_img = np.zeros([ht1, wd1], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        exp_img[yy, xx] = exp1
        valid_img[yy, xx] = 1

        return exp_img, valid_img
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img
    def addfore(self,im1in,im2in,flowin,dcchangein,validin):
        #TODO 添加随机前景

        indexf = np.random.randint(0, self.image_listf.__len__())

        d1, d2, dc_change = frame_utils.readNerfddc(self.fGSddc[indexf])
        flow, valid = frame_utils.readFlowKITTI(self.fflow_list[indexf])
        pds, rdds, ssim = frame_utils.readNerfMask(self.fmask_DPS[indexf])
        pah1, pah2, occall = frame_utils.readNerfMask(self.fmask_PPF[indexf])
        img01f = frame_utils.read_gen(self.image_listf[indexf][0])
        img02f = frame_utils.read_gen(self.image_listf[indexf][1])
        img01f = np.array(img01f).astype(np.uint8)
        img02f = np.array(img02f).astype(np.uint8)

        pmask = np.array(pds < 2).astype(np.float32)
        amask = np.array(pah1 < 0.1).astype(np.float32)
        valid_mask = occall<0.02
        dmask = np.array(rdds < 0.01).astype(np.float32)
        smask = np.array(ssim >0.9).astype(np.float32)
        valid_mask = np.array(valid_mask).astype(np.float32)

        forevalid1 = img01f[:,:,0]>0
        forevalid2  = img02f[:, :, 0] > 0
        forevalid = forevalid1 + forevalid2
        #试一下没有amask的
        validf = dmask *valid_mask*pmask*amask

        coordsf = np.where(forevalid > 0)
        minydf = coordsf[0].min()
        maxydf = coordsf[0].max()
        minxdf = coordsf[1].min()
        maxxdf = coordsf[1].max()

        img01f = img01f[minydf:maxydf, minxdf:maxxdf,:]
        img02f = img02f[minydf:maxydf, minxdf:maxxdf,:]
        validf = validf[minydf:maxydf, minxdf:maxxdf]
        flowf =  flow[minydf:maxydf, minxdf:maxxdf,:]
        dcf = dc_change[minydf:maxydf, minxdf:maxxdf]
        '''
        plt.imshow(img01f)
        plt.show()
        plt.imshow(img02f)
        plt.show()

        validf[(img01f<1)[:,:,0]] = 0
        plt.imshow(validf)
        plt.show()
        flows = np.zeros_like(flow2rgb(flowf))
        flows[(validf>0),:] = flow2rgb(flowf)[(validf>0),:]
        plt.imshow(flows)
        plt.show()
        '''
        #随机前景增强，大小缩放，随机位置叠加，随机放射变化
        scale = 2 ** np.random.uniform(-0.4, 0.4)#!!!!!!!!!!!!!!!!!!0.4
        img1 = cv2.resize(img01f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img02f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        flow, valid = self.resize_sparse_flow_map(flowf, validf, fx=scale, fy=scale)
        dc_out, _ = self.resize_sparse_exp(dcf,validf, fx=scale, fy=scale)
        if np.random.rand() < 0.5: # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            valid = valid[:,::-1]
            dc_out = dc_out[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
        if np.random.rand() < 0.1: # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]
            dc_out = dc_out[::-1, :]
            valid = valid[::-1, :]

        if (im1in.shape[0] - img1.shape[0])>0 and (im1in.shape[1] - img1.shape[1])>0:
            y0 = np.random.randint(0, im1in.shape[0] - img1.shape[0])
            x0 = np.random.randint(0, im1in.shape[1] - img1.shape[1])
            im1inm = np.zeros_like(im1in)
            im2inm = np.zeros_like(im2in)
            flowinm = np.zeros_like(flowin)
            dcchangeinm = np.zeros_like(dcchangein)
            validinm = np.zeros_like(validin)

            im1inm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img1
            im2inm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img2
            flowinm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = flow
            dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1],0] = dc_out
            dcchangeinm[y0:y0 + img1.shape[0], x0:x0 + img1.shape[1], 1] = valid
            validinm[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = valid


            im1in[im1inm[:,:,0]>0] = im1inm[im1inm[:,:,0]>0]
            im2in[im2inm[:,:,0]>0]  = im2inm[im2inm[:,:,0]>0]
            flowin[im1inm[:,:,0]>0]  = flowinm[im1inm[:,:,0]>0]
            dcchangein[im1inm[:, :, 0] > 0] = dcchangeinm[im1inm[:, :, 0] > 0]
            validin[im1inm[:,:,0]>0]  = validinm[im1inm[:,:,0]>0]

        '''
        plt.imshow(im1in)
        plt.show()
        plt.imshow(im2in)
        plt.show()
        plt.imshow(flowin[:,:,0])
        plt.show()
        plt.imshow(dcchangein[:,:,0])
        plt.show()
        plt.imshow(validin)
        plt.show()
        '''
        return im1in,im2in,flowin,dcchangein,validin
    def __getitem__(self, index):
        self.kit = self.kit +1
        if self.test_scene:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            dispnet = np.abs(disparity_loader(self.dispnet[index]))
            return img1, img2, self.extra_info[index],dispnet
        if self.is_test and not self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,valid, self.extra_info[index]  # 这个mask是是否有噪音块的掩膜
        if self.sintel_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,0, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)


        if self.driving or self.sintel:
            flow, valid = frame_utils.readFlowdriving(self.flow_list[index])
        else:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])


        d1,d2,occmask = self.get_dc(index)#这个mask是遮挡掩膜
        dc_change = d2/d1
        dcmask = (dc_change < 1.5) * (dc_change > 0.5)*valid#这个掩膜被包含于光流的掩膜
        dc_change[dcmask==0] = 0#这个东西是不可能等于0


        if self.occlusion:#在设计Sintel的时候要看一下,这一部分是剔除Sintel遮挡
            dcc = dc_change
            dcc = abs(cv2.filter2D(dcc,-1,kernel=self.kernel2))
            maskd = torch.from_numpy(dcc>0.1).bool()
            dcmask = dcmask* ((maskd==0).numpy())*occmask

            dc_change[dcmask==0] = 0
            #再加一个遮挡
            dc_change = np.concatenate((dc_change[:,:,np.newaxis],dcmask[:,:,np.newaxis]),axis =2 )
        else:
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], dcmask[:, :, np.newaxis]), axis=2)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)


        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]


        if self.augmentor is not None:#光流和深度变化率是两个独立的掩膜
             img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
             #img1, img2, flow, dc_change, valid = self.addfore(img1, img2, flow, dc_change, valid)

        for i in range(int(self.k)*2):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            kit1 = (imgb1 != 0).sum()
            kit2 = (imgb2 != 0).sum()
            flag2 = abs(kit2-kit1)/(kit2+kit1+1)
            if flag>1 and flag2<0.5:

                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1

        #总体的遮挡掩膜，首先要包含光流掩膜和深度变化率掩膜
        '''        
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow[:,:,0])
        plt.show()
        plt.imshow(flow[:, :, 1])
        plt.show()
        plt.imshow(dc_change[:, :, 0])
        plt.show()
        plt.imshow(dc_change[:, :, 1])
        plt.show()
        '''

        self.last_image = img2

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        conf = valid.clone()
        return img1, img2, flow, dc_change, valid.float(),conf.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.occ_list = v * self.occ_list
        self.instance = v * self.instance
        self.sematic = v * self.sematic
        self.calib = v * self.calib
        return self

    def __len__(self):
        return len(self.image_list)
class HotFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.driving = False
        self.sintel = False
        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.dispnet =[]

        self.instance = []
        self.sematic = []

        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = []
        self.bezier = mybezier()
        self.rect = retangle()
        self.kit = 0
        self.k = 1
        self.kr = 0
        self.get_depth = 0
        self.kitti_test = 0
        self.sintel_test = 0
        self.fflow_list =[]
        self.fmask_DPS =[]
        self.fmask_PPF =[]
        self.fGSddc =[]
        self.image_listf = []
        self.fimage12 = []
        self.ifocc = 1
        self.last_image = np.random.randn(320,720,3)

    def __getitem__(self, index):
        self.kit = self.kit +1
        if self.test_scene:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            dispnet = np.abs(disparity_loader(self.dispnet[index]))
            return img1, img2, self.extra_info[index],dispnet
        if self.is_test and not self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,valid, self.extra_info[index]  # 这个mask是是否有噪音块的掩膜
        if self.sintel_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,0, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)


        if self.driving or self.sintel:
            flow, valid = frame_utils.readFlowdriving(self.flow_list[index])
        else:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])


        d1,d2,occmask = self.get_dc(index)#这个mask是遮挡掩膜
        dc_change = d2/d1
        dcmask = (dc_change < 1.5) * (dc_change > 0.5)*valid#这个掩膜被包含于光流的掩膜
        dc_change[dcmask==0] = 0#这个东西是不可能等于0


        if self.occlusion:#在设计Sintel的时候要看一下,这一部分是剔除Sintel遮挡
            dcc = dc_change
            dcc = abs(cv2.filter2D(dcc,-1,kernel=self.kernel2))
            maskd = torch.from_numpy(dcc>0.1).bool()
            dcmask = dcmask* ((maskd==0).numpy())*occmask

            dc_change[dcmask==0] = 0
            #再加一个遮挡
            dc_change = np.concatenate((dc_change[:,:,np.newaxis],dcmask[:,:,np.newaxis]),axis =2 )
        else:
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], dcmask[:, :, np.newaxis]), axis=2)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        alpha = 1  # 对比度增益，>1 增强，<1 减弱
        beta = 0  # 亮度偏移，增加亮度可调高
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = 255 - img1
        img1 = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)
        img1 = cv2.bilateralFilter(img1, d=7, sigmaColor=75, sigmaSpace=75)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = 255 - img2
        img2 = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
        img2 = cv2.bilateralFilter(img2, d=7, sigmaColor=75, sigmaSpace=75)
        dc_change = np.array(dc_change).astype(np.float32)


        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]


        if self.augmentor is not None:#光流和深度变化率是两个独立的掩膜
             img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
        for i in range(int(self.k)*2):
            imgb1,imgb2,ansb,flag = self.bezier.get_mask(img1,self.last_image)
            kit1 = (imgb1 != 0).sum()
            kit2 = (imgb2 != 0).sum()
            flag2 = abs(kit2-kit1)/(kit2+kit1+1)
            if flag>1 and flag2<0.5:

                img1[imgb1 != 0] = imgb1[imgb1 != 0]
                img2[imgb2 != 0] = imgb2[imgb2 != 0]
                flow[imgb1[:, :, 0] != 0, :] = ansb[imgb1[:, :, 0] != 0, :2]
                if self.sparse:
                    valid[imgb1[:,:,0]!=0] = 1
                dc_change[imgb1[:,:,0] != 0, 0:1] = ansb[imgb1[:,:,0] != 0, 2:]
                li = ansb[:,:,2]!=0
                dc_change[li,1]=1

        #总体的遮挡掩膜，首先要包含光流掩膜和深度变化率掩膜
        '''        
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow[:,:,0])
        plt.show()
        plt.imshow(flow[:, :, 1])
        plt.show()
        plt.imshow(dc_change[:, :, 0])
        plt.show()
        plt.imshow(dc_change[:, :, 1])
        plt.show()
        '''

        self.last_image = img2

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        conf = valid.clone()
        return img1, img2, flow, dc_change, valid.float(),conf.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.occ_list = v * self.occ_list
        self.instance = v * self.instance
        self.sematic = v * self.sematic
        self.calib = v * self.calib
        return self

    def __len__(self):
        return len(self.image_list)
class MpiSinteltest(FlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/media/lh/extradata/scene_flow/sintel', dstype='clean'):
        super(MpiSinteltest, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion = True
        self.sintel_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                if split != 'test':
                    self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if self.occlusion:
                    for i in range(len(image_list) - 1):
                        self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,align_corners=True,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)
        return depth1.numpy(),depth2.numpy(),1-occ.numpy()
    def depth_to_disp(self,Z, bl=1, fl=1000):
        disp = bl * fl / Z
        return disp
class FhotMpiSintel(HotFlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/home/lh/sence_flowu/sintel', dstype='clean',ifeval = 't_all'):
        super(FhotMpiSintel, self).__init__(aug_params, sparse=True)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.calib = []
        self.occlusion = True
        self.sintel = True
        if split == 'test':
            self.is_test = True
        self.kernel = np.ones([3, 3], np.float32)
        self.kernel2 = np.ones([3, 3], np.float32)*-1
        self.kernel2[1,1] = 8
        if ifeval=='t_test':
            for scene in os.listdir(image_root):
                if scene in ['temple_2','market_6','cave_4','ambush_5','bamboo_1']:
                    image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                    depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
                    occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                    for i in range(len(image_list) - 1):
                        self.image_list += [[image_list[i], image_list[i + 1]]]
                        if split != 'test':
                            self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                        self.extra_info += [(scene, i)]  # scene and frame_id

                    if split != 'test':
                        self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                        if self.occlusion:
                            for i in range(len(image_list) - 1):
                                self.occ_list += [occ_list[i]]
        elif ifeval=='t_train':
            for scene in os.listdir(image_root):
                if scene not in ['temple_2','market_6','cave_4','ambush_5','bamboo_1']:
                    image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                    depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
                    occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                    for i in range(len(image_list) - 1):
                        self.image_list += [[image_list[i], image_list[i + 1]]]
                        if split != 'test':
                            self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                        self.extra_info += [(scene, i)]  # scene and frame_id

                    if split != 'test':
                        self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                        if self.occlusion:
                            for i in range(len(image_list) - 1):
                                self.occ_list += [occ_list[i]]
        else:
            for scene in os.listdir(image_root):
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
                occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[i], image_list[i + 1]]]
                    if split != 'test':
                        self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                if split != 'test':
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                    if self.occlusion:
                        for i in range(len(image_list) - 1):
                            self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()
            #膨胀occ
            '''
            acc = occ.numpy().astype(np.uint8)
            occ = cv2.filter2D(acc,-1,kernel=self.kernel)
            occ = torch.from_numpy(occ>0).bool()
            '''
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,align_corners=True,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)

        return depth1.numpy(),depth2.numpy() ,1-occ.numpy()
class MpiSintel(FlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/home/lh/sence_flowu/sintel', dstype='clean',ifeval = 't_all'):
        super(MpiSintel, self).__init__(aug_params, sparse=True)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.calib = []
        self.occlusion = True
        self.sintel = True
        if split == 'test':
            self.is_test = True
        self.kernel = np.ones([3, 3], np.float32)
        self.kernel2 = np.ones([3, 3], np.float32)*-1
        self.kernel2[1,1] = 8
        if ifeval=='t_test':
            for scene in os.listdir(image_root):
                if scene in ['temple_2','market_6','cave_4','ambush_5','bamboo_1']:
                    image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                    depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
                    occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                    for i in range(len(image_list) - 1):
                        self.image_list += [[image_list[i], image_list[i + 1]]]
                        if split != 'test':
                            self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                        self.extra_info += [(scene, i)]  # scene and frame_id

                    if split != 'test':
                        self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                        if self.occlusion:
                            for i in range(len(image_list) - 1):
                                self.occ_list += [occ_list[i]]
        elif ifeval=='t_train':
            for scene in os.listdir(image_root):
                if scene not in ['temple_2','market_6','cave_4','ambush_5','bamboo_1']:
                    image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                    depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
                    occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                    for i in range(len(image_list) - 1):
                        self.image_list += [[image_list[i], image_list[i + 1]]]
                        if split != 'test':
                            self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                        self.extra_info += [(scene, i)]  # scene and frame_id

                    if split != 'test':
                        self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                        if self.occlusion:
                            for i in range(len(image_list) - 1):
                                self.occ_list += [occ_list[i]]
        else:
            for scene in os.listdir(image_root):
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
                occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[i], image_list[i + 1]]]
                    if split != 'test':
                        self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                if split != 'test':
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                    if self.occlusion:
                        for i in range(len(image_list) - 1):
                            self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()
            #膨胀occ
            '''
            acc = occ.numpy().astype(np.uint8)
            occ = cv2.filter2D(acc,-1,kernel=self.kernel)
            occ = torch.from_numpy(occ>0).bool()
            '''
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,align_corners=True,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)

        return depth1.numpy(),depth2.numpy() ,1-occ.numpy()
class FlyingChairs2(FlowDataset):
    def __init__(self, aug_params=None, root='/home/lh/all_datasets/FlyingChairs2/train'):
        super(FlyingChairs2, self).__init__(aug_params,sparse = True)
        self.rand = False
        self.occlusion = False
        self.driving = True
        images1 = sorted(glob(osp.join(root, '*img_0.png')))
        images2 = sorted(glob(osp.join(root, '*img_1.png')))
        flows = sorted(glob(osp.join(root, '*01.flo')))
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'

        for i in range(len(flows)):
                self.flow_list += [flows[i]]
                self.image_list += [[images1[i], images2[i]]]
        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]

    def get_dc(self,index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        imu = np.ones_like(img1)
        return imu[:,:,0],imu[:,:,0],imu[:,:,0]
class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/media/lh/extradata/scene_flow/FlyingChairs/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'
        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]
        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/media/lh/extradata/scene_flow/flyingthings/flyingthings/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params, sparse=True)
        exclude = np.loadtxt('/home/lh/CSCV_occ/exclude.txt', delimiter=' ', dtype=np.unicode_)
        exclude = set(exclude)
        self.occlusion = False
        self.driving = True
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'
        for cam in ['left','right']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                d0_dirs = sorted(glob(osp.join(root, 'disparity/TRAIN/*/*')))
                d0_dirs = sorted([osp.join(f, cam) for f in d0_dirs])

                dc_dirs = sorted(glob(osp.join(root, 'disparity_change/TRAIN/*/*')))
                dc_dirs = sorted([osp.join(f, direction, cam) for f in dc_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir,d0dir,dcdir in zip(image_dirs, flow_dirs,d0_dirs,dc_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    d0s = sorted(glob(osp.join(d0dir, '*.pfm')))
                    dcs = sorted(glob(osp.join(dcdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        tag = '/'.join(images[i].split('/')[-5:])
                        if tag in exclude:
                            print("Excluding %s" % tag)
                            continue
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                            self.depth_list += [[d0s[i], dcs[i]]]
                            frame_id = images[i].split('/')[-1]
                            self.extra_info += [[frame_id]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]
                            self.depth_list += [[d0s[i+1], dcs[i+1]]]
                            frame_id = images[i+1].split('/')[-1]
                            self.extra_info += [[frame_id]]
        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]
    def triangulation(self, disp, bl=1):#kitti flow 2015

        fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask


class KITTI(FlowDataset):#/home/lh/RAFT_master/dataset/kitti_scene   '/home/xuxian/RAFT3D/datasets'
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/kitti',get_depth=0):
        super(KITTI, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.calib = []
        if split == 'testing':
            self.is_test = True
        if split == 'submit':
            self.is_test = True
        if split =='test':
            self.test_scene = True
        self.occlusion = False
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        instance = []
        semantic = []
        if split == 'training':
            root = osp.join(root, split)
            images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

            instanceo = sorted(glob(osp.join(root, 'instance/*_10.png')))
            semantico = sorted(glob(osp.join(root, 'semantic/*_10.png')))

            for j in range(images2o.__len__()):
                if j%5>0 or self.get_depth:
                    images1.append(images1o[j])
                    images2.append(images2o[j])
                    disp1.append(disp1o[j])
                    disp2.append(disp2o[j])

                    instance.append(instanceo[j])
                    semantic.append(semantico[j])
        elif split=='testing':
            root = osp.join(root, 'training')
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='submit':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='test':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
            disp1LEA = sorted(glob(osp.join(root, 'disp_ganet_testing/*_10.png')))
            self.dispnet = disp1LEA
        else:
            images1 = sorted(glob(osp.join(root, '*.jpg')))
            images2 = images1[1:]
            images1.pop()
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]
        if split == 'training':
            flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for j in range(flowo.__len__()):
                if j%5>0 or self.get_depth:
                    flow.append(flowo[j])
        elif split == 'testing':
            flow = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.flow_list = flow
        self.instance = instance
        self.sematic = semantic


    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask
class HotKITTI(HotFlowDataset):#/home/lh/RAFT_master/dataset/kitti_scene   '/home/xuxian/RAFT3D/datasets'
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/kitti',get_depth=0):
        super(HotKITTI, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.calib = []
        if split == 'testing':
            self.is_test = True
        if split == 'submit':
            self.is_test = True
        if split =='test':
            self.test_scene = True
        self.occlusion = False
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        instance = []
        semantic = []
        if split == 'training':
            root = osp.join(root, split)
            images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

            instanceo = sorted(glob(osp.join(root, 'instance/*_10.png')))
            semantico = sorted(glob(osp.join(root, 'semantic/*_10.png')))

            for j in range(images2o.__len__()):
                if j%5>0 or self.get_depth:
                    images1.append(images1o[j])
                    images2.append(images2o[j])
                    disp1.append(disp1o[j])
                    disp2.append(disp2o[j])

                    instance.append(instanceo[j])
                    semantic.append(semantico[j])
        elif split=='testing':
            root = osp.join(root, 'training')
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='submit':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='test':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
            disp1LEA = sorted(glob(osp.join(root, 'disp_ganet_testing/*_10.png')))
            self.dispnet = disp1LEA
        else:
            images1 = sorted(glob(osp.join(root, '*.jpg')))
            images2 = images1[1:]
            images1.pop()
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]
        if split == 'training':
            flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for j in range(flowo.__len__()):
                if j%5>0 or self.get_depth:
                    flow.append(flowo[j])
        elif split == 'testing':
            flow = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.flow_list = flow
        self.instance = instance
        self.sematic = semantic


    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask
class KITTI12(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/xuxian/RAFT3D/kitti_data',rand =True):
        super(KITTI12, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True
        self.rand = False
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'colored_0/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'colored_0/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class KITTI_test(FlowDataset):#/home/lh/RAFT3D-DEPTH/data_train_test /home/xuxian/RAFT3D/data_train_test#/new_data/kitti_data/datasets/training
    def  __init__(self, aug_params=None, split='kitti_test', root='/home/lh/newdata/scene_flow/kitti/training',get_depth=0):
        super(KITTI_test, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        if split == 'kitti_test':
           self.kitti_test = 1
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]

        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        for j in range(images2o.__len__()):
            images1.append(images1o[j])
            images2.append(images2o[j])
            disp1.append(disp1o[j])
            disp2.append(disp2o[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]

        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
                flow.append(flowo[j])

        self.flow_list = flow

    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask
class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='/home/lh/RAFT-master/dataset/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1

class Nerfdata(NerfDataset):#/home/lh/RAFT_master/dataset/kitti_scene   '/home/xuxian/RAFT3D/datasets'
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/NeRF156',get_depth=0):
        super(Nerfdata, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        self.isnerf = True
        rootf = '/media/lh/sundata/lh/MIPGS10K_flow_foremin/'
        images1 = sorted(glob(osp.join(root, '*/image1/*')))
        images2 = sorted(glob(osp.join(root, '*/image2/*')))
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        self.maskall = sorted(glob(osp.join(root, '*/maskall/*')))
        self.occmask = sorted(glob(osp.join(root, '*/occ_mask/*')))
        self.AOmask = sorted(glob(osp.join(root, '*/ao/*')))
        self.NERFddc = sorted(glob(osp.join(root, '*/depth/*')))
        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, '*/flow/*')))
        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]
#这个是加载3DGS数据的
class GSdata(GSDataset):
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/MIPGS10K_flow/',get_depth=0):
        super(GSdata, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        self.isnerf = True
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'

        for id in range(0,38):
            strid = str(id).zfill(4)
            rootuse = root + strid
            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
        for id in [46,47,48,49,50,53,55,56,64,65,67,69,70,71,72,73,74,75,76,77,79,83,87,88,90,91,92,94,95,160,161,162]:
            strid = str(id).zfill(4)
            rootuse = root + strid
            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))

        for id in [275,276,277,280]:
            strid = str(id).zfill(4)
            rootuse = root + strid


            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
        #加载KITTI里程计数据集的数据

        for id in range(0,12):
            strid = str(id).zfill(2)
            rootuse = root + 'k'+strid
            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]
#这个是加载3DGS数据的
class GSdataall(GSDataset):
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/MIPGS10K_flow/',get_depth=0):
        super(GSdataall, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        self.isnerf = True
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'

        for id in range(0,284):
            strid = str(id).zfill(4)
            rootuse = root + strid
            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
        
        for id in range(0,0):
            strid = str(id).zfill(2)
            rootuse = root + 'k'+strid
            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))

        # TODO 下面开始加载随机运动前景
        for id in range(624):#624
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]


# 这个是加载3DGS数据的
class HotGSdataall(HotGSDataset):
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/RED_flow/', get_depth=0):
        super(HotGSdataall, self).__init__(aug_params, sparse=True)
        self.get_depth = get_depth
        self.occlusion = False
        self.isnerf = True
        rootf = '/home/lh/all_datasets/RED_fore/'
        for folder in os.listdir(root):
            rootuse = root + folder
            images1 = sorted(glob(osp.join(rootuse, 'image1/*')))
            images2 = sorted(glob(osp.join(rootuse, 'image2/*')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.mask_DPS += sorted(glob(osp.join(rootuse, 'mask_DPS/*')))
            self.mask_PPF += sorted(glob(osp.join(rootuse, 'mask_PPF/*')))
            self.GSddc += sorted(glob(osp.join(rootuse, 'depth/*')))
            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))

        # TODO 下面开始加载随机运动前景
        for foldero in os.listdir(root):
            rootusef = rootf + foldero
            imagef1_list = sorted(glob(os.path.join(rootusef, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootusef, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]



#加载从深度中学习光流的数据集
class Anything3D_dataset(AnythingDataset):
    def __init__(self, aug_params=None, root='/home/lh/all_datasets/Anything3DCOCOtraining/'):
        super(Anything3D_dataset, self).__init__(aug_params, sparse=True)


        rootuse = root
        images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
        images2 = sorted(glob(osp.join(rootuse, 'RGB/*_2.png')))
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
        self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))



#加载从深度中学习光流的数据集
class FFD_dataset(FlowTODepthDataset):
    def __init__(self, aug_params=None, root='/media/lh/sundata/lh/DDFKITTI_v666p925/'):
        super(FFD_dataset, self).__init__(aug_params, sparse=True)
        dirictor = os.listdir(root)
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'
        for id in dirictor:
            rootuse = root + id
            images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
            images2 = sorted(glob(osp.join(rootuse, 'RGB/*_1.png')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
            self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))
        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]

class FFD_datasetab(FlowTODepthDataset):
    def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIpicMVtrainingveval2025/'):
        super(FFD_datasetab, self).__init__(aug_params, sparse=True)
        dirictor = os.listdir(root)
        rootf = '/home/lh/all_datasets/MIPGS10K_flow_foremin/'
        for id in dirictor:
            rootuse = root + id
            images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
            images2 = sorted(glob(osp.join(rootuse, 'RGB/*_1.png')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
            self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))

        # TODO 下面开始加载随机运动前景
        for id in range(624):
            strid = str(id).zfill(4)
            rootuse = rootf + strid
            imagef1_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_1.png')))
            imagef2_list = sorted(glob(os.path.join(rootuse, 'fmaskall/*/*_2.png')))

            for idx, (img1, img2) in enumerate(zip(imagef1_list, imagef2_list)):
                ilist = img1.split('/')
                plist = ilist[-1].split('_')[-2]
                self.image_listf += [[img1, img2, float(plist)]]

        # 重新排序，整体列表
        #self.image_listf.sort(key=lambda x: x[-1])
        for im12 in self.image_listf:
            rendernum = im12[0].split('/')[-4]
            imnum = im12[0].split('/')[-2]
            flowdir = rootf + rendernum + '/' + 'flow_fore' + '/' + imnum + '.png'
            maskdirDPS = rootf + rendernum + '/' + 'mask_DPS_fore' + '/' + imnum + '.png'
            maskdirPPF = rootf + rendernum + '/' + 'mask_PPF_fore' + '/' + imnum + '.png'
            maskdirddc = rootf + rendernum + '/' + 'depth_fore' + '/' + imnum + '.png'
            image1f = rootf + rendernum + '/' + 'image1_fore' + '/' + imnum + '.png'
            image2f = rootf + rendernum + '/' + 'image2_fore' + '/' + imnum + '.png'
            self.fflow_list += [flowdir]
            self.fmask_DPS += [maskdirDPS]
            self.fmask_PPF += [maskdirPPF]
            self.fGSddc += [maskdirddc]
            self.fimage12 += [[image1f,image2f]]
class FFD_datasetabt(FlowTODepthDataset):
    def __init__(self, aug_params=None, root='/home/lh/all_datasets/realworld_gogo/'):
    #def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIMVtesting_ab_all_kittiv2/'):
    #def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIMVtesting_ab_bg/'):
        super(FFD_datasetabt, self).__init__(aug_params, sparse=True)
        dirictor = os.listdir(root)
        for id in dirictor:
            rootuse = root + id
            images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
            images2 = sorted(glob(osp.join(rootuse, 'RGB/*_1.png')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
            self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))
class FFD_datasetabt2(FlowTODepthDataset):
    #def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIMVtestingveval/'):
    def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIMVtesting_ab_all_vits/'):
    #def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIMVtesting_ab_all_kittiv2/'):
    #def __init__(self, aug_params=None, root='/home/lh/all_datasets/KITTIMVtesting_ab_bg/'):
        super(FFD_datasetabt2, self).__init__(aug_params, sparse=True)
        dirictor = os.listdir(root)
        for id in dirictor:
            rootuse = root + id
            images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
            images2 = sorted(glob(osp.join(rootuse, 'RGB/*_1.png')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
            self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))
#加载从深度中学习光流的数据集
class FFDSintel_dataset(FlowTODepthDataset):
    def __init__(self, aug_params=None, root='/media/lh/extradata/FFDSintel_v0/'):
        super(FFDSintel_dataset, self).__init__(aug_params, sparse=True)
        dirictor = os.listdir(root)
        for id in dirictor:
            rootuse = root + id
            images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
            images2 = sorted(glob(osp.join(rootuse, 'RGB/*_1.png')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
            self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))
        root = '/media/lh/extradata/FFDSintel_v0F/'
        dirictor = os.listdir(root)
        for id in dirictor:
            rootuse = root + id
            images1 = sorted(glob(osp.join(rootuse, 'RGB/*_0.png')))
            images2 = sorted(glob(osp.join(rootuse, 'RGB/*_1.png')))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [[frame_id]]
                self.image_list += [[img1, img2]]

            self.flow_list += sorted(glob(osp.join(rootuse, 'flow/*')))
            self.Scale_list += sorted(glob(osp.join(rootuse, 'Scale/*')))
class Driving(FlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/home/lh/sence_flowu/driving'):
        super(Driving, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.occlusion = False
        self.driving = True
        level_stars = '/*' * 6
        candidate_pool = glob('%s/optical_flow%s' % (root, level_stars))
        for flow_path in sorted(candidate_pool):
            idd = flow_path.split('/')[-1].split('_')[-2]
            if 'into_future' in flow_path:
                idd_p1 = '%04d' % (int(idd) + 1)
            else:
                idd_p1 = '%04d' % (int(idd) - 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','disparity')
                d0_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd)
                d1_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd_p1)

                dc_path = flow_path.replace('optical_flow', 'disparity_change')
                dc_path = '%s/%s.pfm' % (dc_path.rsplit('/', 1)[0], idd)
                im_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','frames_cleanpass')
                im0_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd)
                im1_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd_p1)
                frame_id = im1_path.split('/')[-1]
                self.extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                self.flow_list += [flow_path]
                self.image_list += [[im0_path,im1_path]]
                self.depth_list += [[d0_path,dc_path,d1_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]
    def triangulation(self, disp,index, bl=1):#kitti flow 2015
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):#能不能顺便来个OCC
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        d2t = np.abs(disparity_loader(self.depth_list[index][2]))
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        uv = coords_grid(d1.shape[1], d1.shape[0])
        flowU = torch.from_numpy(flow)
        d2u = torch.from_numpy(d2t)
        uvn = uv+flowU

        d2uo = bilinear_sampler(d2u, uvn)
        d2uos = d2uo[0,0].detach().numpy()
        derror= np.abs(d2uos-d2)/(d2uos+d2)
        occ = derror>0.1
        #mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)
        return self.triangulation(d1,index),self.triangulation(d2,index),1-occ
class MonkeyD(FlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/media/lh/extradata/scene_flow/Monkey'):
        super(MonkeyD, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.occlusion = False
        self.driving = True
        level_stars = '/*' * 4
        candidate_pool = glob('%s/optical_flow%s' % (root, level_stars))
        for flow_path in sorted(candidate_pool):
            idd = flow_path.split('/')[-1].split('_')[-2]
            if 'into_future' in flow_path:
                idd_p1 = '%04d' % (int(idd) + 1)
            else:
                idd_p1 = '%04d' % (int(idd) - 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','disparity')
                d0_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd)
                d1_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd_p1)

                dc_path = flow_path.replace('optical_flow', 'disparity_change')
                dc_path = '%s/%s.pfm' % (dc_path.rsplit('/', 1)[0], idd)
                im_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','frames_cleanpass')
                im0_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd)
                im1_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd_p1)
                frame_id = im1_path.split('/')[-1]
                self.extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                self.flow_list += [flow_path]
                self.image_list += [[im0_path,im1_path]]
                self.depth_list += [[d0_path,dc_path,d1_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]
    def triangulation(self, disp,index, bl=1):#kitti flow 2015
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):#能不能顺便来个OCC
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        d2t = np.abs(disparity_loader(self.depth_list[index][2]))
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        uv = coords_grid(d1.shape[1], d1.shape[0])
        flowU = torch.from_numpy(flow)
        d2u = torch.from_numpy(d2t)
        uvn = uv+flowU

        d2uo = bilinear_sampler(d2u, uvn)
        d2uos = d2uo[0,0].detach().numpy()
        derror= np.abs(d2uos-d2)/(d2uos+d2)
        occ = derror>0.1
        #mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)
        return self.triangulation(d1,index),self.triangulation(d2,index),1-occ
class Nerfdatakitti(NerfDataset):
    def __init__(self, aug_params=None, split='training', root='/home/lh/all_datasets/kitti_flow',get_depth=0):
        super(Nerfdatakitti, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        self.isnerf = True

        images1 = sorted(glob(osp.join(root, '*/image1/*')))
        images2 = sorted(glob(osp.join(root, '*/image2/*')))
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        self.maskall = sorted(glob(osp.join(root, '*/maskall/*')))
        self.occmask = sorted(glob(osp.join(root, '*/occ_mask/*')))
        self.NERFddc = sorted(glob(osp.join(root, '*/depth/*')))
        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, '*/flow/*')))


def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs2(aug_params)

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        #sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        kitti = KITTI(aug_params, split='training')
        #final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        driving = Driving(aug_params, split='training')
        Monkey = MonkeyD(aug_params, split='training')
        train_dataset = clean_dataset+driving+10*sintel_clean+200*kitti+Monkey

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        driving = Driving(aug_params, split='training')
        Monkey = MonkeyD(aug_params, split='training')
        #gsdata = GSdataall(aug_params)
        train_dataset = 100*sintel_final+100*sintel_clean+Monkey+driving+things
    elif args.stage == 'realworld':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        driving = Driving(aug_params, split='training')
        Monkey = MonkeyD(aug_params, split='training')
        gsdata = GSdataall(aug_params)
        kitti = KITTI(aug_params, split='training')
        train_dataset = gsdata+100*kitti+10*sintel_clean+10*sintel_final+driving+Monkey+things
    elif args.stage == 'DRU':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        #Monkey = MonkeyD(aug_params, split='training')
        train_dataset = driving
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        #kitti = KITTI(aug_params, split='training')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final',ifeval = 't_train')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean',ifeval = 't_train')
        #FFD = FFD_datasetab(aug_params)
        #FFDraw = FFD_dataset(aug_params)
        #FFD = FFD_datasetabt(aug_params)
        #FFD2 = FFD_datasetabt2(aug_params)
        #FFDs = FFDSintel_dataset(aug_params)
        #FFD = FFDSintel_dataset(aug_params)
        #gsdata = GSdata(aug_params)
        driving  = Driving(aug_params, split='training')
        train_dataset =   driving+5*sintel_clean+5*sintel_final
    elif args.stage == 'driving':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.3, 'do_flip': True}
        #things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        #sintel_final = MpiSintel(aug_params, split='training', dstype='final',ifeval = 't_train')
        #sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', ifeval= 't_train')
        #sintel_clean = FhotMpiSintel(aug_params, split='training', dstype='clean',ifeval = 't_train')
        #Monkey = MonkeyD(aug_params, split='training')
        #kittic        = KITTIC(aug_params, split='training')
        #kitti = HotKITTI(aug_params, split='training')
        #kittistereo = KITTI_STEREO(aug_params, split='training')
        #kittistereoT = KITTI_STEREO_test(aug_params, split='training')
        #nerfdata = Nerfdata(aug_params)
        gsdata = HotGSdataall(aug_params)
        #nerfdatak = Nerfdatakitti(aug_params)
        #driving      = Driving(aug_params, split='training')
        ##Anything3DK =  Anything3D_dataset(aug_params)
        #kitti = KITTI(aug_params, split='training')
        train_dataset = gsdata
        #things+5*sintel_final+5*sintel_clean+driving+Monkey+100*kitti
    elif args.stage == 'hotflow':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.3, 'do_flip': True}
        Hotgsdata = HotGSdataall(aug_params)
        kitti = HotKITTI(aug_params, split='training')
        train_dataset = Hotgsdata+1*kitti
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=6, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

