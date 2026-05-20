#这个数据集专门用来优化3D流
#todo 主要需要涉及的数据集应该有KITTI,Spring,FlyingThings3D,vKITTI,Driving,Monkey
#先把这个Spring和Vkitti数据集的数据搞出来，vKITTI严格来说应该和kitti差不多，
#输出应该是包含im1,im2,d1,d2,delta_d1,flow,valid
#这个输出的是深度的倒数，也就是所谓的视差，研究一下RFAT3D是怎么搞的，现在这个x,y,z版本不知道为啥速度很慢，还是要再尝试一下分离的效果
from core.utils.utils import  coords_grid
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
import Dataset_home.flowio as flow_IO
import  cv2
from Dataset_home.augment3d import SceneFlowAugmentor,SceneFlowAugmentorSparse

def decode_array_from_uint16s(highs, lows):
  # 将两个 uint16 数组合并为一个 24 位的数
  combined = (highs.astype(np.uint32) << 16) | lows.astype(np.uint32)

  # 恢复浮动数
  restored = combined.astype(np.float32)

  return restored
def readDepth(filename):
    depth3 = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    depth = decode_array_from_uint16s(depth3[:,:,0].astype(np.float32),depth3[:,:,1].astype(np.float32))
    depth = depth / 1024.0
    return depth

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def get_grid_np(B,H,W):
    meshgrid_base = np.meshgrid(range(0, W), range(0, H))[::-1]
    basey = np.reshape(meshgrid_base[0], [1, 1, 1, H, W])
    basex = np.reshape(meshgrid_base[1], [1, 1, 1, H, W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).float()
    return grid.view( H, W, 2)

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
from scipy.ndimage import  binary_dilation
def get_neighbor_consistency_invalid_mask(
    arr,
    diff_thresh=0.5,
    ksize=5,
    min_support=10,
    dilate_iter=0,
):
    """
    基于邻域一致性检测异常点，返回 invalid mask

    Args:
        arr: 2D numpy array
        diff_thresh: 与中心点差值小于该阈值，认为是“相近邻居”
        ksize: 邻域窗口大小，通常 3 或 5，必须为奇数
        min_support: 至少需要多少个相近邻居，否则判为异常
                     对 3x3 窗口，建议 2~3
                     对 5x5 窗口，建议 4~6
        dilate_iter: invalid mask 膨胀次数，0 表示不膨胀

    Returns:
        invalid_mask: bool array, True 表示无效
    """
    arr = np.asarray(arr, dtype=np.float32)
    assert arr.ndim == 2, "Only support 2D array"
    assert ksize % 2 == 1, "ksize must be odd"

    H, W = arr.shape
    r = ksize // 2

    # 0 / nan / inf 视为已有无效
    invalid_mask = (arr == 0) | (~np.isfinite(arr))

    # padding
    arr_pad = np.pad(arr, ((r, r), (r, r)), mode='edge')
    invalid_pad = np.pad(invalid_mask, ((r, r), (r, r)), mode='constant', constant_values=True)

    support_count = np.zeros((H, W), dtype=np.int32)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue

            neigh = arr_pad[r + dy:r + dy + H, r + dx:r + dx + W]
            neigh_invalid = invalid_pad[r + dy:r + dy + H, r + dx:r + dx + W]

            # 只统计有效邻居
            close = (~neigh_invalid) & (np.abs(neigh - arr) < diff_thresh)
            support_count += close.astype(np.int32)

    # 当前点本身若有效，但周围支持它的相近邻居太少 -> 异常
    isolated_bad = (~invalid_mask) & (support_count < min_support)

    invalid_mask = invalid_mask | isolated_bad

    if dilate_iter > 0:
        invalid_mask = binary_dilation(
            invalid_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=dilate_iter
        )

    return invalid_mask
class SceneFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if sparse:
            self.augmentor = SceneFlowAugmentorSparse(**aug_params)
        else:
            self.augmentor = SceneFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list =[]
        self.spring = False
        self.driving = False
        self.things = False
        self.vkitti = False
        self.kitti = False
        self.monkey = False
        self.sintel = False
        self.tartanair = False
        self.last_image = np.random.randn(320,720,3)
    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        d1, d2, dd1, intrinsics = self.get_dc(index)

        if self.driving or self.monkey:
            flow,_ = frame_utils.readFlowdriving(self.flow_list[index])#第三个维度为有效Valid
            valid = None
        elif self.spring:
            flow = flow_IO.readFlowFile(self.flow_list[index])[::2,::2,:]
            nanm1 = np.isnan(d1)
            nanm2 = np.isnan(d2)
            nanm3 = np.isnan(dd1)
            nanm4 = np.isnan(flow[:,:,0])
            nanm5 = np.isnan(flow[:,:,1])
            valid = (nanm1+nanm2+nanm3+nanm4+nanm5)<1
            #去掉所有NAN的点
            d1[valid==0]=0
            d2[valid==0]=0
            dd1[valid==0]=0
        elif self.vkitti:
            flow,valid = frame_utils.read_vkitti_png_flow(self.flow_list[index])
        elif self.things:
            flow = frame_utils.read_gen(self.flow_list[index])
            valid = None
        elif self.sintel:
            flow = frame_utils.read_gen(self.flow_list[index])
            valid = None
        elif self.kitti:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        elif self.tartanair:
            flow, _ = frame_utils.read_t_flow(self.flow_list[index])
            valid = None


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        d1 = np.array(d1).astype(np.float32)
        d2 = np.array(d2).astype(np.float32)
        dd1 = np.array(dd1).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.sparse:#光流和深度变化率是两个独立的掩膜
            img1, img2, flow,d1,d2,dd1,intrinsics,valid= self.augmentor(img1, img2, flow,d1,d2,dd1,intrinsics,valid)
        else:
            img1, img2, flow, d1, d2, dd1, intrinsics = self.augmentor(img1, img2, flow, d1, d2, dd1, intrinsics)

        '''
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow2rgb(flow))
        plt.show()
        plt.imshow(d1)
        plt.show()
        plt.imshow(dd1)
        plt.show()
        dz = dd1-d1
        dz[dz>2]=0
        plt.imshow(dz)
        plt.show()
        '''
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()


        intrinsics = torch.from_numpy(intrinsics).float()
        d1 = torch.from_numpy(d1[np.newaxis,:,:]).float()
        d2 = torch.from_numpy(d2[np.newaxis,:,:]).float()
        dz = (torch.from_numpy(dd1[np.newaxis,:,:])-d1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).float()
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, d1, d2, dz,intrinsics,valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.occ_list = v * self.occ_list
        self.calib = v * self.calib
        return self
    def __len__(self):
        return len(self.image_list)

class FlyingThings3D(SceneFlowDataset):
    def __init__(self, aug_params=None, root='/media/lh/lh4t/flow3d_data/flyingthings/flyingthings/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params, sparse=False)
        exclude = np.loadtxt('/home/lh/CSCV_occ/exclude.txt', delimiter=' ', dtype=np.unicode_)
        exclude = set(exclude)
        self.things = True
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
                            self.depth_list += [[d0s[i],d0s[i+1], dcs[i]]]
                            frame_id = images[i].split('/')[-1]
                            self.extra_info += [[frame_id]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]
                            self.depth_list += [[d0s[i+1],d0s[i], dcs[i+1]]]
                            frame_id = images[i+1].split('/')[-1]
                            self.extra_info += [[frame_id]]

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1]))
        d3 = np.abs(disparity_loader(self.depth_list[index][2])+d1)
        intrinsics = np.array([1050, 1050, 479.5, 269.5])

        s = 0.5 + np.random.rand()
        d1 = s*d1
        d2 = s*d2
        d3 = s*d3

        return d1,d2,d3,intrinsics
def stem(path):
    return osp.splitext(osp.basename(path))[0]
def make_dict(file_list):
    return {stem(f): f for f in file_list}
class FlyingThings3D_subset(SceneFlowDataset):
    def __init__(
        self,
        aug_params=None,
        root='/mnt/hdd/hanling/sceneflow_data/FlyingThings3D_subset/FlyingThings3D_subset',
        split='train',
        image_dirname='image_clean',
        flow_dirname='flow',
        disparity_dirname='disparity',
        disparity_change_dirname='disparity_change',
        exclude_file=None,
    ):
        super(FlyingThings3D_subset, self).__init__(aug_params, sparse=False)

        exclude = set()
        if exclude_file is not None:
            exclude = np.loadtxt(exclude_file, delimiter=' ', dtype=np.unicode_)
            if np.ndim(exclude) == 0:
                exclude = [str(exclude)]
            exclude = set(exclude)

        self.things = True

        for cam in ['left', 'right']:
            for direction in ['into_future', 'into_past']:
                image_dir = osp.join(root, split, image_dirname, cam)
                d0_dir    = osp.join(root, split, disparity_dirname, cam)
                dc_dir    = osp.join(root, split, disparity_change_dirname, cam, direction)
                flow_dir  = osp.join(root, split, flow_dirname, cam, direction)

                if not (osp.isdir(image_dir) and osp.isdir(d0_dir) and osp.isdir(dc_dir) and osp.isdir(flow_dir)):
                    print(f"[Skip] missing path: cam={cam}, direction={direction}")
                    continue

                images = sorted(glob(osp.join(image_dir, '*')))
                flows  = sorted(glob(osp.join(flow_dir, '*')))
                d0s    = sorted(glob(osp.join(d0_dir, '*')))
                dcs    = sorted(glob(osp.join(dc_dir, '*')))

                image_map = make_dict(images)
                flow_map  = make_dict(flows)
                d0_map    = make_dict(d0s)
                dc_map    = make_dict(dcs)

                print(f"[{cam}][{direction}] #img={len(images)} #flow={len(flows)} #disp={len(d0s)} #dc={len(dcs)}")
                print(f"  image head: {list(sorted(image_map.keys()))[:3]}")
                print(f"  flow  head: {list(sorted(flow_map.keys()))[:3]}")
                print(f"  dc    head: {list(sorted(dc_map.keys()))[:3]}")

                # 以 flow/dc 的交集 id 为锚点
                valid_ids = sorted(set(flow_map.keys()) & set(dc_map.keys()))

                added = 0
                skipped = 0

                for fid in valid_ids:
                    try:
                        fid_int = int(fid)
                    except ValueError:
                        skipped += 1
                        continue

                    if direction == 'into_future':
                        nxt = f"{fid_int + 1:07d}"

                        # 需要 t 和 t+1 都存在
                        if fid not in image_map or nxt not in image_map:
                            skipped += 1
                            continue
                        if fid not in d0_map or nxt not in d0_map:
                            skipped += 1
                            continue

                        tag = '/'.join(image_map[fid].split('/')[-3:])
                        if tag in exclude:
                            print(f"Excluding {tag}")
                            continue

                        self.image_list += [[image_map[fid], image_map[nxt]]]
                        self.flow_list += [flow_map[fid]]
                        self.depth_list += [[d0_map[fid], d0_map[nxt], dc_map[fid]]]
                        self.extra_info += [[osp.basename(image_map[fid])]]
                        added += 1

                    elif direction == 'into_past':
                        prv = f"{fid_int - 1:07d}"

                        # 需要 t 和 t-1 都存在
                        if fid not in image_map or prv not in image_map:
                            skipped += 1
                            continue
                        if fid not in d0_map or prv not in d0_map:
                            skipped += 1
                            continue

                        tag = '/'.join(image_map[fid].split('/')[-3:])
                        if tag in exclude:
                            print(f"Excluding {tag}")
                            continue

                        self.image_list += [[image_map[fid], image_map[prv]]]
                        self.flow_list += [flow_map[fid]]
                        self.depth_list += [[d0_map[fid], d0_map[prv], dc_map[fid]]]
                        self.extra_info += [[osp.basename(image_map[fid])]]
                        added += 1

                print(f"[{cam}][{direction}] added={added}, skipped={skipped}")
    def get_dc(self, index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1]))
        d3 = np.abs(disparity_loader(self.depth_list[index][2]) + d1)
        intrinsics = np.array([1050, 1050, 479.5, 269.5])

        s = 0.5 + np.random.rand()
        d1 = s * d1
        d2 = s * d2
        d3 = s * d3

        return d1, d2, d3, intrinsics
class MonkeyD(SceneFlowDataset):
    def __init__(self, aug_params=None,root='/media/lh/lh4t/flow3d_data/Monkey'):
        super(MonkeyD, self).__init__(aug_params, sparse=False)
        self.calib = []
        self.monkey = True
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
                self.depth_list += [[d0_path,d1_path,dc_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]

    def get_dc(self,index):#获取第一帧深度，第二帧深度，还有第一帧的变化深度,再来个内参
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1]))
        d3 = np.abs(disparity_loader(self.depth_list[index][2])+d1)
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        H,W = d1.shape
        fx, fy, cx, cy = (fl, fl, W/2.0, H/2.0)
        intrinsics = np.array([fx, fy, cx, cy])
        s = 0.5 +  np.random.rand()
        d1 = s*d1
        d2 = s*d2
        d3 = s*d3
        return d1,d2,d3,intrinsics
class Driving(SceneFlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/mnt/hdd/hanling/sceneflow_data/Driving'):
        super(Driving, self).__init__(aug_params, sparse=False)
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
                self.depth_list += [[d0_path,d1_path,dc_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]

    def get_dc(self,index):#获取第一帧深度，第二帧深度，还有第一帧的变化深度,再来个内参
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1]))
        d3 = np.abs(disparity_loader(self.depth_list[index][2])+d1)

        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050

        H,W = d1.shape
        fx, fy, cx, cy = (fl, fl, W/2.0, H/2.0)
        intrinsics = np.array([fx, fy, cx, cy])
        s = 0.1 + 0.4 * np.random.rand()
        d1 = s*d1
        d2 = s*d2
        d3 = s*d3


        return d1,d2,d3,intrinsics

class TartanAirV2(SceneFlowDataset):
    def  __init__(
        self,
        aug_params=None,
        root='/mnt/hdd/hanling/sceneflow_data/tartanair_v2',
        difficulties=('Data_easy', 'Data_hard'),
        trajectory_glob='P*',
        image_folder='image_lcam_front',
        depth_folder='depth_lcam_front',
        flow_folder='flow_lcam_front',
        seg_folder='seg_lcam_front',
        use_seg=False,
    ):
        super(TartanAirV2, self).__init__(aug_params, sparse=False)

        self.root = root
        self.difficulties = difficulties
        self.trajectory_glob = trajectory_glob
        self.use_seg = use_seg
        self.seg_list = []
        self.tartanair = True

        # 常用内参；如果你后面有每个场景/相机自己的标定，再替换这里
        self.default_intrinsics = np.array([320.0, 320.0, 320.0, 320.0], dtype=np.float32)

        # 支持的扩展名
        self.img_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        self.depth_exts = ('*.npy', '*.png')
        self.flow_exts = ('*.npy', '*.npz', '*.png')

        scenes = sorted([d for d in os.listdir(root) if osp.isdir(osp.join(root, d))])

        for scene in scenes:
            scene_root = osp.join(root, scene)

            for difficulty in difficulties:
                diff_root = osp.join(scene_root, difficulty)
                if not osp.isdir(diff_root):
                    continue

                traj_dirs = sorted(glob(osp.join(diff_root, trajectory_glob)))
                traj_dirs = [d for d in traj_dirs if osp.isdir(d)]

                for traj_dir in traj_dirs:
                    image_dir = osp.join(traj_dir, image_folder)
                    depth_dir = osp.join(traj_dir, depth_folder)
                    flow_dir  = osp.join(traj_dir, flow_folder)
                    seg_dir   = osp.join(traj_dir, seg_folder)

                    if not (osp.isdir(image_dir) and osp.isdir(depth_dir) and osp.isdir(flow_dir)):
                        print(f"[Skip] missing folder in {traj_dir}")
                        continue

                    image_list = self._glob_multi(image_dir, self.img_exts)
                    depth_list = self._glob_multi(depth_dir, self.depth_exts)
                    flow_list  = self._glob_multi(flow_dir, self.flow_exts)
                    seg_list   = self._glob_multi(seg_dir, ('*.png',)) if self.use_seg and osp.isdir(seg_dir) else []

                    # 典型假设：flow 对应 i -> i+1
                    n = min(len(image_list) - 1, len(depth_list) - 1, len(flow_list))

                    for i in range(n):
                        self.image_list += [[image_list[i], image_list[i + 1]]]
                        self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                        self.flow_list += [flow_list[i]]

                        if self.use_seg and len(seg_list) >= i + 2:
                            self.seg_list += [[seg_list[i], seg_list[i + 1]]]

                        self.extra_info += [[scene, difficulty, osp.basename(traj_dir), i]]
                    print(self.extra_info[-1])
        print(f"[TartanAirV2] total samples: {len(self.image_list)}")

    def _glob_multi(self, folder, patterns):
        files = []
        for p in patterns:
            files.extend(glob(osp.join(folder, p)))
        return sorted(files)

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_depth(self, depthpath):
        try:
            if not osp.exists(depthpath):
                raise FileNotFoundError(f"depth file not found: {depthpath}")

            ext = osp.splitext(depthpath)[1].lower()

            if ext == ".npy":
                depth = np.load(depthpath)
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                if depth.ndim != 2:
                    raise ValueError(f"bad npy depth shape: {depth.shape}, path={depthpath}")

            elif ext == ".png":
                depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
                if depth_rgba is None:
                    raise ValueError(f"cv2.imread failed, path={depthpath}")

                if depth_rgba.ndim != 3 or depth_rgba.shape[-1] != 4:
                    raise ValueError(
                        f"bad png depth shape: {depth_rgba.shape}, expect HxWx4, path={depthpath}"
                    )

                depth = depth_rgba.view("<f4")
                depth = np.squeeze(depth, axis=-1)

            else:
                raise ValueError(f"unsupported depth ext: {ext}, path={depthpath}")

            if depth.size == 0:
                raise ValueError(f"empty depth array, path={depthpath}")

            if not np.isfinite(depth).any():
                raise ValueError(f"all depth values are non-finite, path={depthpath}")

            return depth.astype(np.float32, copy=False)

        except Exception as e:
            print(f"[DEPTH-ERROR] {depthpath}\n    -> {type(e).__name__}: {e}", flush=True)
            raise

    def _read_flow(self,flowpath):
        flow16 = cv2.imread(flowpath, cv2.IMREAD_UNCHANGED)
        flow32 = flow16[:, :, :2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0

        mask8 = flow16[:, :, 2].astype(np.uint8)
        return flow32, mask8

    def get_dc(self, index):
        flow, valid = self._read_flow(self.flow_list[index])  # H,W,2
        depth1 = self._read_depth(self.depth_list[index][0])  # H,W
        depth2 = self._read_depth(self.depth_list[index][1])  # H,W

        h, w = depth1.shape[:2]

        depth2_t = torch.from_numpy(depth2).float().view(1, 1, h, w)
        flow_t = torch.from_numpy(flow).float()

        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )

        # frame1像素按forward flow找到frame2位置
        x2 = xx + flow_t[..., 0]
        y2 = yy + flow_t[..., 1]

        grid_x = 2.0 * x2 / max(w - 1, 1) - 1.0
        grid_y = 2.0 * y2 / max(h - 1, 1) - 1.0
        grid_norm = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # 从frame2采样目标深度
        depth3 = F.grid_sample(
            depth2_t, grid_norm, align_corners=True, mode='nearest'
        )[0, 0].numpy()

        intrinsics = self.default_intrinsics.copy()

        bf = 80.0
        disp1 = bf / np.clip(depth1, 1e-6, None)
        disp2 = bf / np.clip(depth2, 1e-6, None)
        disp3 = bf / np.clip(depth3, 1e-6, None)

        x2_np = x2.numpy()
        y2_np = y2.numpy()

        out_of_bound = (x2_np < 0) | (x2_np > w - 1) | (y2_np < 0) | (y2_np > h - 1)

        # 投到最近整数像素，做简化z-buffer
        xi = np.rint(x2_np).astype(np.int32)
        yi = np.rint(y2_np).astype(np.int32)

        in_view = (valid > 0) & (~out_of_bound)

        # 每个目标像素只保留 depth3 最小者
        zbuf = np.full((h, w), 100000, dtype=np.float32)
        occ_mask = np.zeros((h, w), dtype=bool)

        ys, xs = np.where(in_view)

        # 第一遍：记录目标像素最小depth3
        for y, x in zip(ys, xs):
            tx, ty = xi[y, x], yi[y, x]
            z = depth3[y, x]
            if z < zbuf[ty, tx]:
                zbuf[ty, tx] = z

        # 第二遍：不是最前面的都标invalid
        eps = 1e-3
        for y, x in zip(ys, xs):
            tx, ty = xi[y, x], yi[y, x]
            z = depth3[y, x]
            if z > zbuf[ty, tx] + eps:
                occ_mask[y, x] = True

        dz = disp3-disp1
        maskud = get_neighbor_consistency_invalid_mask(dz)
        invalid = (
                (valid > 0)
                | out_of_bound
                | occ_mask
                | maskud
                | np.isnan(disp3)
                | np.isinf(disp3)
                | (disp3 > 1e4)
        )
        disp3[invalid] = 1e4

        return disp1, disp2, disp3, intrinsics
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def quick_check_tartanair(dataset, max_samples=None, num_workers=8, log_file="bad_files.txt", print_every=5000):
    """
    直接遍历读取 dataset.image_list / depth_list / flow_list，检查是否可读

    参数:
        dataset: 你的 TartanAirV2 数据集对象
        max_samples: 只检查前多少个 sample，None 表示全量
        num_workers: 并行线程数
        log_file: 坏文件日志输出路径
        print_every: 每检查多少个文件打印一次进度
    """
    n = len(dataset.flow_list)
    if max_samples is not None:
        n = min(n, max_samples)

    total = n * 5  # 2 image + 2 depth + 1 flow

    print(f"[INFO] start checking")
    print(f"[INFO] samples      : {n}")
    print(f"[INFO] total files  : {total}")
    print(f"[INFO] num_workers  : {num_workers}")
    print(f"[INFO] log_file     : {log_file}", flush=True)

    def task_iter():
        for i in range(n):
            yield ("image", i, 0, dataset.image_list[i][0])
            yield ("image", i, 1, dataset.image_list[i][1])
            yield ("depth", i, 0, dataset.depth_list[i][0])
            yield ("depth", i, 1, dataset.depth_list[i][1])
            yield ("flow",  i, -1, dataset.flow_list[i])

    def check_one(task):
        kind, i, j, path = task
        try:
            if not os.path.exists(path):
                return kind, i, j, path, False, "not exists"

            if os.path.getsize(path) == 0:
                return kind, i, j, path, False, "empty file"

            if kind == "image":
                x = dataset._read_image(path)
                if x is None:
                    return kind, i, j, path, False, "read_image returned None"

            elif kind == "depth":
                x = dataset._read_depth(path)
                if x is None:
                    return kind, i, j, path, False, "read_depth returned None"

            elif kind == "flow":
                flow, valid = dataset._read_flow(path)
                if flow is None or valid is None:
                    return kind, i, j, path, False, "read_flow returned None"

            return kind, i, j, path, True, "ok"

        except Exception as e:
            return kind, i, j, path, False, f"{type(e).__name__}: {e}"

    summary = {
        "image": {"ok": 0, "bad": 0},
        "depth": {"ok": 0, "bad": 0},
        "flow":  {"ok": 0, "bad": 0},
    }

    checked = 0

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("kind\tsample_idx\tlocal_idx\tpath\terror\n")

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for kind, i, j, path, ok, msg in tqdm(
                ex.map(check_one, task_iter()),
                total=total,
                desc="Checking files"
            ):
                checked += 1

                if ok:
                    summary[kind]["ok"] += 1
                else:
                    summary[kind]["bad"] += 1
                    f.write(f"{kind}\t{i}\t{j}\t{path}\t{msg}\n")
                    print(f"\n[BAD] kind={kind} sample={i} idx={j} path={path}\n      -> {msg}", flush=True)

                if checked % print_every == 0:
                    print(
                        f"\n[PROGRESS] checked={checked}/{total} | "
                        f"image bad={summary['image']['bad']} | "
                        f"depth bad={summary['depth']['bad']} | "
                        f"flow bad={summary['flow']['bad']}",
                        flush=True
                    )

    print("\n===== Summary =====")
    for k in ["image", "depth", "flow"]:
        print(f"{k}: ok={summary[k]['ok']}, bad={summary[k]['bad']}")
    print(f"bad file log saved to: {log_file}", flush=True)

    return summary

class MpiSintel(SceneFlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/mnt/hdd/hanling/sceneflow_data/Sintel', dstype='clean',ifeval = 't_all'):
        super(MpiSintel, self).__init__(aug_params, sparse=False)
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
        depth3 = (torch.nn.functional.grid_sample(depth2, frepb,align_corners=True,mode='nearest').view(h, w))
        depth3 = depth3.view(h, w)
        H, W = depth1.shape
        fx, fy, cx, cy = (1050, 1050, W/2.0, H/2.0)
        intrinsics = np.array([fx, fy, cx, cy])
        vaild = 1-occ.numpy()
        disp1 = 360/depth1.numpy()
        disp2 = 360/depth2[0,0].numpy()
        disp3 = 360/(depth3+0.00000001).numpy()
        inf_mask = np.isinf(disp3)+ disp3>1000
        disp3[vaild<1 + inf_mask] = 10000
        return disp1,disp2,disp3,intrinsics
class Spring(SceneFlowDataset):
    def __init__(self, aug_params=None,  split='train',root='/media/lh/lh4t/flow3d_data/spring/spring'):
        super(Spring, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.occlusion = False
        self.spring = True
        seq_root = os.path.join(root, split)
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # forward
                for frame in range(1, len(images)):
                    self.data_list.append((frame, scene, cam, "FW"))
                # backward
                for frame in reversed(range(2, len(images)+1)):
                    self.data_list.append((frame, scene, cam, "BW"))

        for frame_data in self.data_list:
            frame, scene, cam, direction = frame_data
            if direction == "FW":
                othertimestep = frame + 1
            else:
                othertimestep = frame - 1
            img0_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
            img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{othertimestep:04d}.png")
            cam_path = os.path.join(self.seq_root, scene, 'cam_data', 'intrinsics.txt')
            cam_data = np.loadtxt(cam_path)
            disp1_path = os.path.join(self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
            disp2_path = os.path.join(self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{othertimestep:04d}.dsp5")
            disp3_path = os.path.join(self.seq_root, scene, f"disp2_{direction}_{cam}",f"disp2_{direction}_{cam}_{frame:04d}.dsp5")
            flow_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}",f"flow_{direction}_{cam}_{frame:04d}.flo5")
            self.extra_info += [[frame]]
            self.flow_list += [flow_path]
            self.image_list += [[img0_path, img1_path]]
            self.depth_list += [[disp1_path, disp2_path, disp3_path]]
            self.calib += [cam_data[frame-1]]


    def get_dc(self,index):#获取第一帧深度，第二帧深度，还有第一帧的变化深度,再来个内参
        #这里的标签都是4k的建议降采样一倍
        d1 = np.abs(flow_IO.readDispFile(self.depth_list[index][0]))[::2, ::2]
        d2 = np.abs(flow_IO.readDispFile(self.depth_list[index][1]))[::2, ::2]
        d3 = np.abs(flow_IO.readDispFile(self.depth_list[index][2]))[::2, ::2]

        fx, fy, cx, cy = self.calib[index]
        intrinsics = np.array([fx, fy, cx, cy])
        s = 0.4 + 0.2 * np.random.rand()
        d1 = s*d1
        d2 = s*d2
        d3 = s*d3


        return d1,d2,d3,intrinsics
class vkitti(SceneFlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/home/lh/all_datasets/vkitti'):
        super(vkitti, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.occlusion = False
        self.vkitti = True
        candidate_all = glob('%s/Scene*/*/*/*wardFlow/*/*.png' % (root))
        extra_info = []
        flow_list = []
        image_list = []
        depth_list = []
        calib_l = []

        for flow_path in sorted(candidate_all):
            idd = flow_path.split('/')[-1].split('_')[-1].split('.')[-2]
            if 'forward' in flow_path:
                idd_p1 = '%05d' % (int(idd) + 1)
            else:
                idd_p1 = '%05d' % (int(idd) - 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('backwardFlow', 'depth').replace('forwardFlow', 'depth')
                d0_path = '%s/depth_%s.png' % (d0_path.rsplit('/', 1)[0], idd)
                d1_path = '%s/depth_%s.png' % (d0_path.rsplit('/', 1)[0], idd_p1)

                Scene_path = flow_path.replace('Flow', 'SceneFlow').replace('flow', 'sceneFlow')

                im_path = flow_path.replace('backwardFlow', 'rgb').replace('forwardFlow', 'rgb')
                im0_path = '%s/rgb_%s.jpg' % (im_path.rsplit('/', 1)[0], idd)
                im1_path = '%s/rgb_%s.jpg' % (im_path.rsplit('/', 1)[0], idd_p1)
                frame_id = im1_path.split('/')[-1]
                extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                flow_list += [flow_path]
                image_list += [[im0_path,im1_path]]
                depth_list += [[d0_path,d1_path,Scene_path]]

                calib_list = np.loadtxt(im0_path.split("frames")[0]+'intrinsic.txt', skiprows=1)
                calib_l += [calib_list[int(idd)][2:]]


        for idx in range(len(image_list)):
            if idx % 5 != 0:
                self.extra_info.append(extra_info[idx])
                self.flow_list.append(flow_list[idx])
                self.image_list.append(image_list[idx])
                self.depth_list.append(depth_list[idx])
                self.calib.append(calib_l[idx])


    def get_dc(self,index):#获取第一帧深度，第二帧深度，还有第一帧的变化深度,再来个内参
        d1 = cv2.imread(self.depth_list[index][0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#以厘米为单位的深度
        d2 = cv2.imread(self.depth_list[index][1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#以厘米为单位的深度
        d3 = (cv2.imread(self.depth_list[index][2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]*2.0/65535.0 -1.0)*1000.0+d1#以厘米为单位的深度变化量

        not_use1 = (d1>20000).copy()
        not_use2 = (d2 > 20000).copy()
        fx, fy, cx, cy = self.calib[index]
        intrinsics = np.array([fx, fy, cx, cy])
        fl = intrinsics[0]
        s = 40 + 40 * np.random.rand()
        d1 = s*fl / d1
        d2 = s*fl / d2
        d3 = s*fl / d3
        #超过200m范围外的就不考虑了
        d1[not_use1] = 0.0001
        d2[not_use2] = 0.0001
        d3[not_use1] = 10000

        return d1,d2,d3,intrinsics
class KITTI(SceneFlowDataset):#/home/lh/RAFT_master/dataset/kitti_scene   '/home/xuxian/RAFT3D/datasets'
    def __init__(self, aug_params=None, split='training', root='/mnt/hdd/hanling/sceneflow_data'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.kitti = True
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        dispp1 = []
        dispp2 = []
        flow =[]

        root = osp.join(root, split)
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

        disp1pre = sorted(glob(osp.join(root, 'disp_monster_training/*_10.png')))
        disp2pre = sorted(glob(osp.join(root, 'disp_monster_training/*_11.png')))

        for j in range(images2o.__len__()):
                images1.append(images1o[j])
                images2.append(images2o[j])
                disp1.append(disp1o[j])
                disp2.append(disp2o[j])
                dispp1.append(disp1pre[j])
                dispp2.append(disp2pre[j])

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2,dispps1,dispps2 in zip(disp1, disp2,dispp1,dispp2):
            self.depth_list += [[disps1, disps2,dispps1,dispps2]]

        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
                flow.append(flowo[j])
        self.flow_list = flow


    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1s = disparity_loader(self.depth_list[index][0])
        d2s = disparity_loader(self.depth_list[index][1])

        d1p = readDepth(self.depth_list[index][2])
        d2p = readDepth(self.depth_list[index][3])
        d3 = d1p+(d2s-d1s)

        s = 0.5 + np.random.rand()


        H, W = d1s.shape
        intrinsics = np.array([721.5377, 721.5377, H/2.0,  W/2.0])

        return s*d1p,s*d2p,s*d3,intrinsics
class KITTIab(SceneFlowDataset):#/home/lh/RAFT_master/dataset/kitti_scene   '/home/xuxian/RAFT3D/datasets'
    def __init__(self, aug_params=None, split='training', root='/mnt/hdd/hanling/sceneflow_data'):
        super(KITTIab, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.kitti = True
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        dispp1 = []
        dispp2 = []
        flow =[]

        root = osp.join(root, split)
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

        disp1pre = sorted(glob(osp.join(root, 'disp_monster_training/*_10.png')))
        disp2pre = sorted(glob(osp.join(root, 'disp_monster_training/*_11.png')))

        for j in range(images2o.__len__()):
            if j % 2 > 0:
                images1.append(images1o[j])
                images2.append(images2o[j])
                disp1.append(disp1o[j])
                disp2.append(disp2o[j])
                dispp1.append(disp1pre[j])
                dispp2.append(disp2pre[j])

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2,dispps1,dispps2 in zip(disp1, disp2,dispp1,dispp2):
            self.depth_list += [[disps1, disps2,dispps1,dispps2]]

        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
            if j % 2 > 0:
                flow.append(flowo[j])
        self.flow_list = flow


    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1s = disparity_loader(self.depth_list[index][0])
        d2s = disparity_loader(self.depth_list[index][1])

        d1p = readDepth(self.depth_list[index][2])
        d2p = readDepth(self.depth_list[index][3])
        d3 = d1p+(d2s-d1s)

        s = 0.5 + np.random.rand()


        H, W = d1s.shape
        intrinsics = np.array([721.5377, 721.5377, H/2.0,  W/2.0])

        return s*d1p,s*d2p,s*d3,intrinsics
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'pretrain':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        thingsc = FlyingThings3D(aug_params,dstype='frames_cleanpass')
        thingsf = FlyingThings3D(aug_params, dstype='frames_finalpass')
        sintelc = MpiSintel(aug_params,dstype='clean')
        sintelf = MpiSintel(aug_params, dstype='final')
        monkey = MonkeyD(aug_params)
        kitti = KITTI(aug_params, split='training')
        vkittiu = vkitti(aug_params, split='training')#67k
        #spring = Spring(aug_params, split='train')#59w
        train_dataset =  monkey+thingsf+thingsc+200*kitti+2*driving+6*sintelc+6*sintelf+vkittiu
    elif args.stage == 'driving':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        kitti = KITTI(aug_params, split='training')
        vkittiu = vkitti(aug_params, split='training')
        train_dataset = vkittiu+200*kitti+driving

    elif args.stage == 'abtrain':#这个是用于消融实验的数据准备，em就用driving和一部分的Sintel吧
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        #Tartan = TartanAirV2(aug_params)
        thingsc = FlyingThings3D_subset(aug_params)


        train_dataset = thingsc
    elif args.stage == 'fineting':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        kitti = KITTI(aug_params, split='training')
        vkittiu = vkitti(aug_params, split='training')
        train_dataset = vkittiu+200*kitti+driving
    elif args.stage == 'Sintel_CamLi':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        thingsc = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        thingsf = FlyingThings3D(aug_params, dstype='frames_finalpass')
        #sintelc = MpiSintel(aug_params,dstype='clean')
        #sintelf = MpiSintel(aug_params, dstype='final')
        train_dataset = thingsc+thingsf
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
class Sintel_test(data.Dataset):
    def __init__(self, aug_params=None, split='training', root='/mnt/hdd/hanling/sceneflow_data/Sintel', dstype='clean',ifeval = 't_all'):

        self.extra_info= []
        self.flow_list = []
        self.image_list = []
        self.depth_list = []
        self.occ_list = []
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
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
                        for i in range(len(image_list) - 1):
                            self.occ_list += [occ_list[i]]
        if ifeval=='t_show':
            for scene in os.listdir(image_root):
                if scene in ['temple_2','market_6','cave_4','ambush_5']:
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
                    for i in range(len(image_list) - 1):
                            self.occ_list += [occ_list[i]]
    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()
            #膨胀occ
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
        depth3 = (torch.nn.functional.grid_sample(depth2, frepb,align_corners=True,mode='nearest').view(h, w))
        depth3 = depth3.view(h, w)
        H, W = depth1.shape
        fx, fy, cx, cy = (1050, 1050, W/2.0, H/2.0)
        intrinsics = np.array([fx, fy, cx, cy])
        vaild = 1-occ.numpy()
        disp1 = 360/depth1.numpy()
        disp2 = 360/depth2[0,0].numpy()
        disp3 = 360/(depth3+0.00000001).numpy()
        inf_mask = np.isinf(disp3)+ disp3>1000
        disp3[vaild<1 + inf_mask] = 10000
        return disp1,disp2,disp3,intrinsics
    def __len__(self):
        return len(self.extra_info)

    def __getitem__(self, index):
        index = index % len(self.image_list)
        d1, d2, dd1, intrinsics = self.get_dc(index)

        flow = frame_utils.read_gen(self.flow_list[index])
        valid = None
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        d1 = np.array(d1).astype(np.float32)
        d2 = np.array(d2).astype(np.float32)
        dd1 = np.array(dd1).astype(np.float32)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        intrinsics = torch.from_numpy(intrinsics).float()
        d1 = torch.from_numpy(d1[np.newaxis,:,:]).float()
        d2 = torch.from_numpy(d2[np.newaxis,:,:]).float()
        dz = (torch.from_numpy(dd1[np.newaxis,:,:])-d1).float()
        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, d1, d2, dz,intrinsics,valid
class vkitti_test(data.Dataset):
    def __init__(self,root='/home/lh/all_datasets/vkitti'):

        candidate_all = glob('%s/Scene*/*/*/*wardFlow/*/*.png' % (root))
        extra_info = []
        flow_list = []
        image_list = []
        depth_list = []
        calib_l = []
        self.extra_info= []
        self.flow_list = []
        self.image_list = []
        self.depth_list = []
        self.calib = []

        for flow_path in sorted(candidate_all):
            idd = flow_path.split('/')[-1].split('_')[-1].split('.')[-2]
            if 'forward' in flow_path:
                idd_p1 = '%05d' % (int(idd) + 1)
            else:
                idd_p1 = '%05d' % (int(idd) - 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('backwardFlow', 'depth').replace('forwardFlow', 'depth')
                d0_path = '%s/depth_%s.png' % (d0_path.rsplit('/', 1)[0], idd)
                d1_path = '%s/depth_%s.png' % (d0_path.rsplit('/', 1)[0], idd_p1)

                Scene_path = flow_path.replace('Flow', 'SceneFlow').replace('flow', 'sceneFlow')

                im_path = flow_path.replace('backwardFlow', 'rgb').replace('forwardFlow', 'rgb')
                im0_path = '%s/rgb_%s.jpg' % (im_path.rsplit('/', 1)[0], idd)
                im1_path = '%s/rgb_%s.jpg' % (im_path.rsplit('/', 1)[0], idd_p1)
                frame_id = im1_path.split('/')[-1]
                extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                flow_list += [flow_path]
                image_list += [[im0_path,im1_path]]
                depth_list += [[d0_path,d1_path,Scene_path]]

                calib_list = np.loadtxt(im0_path.split("frames")[0]+'intrinsic.txt', skiprows=1)
                calib_l += [calib_list[int(idd)][2:]]

        #获取测试序列，每隔上100就取一个，那大概是1600个图像参与测试，那很权威了
        for idx in range(len(image_list)):
            if idx % 50 == 0:
                self.extra_info.append(extra_info[idx])
                self.flow_list.append(flow_list[idx])
                self.image_list.append(image_list[idx])
                self.depth_list.append(depth_list[idx])
                self.calib.append(calib_l[idx])
    def get_dc(self,index):#获取第一帧深度，第二帧深度，还有第一帧的变化深度,再来个内参
        d1 = cv2.imread(self.depth_list[index][0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#以厘米为单位的深度
        d2 = cv2.imread(self.depth_list[index][1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#以厘米为单位的深度
        d3 = (cv2.imread(self.depth_list[index][2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]*2.0/65535.0 -1.0)*1000.0+d1#以厘米为单位的深度变化量

        fx, fy, cx, cy = self.calib[index]
        intrinsics = np.array([fx, fy, cx, cy])
        fl = intrinsics[0]
        d1 = 60*fl / d1
        d2 = 60*fl / d2
        d3 = 60*fl / d3

        return d1,d2,d3,intrinsics
    def __len__(self):
        return len(self.extra_info)

    def __getitem__(self, index):
        index = index % len(self.image_list)
        d1, d2, dd1, intrinsics = self.get_dc(index)



        flow, valid = frame_utils.read_vkitti_png_flow(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        d1 = np.array(d1).astype(np.float32)
        d2 = np.array(d2).astype(np.float32)
        dd1 = np.array(dd1).astype(np.float32)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()


        intrinsics = torch.from_numpy(intrinsics).float()
        d1 = torch.from_numpy(d1[np.newaxis,:,:]).float()
        d2 = torch.from_numpy(d2[np.newaxis,:,:]).float()
        dz = (torch.from_numpy(dd1[np.newaxis,:,:])-d1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, d1, d2, dz,intrinsics,valid
#这里搞一个KITTI的测试版本
class kitti_test(data.Dataset):
    def __init__(self, aug_params=None, split='training', root='/mnt/hdd/hanling/sceneflow_data/KITTI'):
        self.extra_info= []
        self.flow_list = []
        self.image_list = []
        self.depth_list = []
        self.calib = []
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        dispp1 = []
        dispp2 = []
        flow =[]

        root = osp.join(root, split)
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

        disp1pre = sorted(glob(osp.join(root, 'disp_monster_training/*_10.png')))
        disp2pre = sorted(glob(osp.join(root, 'disp_monster_training/*_11.png')))

        for j in range(images2o.__len__()):
            if j % 2 == 0:
                images1.append(images1o[j])
                images2.append(images2o[j])
                disp1.append(disp1o[j])
                disp2.append(disp2o[j])
                dispp1.append(disp1pre[j])
                dispp2.append(disp2pre[j])

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2,dispps1,dispps2 in zip(disp1, disp2,dispp1,dispp2):
            self.depth_list += [[disps1, disps2,dispps1,dispps2]]

        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
            if j % 2 == 0:
                flow.append(flowo[j])
        self.flow_list = flow


    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1s = disparity_loader(self.depth_list[index][0])
        d2s = disparity_loader(self.depth_list[index][1])

        d1p = readDepth(self.depth_list[index][2])
        d2p = readDepth(self.depth_list[index][3])
        d3 = d1p+(d2s-d1s)

        H, W = d1s.shape
        intrinsics = np.array([721.5377, 721.5377, H/2.0,  W/2.0])

        return d1p,d2p,d3,intrinsics
    def __len__(self):
        return len(self.extra_info)

    def __getitem__(self, index):
        index = index % len(self.image_list)
        d1, d2, dd1, intrinsics = self.get_dc(index)
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        d1 = np.array(d1).astype(np.float32)
        d2 = np.array(d2).astype(np.float32)
        dd1 = np.array(dd1).astype(np.float32)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()


        intrinsics = torch.from_numpy(intrinsics).float()
        d1 = torch.from_numpy(d1[np.newaxis,:,:]).float()
        d2 = torch.from_numpy(d2[np.newaxis,:,:]).float()
        dz = (torch.from_numpy(dd1[np.newaxis,:,:])-d1).float()

        valid = torch.from_numpy(valid)

        return img1, img2, flow, d1, d2, dz,intrinsics,valid
class kitti_testsubmit(data.Dataset):
    def __init__(self, aug_params=None, split='testing', root='/home/lh/all_datasets/kitti'):
        self.extra_info= []
        self.flow_list = []
        self.image_list = []
        self.depth_list = []
        self.calib = []
        images1 =[]
        images2 =[]
        dispp1 = []
        dispp2 = []

        root = osp.join(root, split)
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        #disp1pre = sorted(glob(osp.join(root, 'disp_monster_testing/*_10.png')))
        #disp2pre = sorted(glob(osp.join(root, 'disp_monster_testing/*_11.png')))

        disp1pre = sorted(glob(osp.join(root, 'disp_ganet_testing/*_10.png')))
        disp2pre = sorted(glob(osp.join(root, 'disp_ganet_testing/*_11.png')))

        #disp1pre = sorted(glob(osp.join(root, 'disp_lea_testing/*_10.png')))
        #disp2pre = sorted(glob(osp.join(root, 'disp_lea_testing/*_11.png')))
        for j in range(images2o.__len__()):
                images1.append(images1o[j])
                images2.append(images2o[j])
                dispp1.append(disp1pre[j])
                dispp2.append(disp2pre[j])

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for dispps1,dispps2 in zip(dispp1,dispp2):
            self.depth_list += [[dispps1,dispps2]]



    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        #d1p = readDepth(self.depth_list[index][0])
        #d2p = readDepth(self.depth_list[index][1])
        d1p = np.abs(disparity_loader(self.depth_list[index][0]))
        d2p = np.abs(disparity_loader(self.depth_list[index][1]))
        return d1p,d2p
    def __len__(self):
        return len(self.extra_info)

    def __getitem__(self, index):
        index = index % len(self.image_list)
        d1, d2= self.get_dc(index)
        frame_id = self.extra_info[index][0]
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        d1 = np.array(d1).astype(np.float32)
        d2 = np.array(d2).astype(np.float32)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        d1 = torch.from_numpy(d1[np.newaxis,:,:]).float()
        d2 = torch.from_numpy(d2[np.newaxis,:,:]).float()

        return img1, img2, d1, d2,frame_id