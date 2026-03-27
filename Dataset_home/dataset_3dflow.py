#这个数据集专门用来优化3D流
#todo 主要需要涉及的数据集应该有KITTI,Spring,FlyingThings3D,vKITTI,Driving,Monkey
#先把这个Spring和Vkitti数据集的数据搞出来，vKITTI严格来说应该和kitti差不多，
#输出应该是包含im1,im2,d1,d2,delta_d1,flow,valid
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
import Dataset_home.flowio as flowio
import  cv2
from Dataset_home.augment3d import SceneFlowAugmentor
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

class SceneFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        self.augmentor = SceneFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.depth_list = []
        self.image_list = []
        self.extra_info = []

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
        flow,_ = frame_utils.readFlowdriving(self.flow_list[index])#第三个维度为有效Valid
        valid = None
        d1,d2,dd1,intrinsics = self.get_dc(index)

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

        if self.augmentor is not None:#光流和深度变化率是两个独立的掩膜
             img1, img2, flow,d1,d2,dd1,intrinsics= self.augmentor(img1, img2, flow,d1,d2,dd1,intrinsics)



        '''
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(flow2rgb(flow))
        plt.show()
        plt.imshow(d1.clip(0,32))
        plt.show()
        plt.imshow(dd1.clip(0,32))
        plt.show()
        '''
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()


        intrinsics = torch.from_numpy(intrinsics).float()
        d1 = torch.from_numpy(d1[np.newaxis,:,:]).float()
        d2 = torch.from_numpy(d2[np.newaxis,:,:]).float()
        dd1 = torch.from_numpy(dd1[np.newaxis,:,:]).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, d1, d2,dd1,intrinsics,valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.calib = v * self.calib
        return self
    def __len__(self):
        return len(self.image_list)


class Driving(SceneFlowDataset):
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
                self.depth_list += [[d0_path,d1_path,dc_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]

    def triangulation(self, disp,index, bl=1):#kitti flow 2015
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):#获取第一帧深度，第二帧深度，还有第一帧的变化深度,再来个内参
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1]))
        d1c = np.abs(disparity_loader(self.depth_list[index][2])+d1)

        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050

        H,W = d1.shape
        fx, fy, cx, cy = (fl, fl, W/2.0, H/2.0)
        intrinsics = np.array([fx, fy, cx, cy])

        return self.triangulation(d1,index),self.triangulation(d2,index),self.triangulation(d1c,index),intrinsics




def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'pretrain':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        train_dataset =  driving
    elif args.stage == 'fineting':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.5, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        train_dataset = driving


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=1, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
