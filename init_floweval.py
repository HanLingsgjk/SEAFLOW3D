import sys
from skimage.metrics import structural_similarity
sys.path.append('core')
import cv2
from glob import glob
import os.path as osp
from PIL import Image
import argparse
import imageio
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.util_flow import readPFM
import core.dataset_occ as datasets
from core.utils import frame_utils
from core.utils import flow_viz
from torch.utils.data import DataLoader
from core.utils.utils import InputPadder, forward_interpolate,coords_grid
from core.utils.flow_viz import flow2rgb
from core.raft_init import RAFT_init
from core.raft_initv2 import RAFT_initv2
from model_home.raft_base import RAFT
#from model_home.sea_raft import RAFT
from model_home.raft_initscale8 import RAFT_initv3

from model_home.sea_raft_init import RAFT_seainit
from model_home.ccmr import CCMR
#raftinit_v2catconv.pth raft_orin.pth
@torch.no_grad()
@torch.no_grad()
def validate_kittiraft(model, iters=12,f_iter=0):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI_test(split='kitti_test')

    out_list, epe_list = [], []
    flow_liste = []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, _, _, _, _, mask, _, frame_id= val_dataset[val_id]
        valid_gt = torch.from_numpy(mask)
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=32)
        image1, image2 = padder.pad(image1, image2)

        #res = gma_forward(image1, image2)
        #flow_pr = res['flow_preds'][0]
        if f_iter>0:
            flow_low, flow_pr,flowlist = model(image1, image2, iters=iters, test_mode=True,frist_iter=f_iter)
        else:
            flow_low, flow_pr, flowlist = model(image1, image2, iters=iters, test_mode=True)
        errorlist = []
        usedmask = valid_gt>0.5
        epe = torch.sum((flow_gt) ** 2, dim=0).sqrt()
        errorlist.append(epe[usedmask].mean().item())
        for flowp in flowlist:
            flow = padder.unpad(flowp[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            errorlist.append(epe[usedmask].mean().item())

        flow = padder.unpad(flow_pr[0]).cpu()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        flow_liste.append(errorlist)
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    flow_listn = np.array(flow_liste)
    flow_listn = flow_listn.mean(axis=0)

    fd = (((flow_listn[0])-(flow_listn[1]))/(flow_listn[0]))*100
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f, %f" % (epe, f1, fd))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def validate_Sintel_train(model, iters=6,iftest=False,f_iter=0):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.MpiSinteleval(dtype='clean',iftest=iftest)#kitti_test
    out_list, epe_list = [], []
    flow_liste = []
    for val_id in range(len(val_dataset)):#len(val_dataset)
        #print(val_id)
        image1, image2, flow_gt, dc_gt, occmask, valid_gt, extra_info = val_dataset[val_id]
        valid_gt = torch.from_numpy(valid_gt)
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=16)
        image1, image2 = padder.pad(image1, image2)

        if f_iter > 0:
            flow_low, flow_pr, flowlist = model(image1, image2, iters=iters, test_mode=True, frist_iter=f_iter)
        else:
            flow_low, flow_pr, flowlist = model(image1, image2, iters=iters, test_mode=True)
        errorlist = []
        usedmask = valid_gt>0.5
        epe = torch.sum((flow_gt) ** 2, dim=0).sqrt()
        errorlist.append(epe[usedmask].mean().item())
        for flowp in flowlist:
            flow = padder.unpad(flowp[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            errorlist.append(epe[usedmask].mean().item())


        flow = padder.unpad(flow_pr[0]).cpu()
        '''
        flows = flow.permute(1,2,0).numpy()
        plt.imshow(flow2rgb(flows))
        plt.show()
        flows = flow_gt.permute(1,2,0).numpy()
        plt.imshow(flow2rgb(flows))
        plt.show()
        '''
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        flow_liste.append(errorlist)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    flow_listn = np.array(flow_liste)
    flow_listn = flow_listn.mean(axis=0)
    fd = (((flow_listn[0])-(flow_listn[1]))/(flow_listn[0]))*100
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    print("Validation clean Sintel: %f, %f, %f" % (epe, f1,fd))
    #TODO 测评最终版本的Sintel
    flow_liste = []
    val_dataset = datasets.MpiSinteleval(dtype='final',iftest=iftest)#kitti_test
    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):#len(val_dataset)
        #print(val_id)
        image1, image2, flow_gt, dc_gt, occmask, valid_gt, extra_info = val_dataset[val_id]
        valid_gt = torch.from_numpy(valid_gt)
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=32)
        image1, image2 = padder.pad(image1, image2)
        errorlist = []
        if f_iter > 0:
            flow_low, flow_pr, flowlist = model(image1, image2, iters=iters, test_mode=True, frist_iter=f_iter)
        else:
            flow_low, flow_pr, flowlist = model(image1, image2, iters=iters, test_mode=True)
        usedmask = valid_gt>0.5
        epe = torch.sum((flow_gt) ** 2, dim=0).sqrt()
        errorlist.append(epe[usedmask].mean().item())
        for flowp in flowlist:
            flow = padder.unpad(flowp[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            errorlist.append(epe[usedmask].mean().item())

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        flow_liste.append(errorlist)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    flow_listn = np.array(flow_liste)
    flow_listn = flow_listn.mean(axis=0)
    fd = (((flow_listn[0])-(flow_listn[1]))/(flow_listn[0]))*100
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation final Sintel: %f, %f, %f" % (epe, f1,fd))
    return {'kitti-epe': epe, 'kitti-f1': f1}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--model_type', default='ccmr')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--start', default=0, type=int,
                        help='where to start')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))

    pretrained_dict = torch.load(args.model)
    old_list = {}
    for k, v in pretrained_dict.items():
        # if k.find('encoder.convc1')<0 :
        old_list.update({k: v})
    model.load_state_dict(old_list, strict=False)

    model.cuda()
    model.eval()

    with torch.no_grad():
        validate_kittiraft(model.module, iters=12,f_iter=6)
        validate_Sintel_train(model.module, iters=12, iftest=True,f_iter=6)
