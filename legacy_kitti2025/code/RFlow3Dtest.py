import sys
sys.path.append('core')
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_home.RFlow3Duvdx4 import RFlow3D_MS
from model_home.RFlow3Duvd_MS import RFlow3D_MSD
from model_home.RFlow3Duvd_MSv3 import RFlow3D_MSDv3
from model_home.RFlow3Duvd_MSv5 import RFlow3D_MSDv5
from model_home.RFlow3Duvd_MSv7 import RFlow3D_MSDv7
from model_home.RFlow3D_MSabv12 import RFlow3D_MSab_v12
from model_home.RFlow3D_MSabv0 import RFlow3D_MSab_v0
from model_home.RFlow3D_MSabv1 import RFlow3D_MSab_v1
from model_home.RFlow3D_MSabv2 import RFlow3D_MSab_v2
from model_home.RFlow3D_MSabv3 import RFlow3D_MSab_v3
from model_home.RFlow3D_MSabv4_1 import RFlow3D_MSab_v7
from model_home.RFlow3D_MSabv4_2 import RFlow3D_MSab_v8
from model_home.RFlow3D_MSabv4_3 import RFlow3D_MSab_v9


from model_home.RFlow3D_MSabv12 import RFlow3D_MSab_v12
import Dataset_home.dataset_3dflowuvd as datasets
from core.utils.utils import InputPadder
from core.utils.flow_viz import flow2rgb
from core.utils import frame_utils

def validate_sintel(model, iters=6,typei= 't_test'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.Sintel_test(ifeval=typei,dstype='clean')

    out_list, epe_list = [], []
    outdz_list, epedz_list = [], []
    time_all = 0
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, d1, d2, dz, intrinsics, valid= val_dataset[val_id]
        validd = (dz < 1000)[0, :, :]
        valid_gt = valid
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        d3_gt = d1 + dz
        d1_in = d1.detach().clone()
        d1 = d1[None].cuda()
        d2 = d2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=32)
        image1, image2 = padder.pad(image1, image2)
        d1in, d2in = padder.pad(d1, d2)
        time1 = time.time()
        _, flow_pre, dz_pre = model(image1, image2, d1in, d2in, iters=iters, test_mode=True)
        time2 = time.time()
        time_all =time_all+ time2-time1
        flow = padder.unpad(flow_pre).cpu()
        dz_pre = padder.unpad(dz_pre).cpu()
        dz_gt = dz
        '''
        dz_gtS = dz_gt.detach().cpu().numpy()
        dz_pres = dz_pre[0].detach().cpu().numpy()
        d1s = d1.detach().cpu().numpy()
        d2s = d2.detach().cpu().numpy()
        flowpre = flow[0].permute(1,2,0).detach().cpu().numpy()
        flow_gts = flow_gt.permute(1,2,0).detach().cpu().numpy()
        plt.imshow(flow2rgb(flow_gts))
        plt.show()
        plt.imshow(flow2rgb(flowpre))
        plt.show()
        plt.imshow(dz_gtS[0])
        plt.show()
        plt.imshow(dz_pres[0])
        plt.show()
        plt.imshow(np.abs(dz_pres[0]-dz_gtS[0]))
        plt.show()
        '''
        #不行了这个光流必须搞个采样了，这他妈的数据集有问题啊
        epedz = torch.sum((dz_gt - dz_pre[0]) ** 2, dim=0).sqrt()
        magdz = torch.sum(dz_gt ** 2, dim=0).sqrt()
        epedz = epedz.view(-1)
        magdz = magdz.view(-1)

        epe = torch.sum((flow[0] - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        vald = validd.view(-1) >= 0.5
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        outdz = ((epedz > 0.1) & ((epedz / magdz) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        epedz_list.append(epedz[vald].mean().item())
        outdz_list.append(outdz[vald].cpu().numpy())
    timemean = time_all/val_id
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epedz_list = np.array(epedz_list)
    outdz_list = np.concatenate(outdz_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    epedz = np.mean(epedz_list)
    f1dz = 100 * np.mean(outdz_list)

    print("Validation Sintel Flow: %f, %f;  Dz: %f, %f;time_mean %f;" % (epe, f1,epedz,f1dz,timemean))
    return {'sintel-epe': epe, 'sintel-f1': f1}
def validate_vkitti(model, iters=6):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.vkitti_test()
    time_all = 0
    out_list, epe_list = [], []
    outdz_list, epedz_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, d1, d2, dz, intrinsics, valid= val_dataset[val_id]

        valid_gt = valid
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        d3_gt = d1 + dz
        d1_in = d1.detach().clone()
        d1 = d1[None].cuda()
        d2 = d2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=32)
        image1, image2 = padder.pad(image1, image2)
        d1in, d2in = padder.pad(d1, d2)

        time1 = time.time()
        _, flow_pre, dz_pre = model(image1, image2, d1in, d2in, iters=iters, test_mode=True)
        time2 = time.time()
        time_all =time_all+ time2-time1

        flow = padder.unpad(flow_pre).cpu()
        dz_pre = padder.unpad(dz_pre).cpu()
        dz_gt = dz
        '''
        dz_gtS = dz_gt.detach().cpu().numpy()
        dz_pres = dz_pre[0].detach().cpu().numpy()
        d1s = d1.detach().cpu().numpy()
        d2s = d2.detach().cpu().numpy()
        flowpre = flow[0].permute(1,2,0).detach().cpu().numpy()
        flow_gts = flow_gt.permute(1,2,0).detach().cpu().numpy()
        plt.imshow(flow2rgb(flow_gts))
        plt.show()
        plt.imshow(flow2rgb(flowpre))
        plt.show()
        plt.imshow(dz_gtS[0])
        plt.show()
        plt.imshow(dz_pres[0])
        plt.show()
        plt.imshow(np.abs(dz_pres[0]-dz_gtS[0]))
        plt.show()
        '''
        #不行了这个光流必须搞个采样了，这他妈的数据集有问题啊
        epedz = torch.sum((dz_gt - dz_pre[0]) ** 2, dim=0).sqrt()
        magdz = torch.sum(dz_gt ** 2, dim=0).sqrt()
        epedz = epedz.view(-1)
        magdz = magdz.view(-1)

        epe = torch.sum((flow[0] - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        outdz = ((epedz > 0.1) & ((epedz / magdz) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        epedz_list.append(epedz[val].mean().item())
        outdz_list.append(outdz[val].cpu().numpy())
    timemean = time_all/val_id
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epedz_list = np.array(epedz_list)
    outdz_list = np.concatenate(outdz_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    epedz = np.mean(epedz_list)
    f1dz = 100 * np.mean(outdz_list)

    print("Validation KITTI Flow: %f, %f;  Dz: %f, %f;time_mean %f;" % (epe, f1,epedz,f1dz,timemean))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def validate_kitti(model, iters=6):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.kitti_test()
    time_all = 0
    out_list, epe_list = [], []
    outdz_list, epedz_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, d1, d2, dz, intrinsics, valid= val_dataset[val_id]
        valid_gt = valid
        d1_in = d1.detach().clone()
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        d3_gt = d1+dz
        d1 = d1[None].cuda()
        d2 = d2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=32)
        image1, image2 = padder.pad(image1, image2)
        d1in, d2in = padder.pad(d1, d2)

        time1 = time.time()
        _, flow_pre, dz_pre = model(image1, image2, d1in, d2in, iters=iters, test_mode=True)
        time2 = time.time()
        time_all = time_all + time2 - time1


        flow = padder.unpad(flow_pre).cpu()
        dz_pre = padder.unpad(dz_pre).cpu()
        dz_gt = dz
        '''
        dz_gtS = dz_gt.detach().cpu().numpy()
        dz_pres = dz_pre[0].detach().cpu().numpy()
        d1s = d1.detach().cpu().numpy()[0][0]
        d2s = d2.detach().cpu().numpy()[0][0]
        flowpre = flow[0].permute(1,2,0).detach().cpu().numpy()
        flow_gts = flow_gt.permute(1,2,0).detach().cpu().numpy()
        plt.imshow(flow2rgb(flow_gts))
        plt.show()
        plt.imshow(flow2rgb(flowpre))
        plt.show()
        plt.imshow(d1s)
        plt.show()
        plt.imshow(d2s)
        plt.show()
        dzout = np.abs(dz_pres[0]-dz_gtS[0])
        dzout[valid<1]=0
        plt.imshow(dzout)
        plt.show()
        '''
        #不行了这个光流必须搞个采样了，这他妈的数据集有问题啊
        epedz = torch.sum((dz_gt - (dz_pre[0])) ** 2, dim=0).sqrt()
        magdz = torch.sum(dz_gt** 2, dim=0).sqrt()
        epedz = epedz.view(-1)
        magdz = magdz.view(-1)

        epe = torch.sum((flow[0] - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        outdz = ((epedz > 0.1) & ((epedz / magdz) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        epedz_list.append(epedz[val].mean().item())
        outdz_list.append(outdz[val].cpu().numpy())
    timemean = time_all/val_id
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epedz_list = np.array(epedz_list)
    outdz_list = np.concatenate(outdz_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    epedz = np.mean(epedz_list)
    f1dz = 100 * np.mean(outdz_list)

    print("Validation KITTI Flow: %f, %f;  Dz: %f, %f;time_mean %f;" % (epe, f1,epedz,f1dz,timemean))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def kitti_sceneflow_submission(model):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.kitti_testsubmit()
    output_filenamerootfv= '/home/lh/CSCV_occ/submit_preRF3D619/flow_viz'
    if os.path.exists(output_filenamerootfv) == False:
        os.makedirs(output_filenamerootfv)
    output_filenamerootf= '/home/lh/CSCV_occ/submit_preRF3D619/flow'
    if os.path.exists(output_filenamerootf) == False:
        os.makedirs(output_filenamerootf)
    output_filenamerootd1= '/home/lh/CSCV_occ/submit_preRF3D619/disp_0'
    if os.path.exists(output_filenamerootd1) == False:
        os.makedirs(output_filenamerootd1)
    output_filenamerootd2= '/home/lh/CSCV_occ/submit_preRF3D619/disp_1'
    if os.path.exists(output_filenamerootd2) == False:
        os.makedirs(output_filenamerootd2)
    for test_id in range(0,len(test_dataset),1):

        image1, image2, disp1,disp2,frame_id = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti',sp = 16)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        d1 = disp1[None].cuda()
        d2 = disp2[None].cuda()
        d1in, d2in = padder.pad(d1, d2)

        _, flow_pr, dz_pre = model(image1, image2, d1in, d2in, test_mode=True)


        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).detach().cpu().numpy()
        dz_pre = padder.unpad(dz_pre).detach().cpu().numpy()
        disp2o = dz_pre[0,0]+disp1[0].detach().cpu().numpy()

        flow_viz = cv2.cvtColor(flow2rgb(flow), cv2.COLOR_RGB2BGR)

        cv2.imwrite('%s/%s' % (output_filenamerootfv, frame_id), flow_viz*255)

        disp1 =  (disp1[0].detach().cpu().numpy() * 256).astype('uint16')
        disp2 =  (disp2o * 256).astype('uint16')
        output_filename = os.path.join(output_filenamerootf, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        cv2.imwrite('%s/%s' % (output_filenamerootd1, frame_id), disp1)
        cv2.imwrite('%s/%s' % (output_filenamerootd2, frame_id), disp2)



        print(test_id)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--model_type', default='ccmr')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--itr1', type=int, default=6)
    parser.add_argument('--itr2', type=int, default=4)
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

    model = torch.nn.DataParallel(RFlow3D_MSab_v8(args))

    pretrained_dict = torch.load(args.model)
    old_list = {}
    for k, v in pretrained_dict.items():
        old_list.update({k: v})
    model.load_state_dict(old_list, strict=False)

    model.cuda()
    model.eval()

    with torch.no_grad():
        #validate_vkitti(model.module, iters=6)

        #kitti_sceneflow_submission(model.module)

        #validate_kitti(model.module, iters=6)
        validate_sintel(model.module,typei= 't_all')