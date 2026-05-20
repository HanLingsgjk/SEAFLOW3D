import sys
sys.path.append('core')
import argparse
import os
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os.path as osp
from model_home.RFlow3D_MSabv4_2 import RFlow3D_MSab_v8
import Dataset_home.dataset_3dflowuvd as datasets
from core.utils.utils import InputPadder
from core.utils.flow_viz import flow2rgb
from PIL import Image
from core.utils import frame_utils
def sintel_show(model, iters,typei= 't_test',detype='final'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.Sintel_test(ifeval=typei,dstype=detype)

    out_list, epe_list = [], []
    outdz_list, epedz_list = [], []
    time_all = 0
    save_root = '/home/lh/CSCV_occ_new/error_show/sintel_SEAshow/'
    for val_id in range(len(val_dataset)):
        if val_id >0:
            print(time_all/val_id)
        image1, image2, flow_gt, d1, d2, dz, intrinsics, valid= val_dataset[val_id]
        valid_gt = valid
        flow_gt = flow_gt.cuda()
        ht, wd = image1.shape[1:]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        d1 = d1[None].cuda()
        d2 = d2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti',sp=32)
        image1, image2 = padder.pad(image1, image2)
        d1in, d2in = padder.pad(d1, d2)

        time1 = time.time()
        _, flow_pre, dz_pre = model(image1, image2, d1in, d2in, iters=iters, test_mode=True)
        time2 = time.time()
        time_all =time_all+ time2-time1
        flow = padder.unpad(flow_pre)

        #不行了这个光流必须搞个采样了，这他妈的数据集有问题啊
        plt.imshow(flow2rgb(flow[0].permute(1,2,0).detach().cpu().numpy()))
        plt.axis('off')

        #epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        epe = torch.sum((flow[0] - flow_gt) ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        val = valid_gt.view(-1) >= 0.5
        f1dz = np.mean(epe[val].cpu().numpy())
        # 左下角显示误差（白色加粗）
        plt.text(
            10,  # x 偏移
            60,  # y 位置
            f'Fl-EPE : {f1dz:.2f}',  # 文本内容
            color='black',
            fontsize=14,
            fontweight='bold',
            va='bottom',
            ha='left',
            bbox=dict(
                facecolor='lightgray',  # 背景颜色，可以改成 'lightgray'
                alpha=0.5,  # 透明度
                edgecolor='none',  # 去掉边框
                boxstyle='round,pad=0.3'  # 圆角和内边距
            )
        )
        save_name = save_root+str(val_id).zfill(4) +'a.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')


        #todo 这里存深度变化率图会不会好一点
        save_nameb = save_root + str(val_id).zfill(4) + 'c.png'
        plt.imshow(image1[0].permute(1,2,0).detach().cpu().numpy()/255)
        plt.axis('off')
        plt.savefig(save_nameb, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        save_namec = save_root + str(val_id).zfill(4) + 'b.png'
        plt.imshow(flow2rgb(flow_gt.permute(1,2,0).detach().cpu().numpy()))
        plt.axis('off')
        plt.savefig(save_namec, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')


    return 0
def validate_sintel(model, iters=6,typei= 't_test',detype='final'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.Sintel_test(ifeval=typei,dstype=detype)

    out_list, epe_list = [], []
    outdz_list, epedz_list = [], []
    time_all = 0
    save_root = '/home/lh/CSCV_occ_new/error_show/sintelv0/'
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
        val = valid_gt.view(-1) >= 0.5
        vald = validd.view(-1) >= 0.5
        dz_gtS = dz_gt.detach().cpu().numpy()
        dz_pres = dz_pre[0].detach().cpu().numpy()

        #不行了这个光流必须搞个采样了，这他妈的数据集有问题啊
        epedz = torch.sum((dz_gt - dz_pre[0]) ** 2, dim=0).sqrt()
        magdz = torch.sum(dz_gt ** 2, dim=0).sqrt()
        '''
        dzshow = epedz
        dzshow = (dzshow*0.5).clip(0,1)
        cmap = plt.get_cmap('jet')
        # 将灰度值映射到RGBA伪彩色
        dzout_color = cmap(dzshow)[:, :, :3]  # 取前三通道 (RGB)
        # 将 mask==0 的区域置为黑色
        dzout_color[validd<1] = 1.0
        plt.imshow(dzout_color)
        plt.axis('off')
        '''

        epedz = epedz.view(-1)
        magdz = magdz.view(-1)

        epe = torch.sum((flow[0] - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        outdz = ((epedz > 0.1) & ((epedz / magdz) > 0.05)).float()
        '''
        f1dz = np.mean(epedz[vald].cpu().numpy())
        # 左下角显示误差（白色加粗）
        plt.text(
            10,  # x 偏移
            60,  # y 位置
            f'EPE : {f1dz:.2f}',  # 文本内容
            color='white',
            fontsize=14,
            fontweight='bold',
            va='bottom',
            ha='left',
            bbox=dict(
                facecolor='lightgray',  # 背景颜色，可以改成 'lightgray'
                alpha=0.5,  # 透明度
                edgecolor='none',  # 去掉边框
                boxstyle='round,pad=0.3'  # 圆角和内边距
            )
        )
        save_name = save_root+str(val_id).zfill(4) +'a.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        #todo 这里存深度变化率图会不会好一点
        save_nameb = save_root + str(val_id).zfill(4) + 'b.png'
        plt.imshow(image1[0].permute(1,2,0).detach().cpu().numpy()/255)
        plt.axis('off')
        plt.savefig(save_nameb, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        '''

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

def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
        pth, 'PNG')
from Dataset_home.dataset_3dflowuvd import readDepth
from MonSter.demogo import run_on_pair


def Demo_runtime(model):
    left_root = '/mnt/hdd/hanling/KITTI/testing/image_2/'
    right_root = '/mnt/hdd/hanling/KITTI/testing/image_3/'

    output_filenameflow = os.path.join('/mnt/hdd/hanling/KITTI/testing/', 'flowshow/')
    if os.path.exists(output_filenameflow) == False:
        os.makedirs(output_filenameflow)

    left_images_all = sorted(glob(osp.join(left_root, '*')))
    right_images_all = sorted(glob(osp.join(right_root, '*')))

    left_images1 = left_images_all[:-1]
    left_images2 = left_images_all[1:]
    right_images1 = right_images_all[:-1]
    right_images2 = right_images_all[1:]

    depth_cache = {}
    depth_cache_keys = []
    CACHE_SIZE = 3

    def get_depth(left_path, right_path):
        key = (left_path, right_path)

        if key in depth_cache:
            return depth_cache[key]

        depth = run_on_pair(left_path, right_path)

        depth_cache[key] = depth
        depth_cache_keys.append(key)

        if len(depth_cache_keys) > CACHE_SIZE:
            old_key = depth_cache_keys.pop(0)
            if old_key in depth_cache:
                del depth_cache[old_key]

        return depth

    for idx in range(len(left_images1)):
        print(idx)

        left_t = left_images1[idx]
        left_t1 = left_images2[idx]
        right_t = right_images1[idx]
        right_t1 = right_images2[idx]

        img1 = frame_utils.read_gen(left_t)
        img2 = frame_utils.read_gen(left_t1)

        d1p = get_depth(left_t, right_t)
        d2p = get_depth(left_t1, right_t1)

        # 后面保持你的原逻辑

        idout = osp.splitext(osp.basename(left_t))[0]

        img1 = np.array(img1)
        img2 = np.array(img2)

        if img1.ndim < 3:
            img1 = np.concatenate([img1[:, :, None]] * 3, axis=-1).astype(np.uint8)[..., :3]
            img2 = np.concatenate([img2[:, :, None]] * 3, axis=-1).astype(np.uint8)[..., :3]
        else:
            img1 = img1.astype(np.uint8)[..., :3]
            img2 = img2.astype(np.uint8)[..., :3]

        img1show = img1.astype(np.float32).copy()

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        image1 = img1[None].cuda()
        image2 = img2[None].cuda()
        d1p = torch.from_numpy(d1p[None, None]).float().cuda()
        d2p = torch.from_numpy(d2p[None, None]).float().cuda()

        padder = InputPadder(image1.shape, mode='kitti', sp=16)
        image1, image2 = padder.pad(image1, image2)
        d1p, d2p = padder.pad(d1p, d2p)

        _, flow_pre, dz_pre = model(image1, image2, d1p, d2p, iters=666, test_mode=True)

        flow = padder.unpad(flow_pre[0]).detach().cpu()
        dz_pre = padder.unpad(dz_pre).detach().cpu()

        flowviz = flow2rgb(flow.permute(1, 2, 0).numpy())
        frame_id = idout + '.png'

        dchange = dz_pre[0, 0].numpy()
        dchange = (dchange * 0.05).clip(0, 1) * 255

        colormap = plt.get_cmap('plasma')
        heatmap = colormap(dchange.astype(np.uint8)).astype(np.float32)[:, :, :3]

        imgout = np.zeros((dchange.shape[0] * 3, dchange.shape[1], 3), dtype=np.float32)
        imgout[:dchange.shape[0], :, :] = img1show / 255.0
        imgout[dchange.shape[0]:dchange.shape[0] * 2, :, :] = flowviz
        imgout[-dchange.shape[0]:, :, :] = heatmap

        save_img_u8(imgout, os.path.join(output_filenameflow, frame_id))
        print(frame_id)
def Demo_vis_dir(model):
    imroot ='/mnt/hdd/hanling/KITTI/testing/image_2/'
    depthroot = '/mnt/hdd/hanling/KITTI/testing/mout/'

    output_filenameflow = os.path.join('/mnt/hdd/hanling/KITTI/testing/','flowshow/')
    if os.path.exists(output_filenameflow) == False:
        os.makedirs(output_filenameflow)

    images1 = sorted(glob(osp.join(imroot,'*')))
    images2 = images1[1:]
    images1.pop()

    disp1 = sorted(glob(osp.join(depthroot, '*')))
    disp2 = disp1[1:]
    disp1.pop()
    for id in range(images1.__len__()):
        print(id)
        id = id
        img1 = frame_utils.read_gen(images1[id])
        img2 = frame_utils.read_gen(images2[id])
        d1p = readDepth(disp1[id])
        d2p = readDepth(disp2[id])
        pathsplit = images1[id].split('/')
        idout = pathsplit[-1].split('.')[0]

        img1 = np.array(img1)
        img2 = np.array(img2)
        if img1.shape.__len__()<3:
            img1 = np.concatenate([img1[:,:,np.newaxis],img1[:,:,np.newaxis],img1[:,:,np.newaxis]],axis=-1).astype(np.uint8)[..., :3]
            img2 = np.concatenate([img2[:, :, np.newaxis], img2[:, :, np.newaxis], img2[:, :, np.newaxis]], axis=-1).astype(np.uint8)[..., :3]
        else:
            img1 = img1.astype(np.uint8)[..., :3]
            img2 = img2.astype(np.uint8)[..., :3]
        img1show = img1.astype(np.float32).copy()

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()
        d1p = torch.from_numpy(d1p[None,None, :,  :]).float().cuda()
        d2p = torch.from_numpy(d2p[None,None, :,  :]).float().cuda()


        padder = InputPadder(image1.shape, mode='kitti',sp=16)
        image1, image2 = padder.pad(image1, image2)
        d1p, d2p = padder.pad(d1p, d2p)
        #计算光流
        _, flow_pre, dz_pre = model(image1, image2, d1p, d2p, iters=666, test_mode=True)
        flow = padder.unpad(flow_pre[0]).detach().cpu()
        dz_pre = padder.unpad(dz_pre).detach().cpu()

        flowviz = flow2rgb(flow.permute(1, 2, 0).numpy())
        frame_id = idout+'.png'
        frameL_id = idout + 'L.png'
        frameR_id = idout + 'R.png'


        dchange =dz_pre[0, 0].numpy()
       # plt.imshow(dchange,cmap='plasma')
        #plt.show()
        dchange = ((dchange ) * 0.05).clip(0, 1) * 255
        colormap = plt.get_cmap('plasma')#plasma viridis
        heatmap = (colormap((dchange).astype(np.uint8))).astype(np.float32)[:, :, :3]
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)



        imgout = np.zeros((dchange.shape[0]*3,dchange.shape[1],3))
        imgout[:dchange.shape[0],:,:] = img1show/255.0
        imgout[dchange.shape[0]:dchange.shape[0]*2,:, :] = flowviz
        imgout[-dchange.shape[0]:,:, :] = heatmap
        #imlist.append(Image.fromarray((imgout* 255.0).astype(np.uint8)))

        save_img_u8(imgout,output_filenameflow+frame_id)
        print(frame_id)

import torch
import numpy as np
import time
from tqdm import tqdm

import torch
import numpy as np
from tqdm import tqdm
def validate_sintel_sequence(model, iters=8, typei='t_test', detype='clean'):
    """
    Evaluate Sintel sequence (Clean/Final): return statistics for
    - Flow EPE
    - Flow Outlier rate
    - Depth variation (dz) EPE
    - Depth variation Outlier rate
    Each as 1+iters length arrays (zero + predictions)
    """
    model.eval()
    val_dataset = datasets.Sintel_test(ifeval=typei, dstype=detype)

    epe_all, f1_all = [], []
    epe_dz_all, f1_dz_all = [], []

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, d1, d2, dz_gt, intrinsics, valid_gt = val_dataset[val_id]
        validd = (dz_gt < 1000)[0, :, :]
        image1, image2, d1, d2 = [x[None].cuda() for x in [image1, image2, d1, d2]]
        padder = InputPadder(image1.shape, mode='kitti', sp=32)
        image1, image2 = padder.pad(image1, image2)
        d1in, d2in = padder.pad(d1, d2)
        flow_gt, dz_gt = flow_gt.cuda(), dz_gt.cuda()
        valid = valid_gt.cuda() > 0.5
        vald = validd.cuda() > 0.5

        with torch.no_grad():
            flow_preds, dz_preds = model(image1, image2, d1in, d2in, iters=iters, test_mode=False)

        # ---- baseline: zero field ----
        zero_flow = torch.zeros_like(flow_gt)
        epe_zero = torch.sqrt(torch.sum((zero_flow - flow_gt) ** 2, dim=0))
        mag = torch.sqrt(torch.sum(flow_gt ** 2, dim=0))
        out_zero = ((epe_zero > 3.0) & ((epe_zero / (mag + 1e-6)) > 0.05)).float()
        epe_zero_mean = epe_zero[valid].mean().item()
        f1_zero_mean = 100 * out_zero[valid].mean().item()

        zero_dz = torch.zeros_like(dz_gt)
        epe_dz_zero = torch.abs(zero_dz - dz_gt)
        out_dz_zero = ((epe_dz_zero > 0.1) & ((epe_dz_zero / (torch.abs(dz_gt) + 1e-6)) > 0.05)).float()
        epe_dz_zero_mean = epe_dz_zero[0][vald].mean().item()
        f1_dz_zero_mean = 100 * out_dz_zero[0][vald].mean().item()

        epe_seq, f1_seq = [epe_zero_mean], [f1_zero_mean]
        epe_dz_seq, f1_dz_seq = [epe_dz_zero_mean], [f1_dz_zero_mean]

        # ---- per iteration ----
        for flow_pred, dz_pred in zip(flow_preds, dz_preds):
            flow_pred = padder.unpad(flow_pred)[0]
            dz_pred = padder.unpad(dz_pred)[0]

            # --- Flow ---
            epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=0))
            mag = torch.sqrt(torch.sum(flow_gt ** 2, dim=0))
            out = ((epe > 3.0) & ((epe / (mag + 1e-6)) > 0.05)).float()

            epe_seq.append(epe[valid].mean().item())
            f1_seq.append(100 * out[valid].mean().item())

            # --- Depth Variation ---
            epe_dz = torch.abs(dz_pred - dz_gt)
            out_dz = ((epe_dz > 0.1) & ((epe_dz / (torch.abs(dz_gt) + 1e-6)) > 0.05)).float()

            epe_dz_seq.append(epe_dz[0][vald].mean().item())
            f1_dz_seq.append(100 * out_dz[0][vald].mean().item())

        epe_all.append(epe_seq)
        f1_all.append(f1_seq)
        epe_dz_all.append(epe_dz_seq)
        f1_dz_all.append(f1_dz_seq)

    # ---- Average over all samples ----
    epe_all = np.array(epe_all).mean(axis=0)
    f1_all = np.array(f1_all).mean(axis=0)
    epe_dz_all = np.array(epe_dz_all).mean(axis=0)
    f1_dz_all = np.array(f1_dz_all).mean(axis=0)

    # ---- Print summary ----
    print(f"\n========= Sintel ({detype.capitalize()}) Evaluation =========")
    print("Iter | 0(Zero) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8")
    print("Flow-EPE  | " + " | ".join(f"{v:.4f}" for v in epe_all))
    print("Flow-F1(%)| " + " | ".join(f"{v:.2f}" for v in f1_all))
    print("DZ-EPE    | " + " | ".join(f"{v:.4f}" for v in epe_dz_all))
    print("DZ-F1(%)  | " + " | ".join(f"{v:.2f}" for v in f1_dz_all))
    print("====================================\n")

    return {
        "flow_epe": epe_all,
        "flow_f1": f1_all,
        "dz_epe": epe_dz_all,
        "dz_f1": f1_dz_all
    }

def validate_kitti_sequence(model, iters=8):
    """
    Evaluate KITTI sequence: return statistics for
    - Flow EPE
    - Flow Outlier rate
    - Depth variation (dz) EPE
    - Depth variation Outlier rate
    Each as 1+iters length arrays (zero + 8 predictions)
    """
    model.eval()
    val_dataset = datasets.kitti_test()

    epe_all, f1_all = [], []
    epe_dz_all, f1_dz_all = [], []

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, d1, d2, dz_gt, intrinsics, valid_gt = val_dataset[val_id]
        image1, image2, d1, d2 = [x[None].cuda() for x in [image1, image2, d1, d2]]
        padder = InputPadder(image1.shape, mode='kitti', sp=32)
        image1, image2 = padder.pad(image1, image2)
        d1in, d2in = padder.pad(d1, d2)
        flow_gt, dz_gt = flow_gt.cuda(), dz_gt.cuda()
        valid = valid_gt.cuda() > 0.5

        with torch.no_grad():
            flow_preds, dz_preds = model(image1, image2, d1in, d2in, iters=iters, test_mode=False)

        # ---- baseline: zero field ----
        zero_flow = torch.zeros_like(flow_gt)
        epe_zero = torch.sqrt(torch.sum((zero_flow - flow_gt) ** 2, dim=0))
        mag = torch.sqrt(torch.sum(flow_gt ** 2, dim=0))
        out_zero = ((epe_zero > 3.0) & ((epe_zero / (mag + 1e-6)) > 0.05)).float()
        epe_zero_mean = epe_zero[valid].mean().item()
        f1_zero_mean = 100 * out_zero[valid].mean().item()

        zero_dz = torch.zeros_like(dz_gt)
        epe_dz_zero = torch.abs(zero_dz - dz_gt)
        out_dz_zero = ((epe_dz_zero > 0.1) & ((epe_dz_zero / (torch.abs(dz_gt) + 1e-6)) > 0.05)).float()
        epe_dz_zero_mean = epe_dz_zero[0][valid].mean().item()
        f1_dz_zero_mean = 100 * out_dz_zero[0][valid].mean().item()

        epe_seq, f1_seq = [epe_zero_mean], [f1_zero_mean]
        epe_dz_seq, f1_dz_seq = [epe_dz_zero_mean], [f1_dz_zero_mean]

        # ---- per iteration ----
        for flow_pred, dz_pred in zip(flow_preds, dz_preds):
            flow_pred = padder.unpad(flow_pred)[0]
            dz_pred = padder.unpad(dz_pred)[0]

            # --- Flow ---
            epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=0))
            mag = torch.sqrt(torch.sum(flow_gt ** 2, dim=0))
            out = ((epe > 3.0) & ((epe / (mag + 1e-6)) > 0.05)).float()

            epe_seq.append(epe[valid].mean().item())
            f1_seq.append(100 * out[valid].mean().item())

            # --- Depth Variation ---
            epe_dz = torch.abs(dz_pred - dz_gt)
            out_dz = ((epe_dz > 0.1) & ((epe_dz / (torch.abs(dz_gt) + 1e-6)) > 0.05)).float()

            epe_dz_seq.append(epe_dz[0][valid].mean().item())
            f1_dz_seq.append(100 * out_dz[0][valid].mean().item())

        epe_all.append(epe_seq)
        f1_all.append(f1_seq)
        epe_dz_all.append(epe_dz_seq)
        f1_dz_all.append(f1_dz_seq)

    # ---- Average over all samples ----
    epe_all = np.array(epe_all).mean(axis=0)
    f1_all = np.array(f1_all).mean(axis=0)
    epe_dz_all = np.array(epe_dz_all).mean(axis=0)
    f1_dz_all = np.array(f1_dz_all).mean(axis=0)

    # ---- Print summary ----
    print("\n========= KITTI Evaluation =========")
    print("Iter | 0(Zero) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8")
    print("Flow-EPE  | " + " | ".join(f"{v:.4f}" for v in epe_all))
    print("Flow-F1(%)| " + " | ".join(f"{v:.2f}" for v in f1_all))
    print("DZ-EPE    | " + " | ".join(f"{v:.4f}" for v in epe_dz_all))
    print("DZ-F1(%)  | " + " | ".join(f"{v:.2f}" for v in f1_dz_all))
    print("====================================\n")

    return {
        "flow_epe": epe_all,
        "flow_f1": f1_all,
        "dz_epe": epe_dz_all,
        "dz_f1": f1_dz_all
    }

def validate_kitti(model, iters=6):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.kitti_test()
    time_all = 0
    out_list, epe_list = [], []
    outdz_list, epedz_list = [], []
    for val_id in range(len(val_dataset)):
        print(val_id)
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

        dz_gtS = dz_gt.detach().cpu().numpy()
        dz_pres = dz_pre[0].detach().cpu().numpy()

        flowpre = flow[0].permute(1,2,0).detach().cpu().numpy()
        flow_gts = flow_gt.permute(1,2,0).detach().cpu().numpy()


        #不行了这个光流必须搞个采样了，这他妈的数据集有问题啊
        epedz = torch.sum((dz_gt - (dz_pre[0])) ** 2, dim=0).sqrt()
        magdz = torch.sum(dz_gt** 2, dim=0).sqrt()
        '''
        dzshow = epedz
        dzshow = (dzshow*0.5).clip(0,1)
        cmap = plt.get_cmap('jet')
        # 将灰度值映射到RGBA伪彩色
        dzout_color = cmap(dzshow)[:, :, :3]  # 取前三通道 (RGB)
        # 将 mask==0 的区域置为黑色
        dzout_color[valid<1] = 1.0
        plt.imshow(dzout_color)
        plt.axis('off')
        '''

        epedz = epedz.view(-1)
        magdz = magdz.view(-1)

        epe = torch.sum((flow[0] - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        outdz = ((epedz > 0.1) & ((epedz / magdz) > 0.05)).float()
        '''
        f1dz = np.mean(epedz[val].cpu().numpy())
        
        # 左下角显示误差（白色加粗）
        plt.text(
            15,  # x 偏移
            65,  # y 位置
            f'EPE : {f1dz:.2f}',  # 文本内容
            color='black',
            fontsize=14,
            fontweight='bold',
            va='bottom',
            ha='left',
            bbox=dict(
                facecolor='lightgray',  # 背景颜色，可以改成 'lightgray'
                alpha=0.5,  # 透明度
                edgecolor='none',  # 去掉边框
                boxstyle='round,pad=0.3'  # 圆角和内边距
            )
        )
        save_name = save_root+str(val_id).zfill(4) +'a.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        #todo 这里存深度变化率图会不会好一点
        save_nameb = save_root + str(val_id).zfill(4) + 'b.png'
        plt.imshow(image1[0].permute(1,2,0).detach().cpu().numpy()/255)
        plt.axis('off')
        plt.savefig(save_nameb, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        

        save_namec = save_root + str(val_id).zfill(4) + 'c.png'
        res1 = np.concatenate((dz_pres[0], dz_gtS[0]), axis=0)

        # 假设 res1 是 (H,W)，mask 同形状
        norm_res1 = (res1 - np.min(res1)) / (np.max(res1) - np.min(res1) + 1e-8)  # 归一化到 [0,1]
        # 使用 colormap 映射为 RGB
        cmap = cm.get_cmap('plasma')
        res1_rgb = cmap(norm_res1)[:, :, :3]  # (H,W,3)，值在 [0,1]
        h,w = valid.shape
        # mask 为 False 的地方置黑
        res1_rgb[h:,:,:][valid<1] = 1.0
        plt.imshow(res1_rgb,cmap='plasma')
        plt.axis('off')
        plt.savefig(save_namec, dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close('all')
        '''
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
    output_filenamerootfv= '/home/lh/CSCV_occ_new/submit_preRF3D627/flow_viz'
    if os.path.exists(output_filenamerootfv) == False:
        os.makedirs(output_filenamerootfv)
    output_filenamerootf= '/home/lh/CSCV_occ_new/submit_preRF3D627/flow'
    if os.path.exists(output_filenamerootf) == False:
        os.makedirs(output_filenamerootf)
    output_filenamerootd1= '/home/lh/CSCV_occ_new/submit_preRF3D627/disp_0'
    if os.path.exists(output_filenamerootd1) == False:
        os.makedirs(output_filenamerootd1)
    output_filenamerootd2= '/home/lh/CSCV_occ_new/submit_preRF3D627/disp_1'
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
    #parser.add_argument('--itr1', type=int, default=6)
    parser.add_argument('--itr1', type=int, default=6)
    parser.add_argument('--itr2', type=int, default=4)
    parser.add_argument('--it1', type=int, default=2)
    parser.add_argument('--it2', type=int, default=4)
    parser.add_argument('--it3', type=int, default=3)
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
        validate_sintel(model, iters=6, typei='t_all', detype='final')
        #kitti_sceneflow_submission(model.module)
        #Demo_vis_dir(model.module)
        #Demo_runtime(model.module)
        #validate_kitti(model.module, iters=6)
        #validate_kitti_sequence(model.module, iters=8)
        #validate_sintel_sequence(model.module,typei= 't_test')
        #sintel_show(model.module,6, typei='t_show')

