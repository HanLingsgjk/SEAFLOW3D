# TPCV
This repository contains the source code for our paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create -n seaflow3d python=3.9
conda activate seaflow3d
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install matplotlib==3.5
pip install opencv-python==4.8.1.78
pip install tqdm
pip install pypng
pip install scipy
pip install einops
pip install tensorboard
pip install timm==0.6.13
pip install scikit-image==0.21.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0
pip install h5py

cd alt_cuda_corr/
pip install -e . --no-build-isolation

#for flowmatching 
pip install git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38
pip install open3d


#for mono flow
pip install git+https://github.com/microsoft/MoGe.git

```

## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.

#from core.raft_scale_flow_343_S2 import RAFT343used
nohup python train.py --name raft-driving_567_orin_r4rs1_S2 --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 1 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs1_S2_THINGS.txt 2>&1 &
#from core.raft_scale_flow_343 import RAFT343used
nohup python train.py --name raft-driving_567_orin_r4rs1_FF --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 1 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs1_THINGS_FF.txt 2>&1 &
#from core.raft_scale_flow_343 import RAFT343used
nohup python train.py --name raft-driving_567_orin_r3rs1_FF --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 1 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r3rs1_THINGS_FF.txt 2>&1 &

#from core.raft_scale_flow_343 import RAFT343used
nohup python train.py --name raft-driving_567_orin_r4rs0_FF --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 1 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs0_THINGS_FF.txt 2>&1 &
#from core.raft_scale_flow_343 import RAFT343used
nohup python train.py --name raft-driving_567_orin_r4rs2_FF --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 1 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs2_THINGS_FF.txt 2>&1 &

#from core.raft_scale_xr3 import RAFT343used3
nohup python train.py --name raft-driving_196_orin_r4rs1_FF --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs1_THINGS_FS.txt 2>&1 &
#from core.raft_scale_allin import RAFT343used1
nohup python train.py --name raft-driving_796_orin_r4rs1_FF --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 1 --num_steps 80000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs1_796THINGS_FF.txt 2>&1 &





nohup python train.py --name raft-driving_343_orin_r4rs1_allin --stage things --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/200000_raft-things.pth --gpus 0 --num_steps 60000 --batch_size 3 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.85 > myout_r4rs1_allin_THINGS.txt 2>&1 &
nohup python train.py --name raft-kitti_343_orin_r4rs1_FFALL --stage kitti --validation kitti --restore_ckpt /home/xuxian/RAFT3D/checkpoints/raft-driving_567_orin_r4rs1_TT.pth --gpus 0 1 --num_steps 60000 --batch_size 6 --lr 0.000125 --image_size 320 960 --wdecay 0.0001 --gamma=0.85 > myout_r4rs1_FF_kittiALL.txt 2>&1 &