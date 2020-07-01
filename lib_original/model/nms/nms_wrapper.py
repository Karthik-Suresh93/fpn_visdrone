# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from model.utils.config import cfg
from model.nms.nms_gpu import nms_gpu
import sys;sys.path.append("/home/ksuresh/fpn.pytorch-master/soft-nms/lib/nms")
from cpu_nms import cpu_nms, cpu_soft_nms
import numpy as np


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        print("dets shape is 0...!!!!!!!!!!!!!!!!!!!!!!!")
        return []
    # ---numpy version---
    # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    # ---pytorch version---
    if not force_cpu:
        return nms_gpu(dets, thresh)
    else:
        dets = dets.numpy()
        keep=cpu_nms(dets, thresh)
        return torch.from_numpy(np.array(keep)).float().to("cuda:1")  #converting to float tensor tensor 


def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
    device_id = dets.get_device()
    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return torch.from_numpy(np.array(keep)).float().to('cuda:{}'.format(device_id))   #converting to cuda tensor on device_id of dets...(the only gpu available)
