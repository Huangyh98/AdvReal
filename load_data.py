import fnmatch
import math
import os
import sys
import time
from operator import itemgetter
import gc
import numpy as np
from numpy.random import randn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import utils_camou
import numpy as np
from arch.yolov3_models import YOLOv3Darknet
import fnmatch
import math
import os
import sys
import time
from operator import itemgetter
import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    FoVPerspectiveCameras,
    Textures
)
from pytorch3d.renderer.cameras import look_at_view_transform

class MaxProbExtractor(nn.Module):
    def __init__(self, cls_id, num_cls):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = lambda obj, cls: obj
    def forward(self, output, gt, loss_type, iou_thresh):
        det_loss = []
        max_probs = []
        num = 0
        for i, boxes in enumerate(output):
            ious = torchvision.ops.box_iou(boxes['boxes'], gt[i].unsqueeze(0)).squeeze(1)
            mask = ious.ge(iou_thresh)
            if True:
                mask = mask.logical_and(boxes['labels'] == 1)
            ious = ious[mask]
            scores = boxes['scores'][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        return det_loss, max_probs

class YOLOv2MaxProbExtractor(nn.Module):
    def __init__(self, cls_id, num_cls, model, figsize):
        super(YOLOv2MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize
        self.model = model
    def forward(self, YOLOoutputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        box_all = utils_camou.get_region_boxes_general(YOLOoutputs, self.model, conf_thresh=0.2, name="yolov2")
        for i in range(len(box_all)):
            boxes = box_all[i]
            assert boxes.shape[1] == 7, f"Expected 7 elements per box, got {boxes.shape[1]}"
            x_center, y_center, width, height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            w1 = x_center - width / 2
            h1 = y_center - height / 2
            w2 = x_center + width / 2
            h2 = y_center + height / 2
            bbox = torch.stack([w1, h1, w2, h2], dim=-1)
            ious = torchvision.ops.box_iou(bbox.view(-1, 4) * self.figsize, gt[i].unsqueeze(0)).squeeze(-1)
            if ious.numel() == 0:
                ious = torch.zeros_like(ious)
            mask = ious.ge(iou_thresh) & (boxes[..., 6] == self.cls_id)
            ious = ious[mask]
            scores = boxes[..., 4][mask]
            if ious.numel() > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(ious[ids])
                    max_probs.append(ious[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    max_score, _ = torch.max(boxes[..., 4], dim=0)
                    det_loss.append(max_score)
                    max_probs.append(max_score)
                    num += 1
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")
            else:
                det_loss.append(torch.tensor(0.0).to(YOLOoutputs.device))
                max_probs.append(torch.tensor(0.0).to(YOLOoutputs.device))
        if len(det_loss) > 0:
            det_loss = torch.stack(det_loss).mean()
        else:
            det_loss = torch.tensor(0.0).to(YOLOoutputs.device)
        max_probs = torch.stack(max_probs)
        return det_loss, max_probs

class YOLOv3MaxProbExtractor(nn.Module):
    def __init__(self, cls_id, num_cls, model,figsize):
        super(YOLOv3MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize
        self.model = model
    def forward(self, YOLOoutputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        box_all = utils_camou.get_region_boxes_general(YOLOoutputs, self.model, conf_thresh=0.2, name="yolov3")
        for i in range(len(box_all)):
            boxes = box_all[i]
            assert boxes.shape[1] == 7
            w1 = boxes[...,0] - boxes[..., 2]/2
            h1 = boxes[...,1] - boxes[..., 3]/2
            w2 = boxes[...,0] + boxes[..., 2]/2
            h2 = boxes[...,1] + boxes[..., 3]/2
            bbox = torch.stack([w1,h1,w2,h2],dim=-1)
            ious = torchvision.ops.box_iou(bbox.view(-1,4)*self.figsize,gt[i].unsqueeze(0)).squeeze(-1)
            mask = ious.ge(iou_thresh)
            if True:
                mask = mask.logical_and(boxes[...,6]==0)
            ious = ious[mask]
            scores = boxes[...,4][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        if len(det_loss) > 0:
            det_loss = torch.stack(det_loss).mean()
        else:
            det_loss = torch.tensor(0.0).to(YOLOoutputs.device)
        max_probs = torch.stack(max_probs)
        return det_loss, max_probs

class YOLOv5MaxProbExtractor(nn.Module):
    def __init__(self, cls_id, num_cls, model, figsize):
        super(YOLOv5MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize
        self.model = model
    def forward(self, YOLOoutputs, gt, loss_type, iou_thresh):
        det_loss = []
        max_probs = []
        num = 0
        box_all = utils_camou.get_region_boxes_general(YOLOoutputs, self.model, conf_thresh=0.2, name="yolov5")
        for i in range(len(box_all)):
            boxes = box_all[i]
            w_center = boxes[...,0]
            h_center = boxes[...,1]
            w_width = boxes[...,2]
            h_height = boxes[...,3]
            conf = boxes[...,4]
            cls_idx = boxes[...,6].long()
            w1 = w_center - w_width/2
            h1 = h_center - h_height/2
            w2 = w_center + w_width/2
            h2 = h_center + h_height/2
            bbox = torch.stack([w1,h1,w2,h2],dim=-1)
            ious = torchvision.ops.box_iou(bbox.view(-1,4)*self.figsize, gt[i].unsqueeze(0)).squeeze(-1)
            mask = ious.ge(iou_thresh)
            attack_cls = int(0)
            mask = mask & (cls_idx == attack_cls)
            valid_ious = ious[mask]
            valid_scores = conf[mask]
            if valid_scores.numel() > 0:
                if loss_type == 'max_iou':
                    chosen_score = valid_scores.max()
                    det_loss.append(chosen_score)
                    max_probs.append(chosen_score)
                    num += 1
                elif hasattr(self, 'cfg') and hasattr(self.cfg, 'topx_conf'):
                    topx = self.cfg.topx_conf
                    sorted_scores, _ = torch.sort(valid_scores, descending=True)
                    top_scores = sorted_scores[:topx]
                    mean_conf = top_scores.mean()
                    det_loss.append(mean_conf)
                    max_probs.append(mean_conf)
                    num += top_scores.size(0)
                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        return det_loss, max_probs

class DeformableDetrProbExtractor(nn.Module):
    def __init__(self, cls_id, num_cls, figsize):
        super(DeformableDetrProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize
    def forward(self, outputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        logits = outputs['logits'][...,1]
        prob = F.softmax(outputs['logits'],dim=-1)[...,1]
        labels = torch.argmax(outputs['logits'],dim=-1)
        batch = prob.shape[0]
        for i in range(batch):
            bbox = outputs['pred_boxes'][i]
            w1 = bbox[...,0] - bbox[..., 2]/2
            h1 = bbox[...,1] - bbox[..., 3]/2
            w2 = bbox[...,0] + bbox[..., 2]/2
            h2 = bbox[...,1] + bbox[..., 3]/2
            bboxes = torch.stack([w1,h1,w2,h2],dim=-1)
            ious = torchvision.ops.box_iou(bboxes.view(-1,4).detach()*self.figsize,gt[i].unsqueeze(0)).squeeze(-1)
            mask = ious.ge(iou_thresh)
            mask = mask.logical_and(labels[i] == 1)
            ious = ious[mask]
            scores = prob[i][mask]
            logit = logits[i][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_iou_mtconf':
                    _, ids = torch.max(ious*scores,dim=0)
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_logit':
                    _, ids = torch.max(ious,dim=0)
                    det_loss.append(logit[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)
                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        return det_loss, max_probs

class NPSCalculator(nn.Module):
    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)
    def forward(self, adv_patch):
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        color_dist_prod = torch.min(color_dist, 0)[0]
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)
    def get_printability_array(self, printability_file, side):
        printability_list = []
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)
        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa

class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()
    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

class TotalVariation_patch(nn.Module):
    def __init__(self):
        super(TotalVariation_patch, self).__init__()
    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.02
        self.min_scale = -0.28
        self.max_scale = 0.47
        self.translation_x = 0.8
        self.translation_y = 1.0
    def forward(self, img_batch, adv_patch):
        B, _, Ht, Wt = img_batch.shape
        _, _, Ho, Wo = adv_patch.shape
        adv_patch = adv_patch[:B]
        mask = (adv_patch[:, -1:, ...] > 0).to(adv_patch)
        adv_patch = adv_patch[:, :-1, ...]
        contrast = adv_patch.new(size=[B]).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = adv_patch.new(size=[B]).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise = adv_patch.new(adv_patch.shape).uniform_(-1, 1) * self.noise_factor
        adv_patch = adv_patch * contrast + brightness + noise
        adv_patch = adv_patch.clamp(0, 1)
        adv_patch = torch.cat([adv_patch, mask], dim=1)
        scale = adv_patch.new(size=[B]).uniform_(self.min_scale, self.max_scale).exp()
        mesh_bord = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        mesh_bord = mesh_bord / mesh_bord.new([Ho, Wo, Ho, Wo]) * 2 - 1
        pos_param = mesh_bord + mesh_bord.new([1, 1, -1, -1]) * scale.unsqueeze(-1)
        tymin, txmin, tymax, txmax = pos_param.unbind(-1)
        xdiff = (-txmax + txmin).clamp(min=0)
        xmiddle = (txmax + txmin) / 2
        ydiff = (-tymax + tymin).clamp(min=0)
        ymiddle = (tymax + tymin) / 2
        tx = txmin.new(txmin.shape).uniform_(-0.5, 0.5) * xdiff * self.translation_x + xmiddle
        ty = tymin.new(tymin.shape).uniform_(-0.5, 0.5) * ydiff * self.translation_y + ymiddle
        theta = adv_patch.new_zeros(B, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 0, 1] = 0
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, img_batch.shape)
        adv_batch = F.grid_sample(adv_patch, grid, padding_mode='zeros')
        mask = adv_batch[:, -1:]
        adv_batch = adv_batch[:, :-1] * mask + img_batch * (1 - mask)
        gt = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        gt = gt[:, [1, 0, 3, 2]].unbind(0)
        return adv_batch, gt

class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()
    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch

class InriaDataset(Dataset):
    def __init__(self, img_dir, imgsize, shuffle=True, if_square=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        self.len = n_images
        self.img_dir = img_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        self.if_square = if_square
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.pad_and_scale(image)
        transform = transforms.ToTensor()
        image = transform(image)
        return image
    def pad_and_scale(self, img):
        w, h = img.size
        if w==h:
            padded_img = img
        elif self.if_square:
            a = min(w, h)
            ww = (w - a) // 2
            hh = (h - a) // 2
            padded_img = img.crop([ww, hh, ww+a, hh+a])
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)
        return padded_img
    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

class NuScenesDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(NuScenesDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in self.img_names]
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGBA')
        if self.transform:
            image = self.transform(image)
        return image
