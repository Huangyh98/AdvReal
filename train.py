"""
Training code for Adversarial patch training
"""
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import sys
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import gc
from load_data import *
from transformers import DeformableDetrForObjectDetection
import torch
from torchvision import transforms
import torchvision
from tensorboardX import SummaryWriter
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from utils.parser import ConfigParser
from attack.attacker import UniversalAttacker
from utils.loader import dataLoader
from utils.parser import logger
from utils.plot import VisualBoard
from torch.utils.data import DataLoader
from utils.utils import *
sys.path.append(os.path.abspath(''))
from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from render import ImageRenderer
def init(detector_attacker: UniversalAttacker, cfg: ConfigParser, data_root: str, args: object =None, log: bool =True):
    if log: logger(cfg, args)

    data_sampler = None
    data_loader_tsea = dataLoader(data_root,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler, shuffle=True)

    detector_attacker.init_universal_patch(args.patch)
    detector_attacker.init_attaker()

    vlogger = None
    if log and args and not args.debugging:
        vlogger = VisualBoard(name=args.board_name, new_process=args.new_process,
                              optimizer=detector_attacker.attacker)
        detector_attacker.vlogger = vlogger

    return data_loader_tsea, vlogger

def collate_fn(batch):
    return batch
def get_nuscenes_loader(img_dir, batch_size=4, shuffle=True, num_workers=4, transform=None):
    dataset = NuScenesDataset(img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return loader
# add path for demo utils functions 


class PatchTrainer(object):
    def __init__(self, args):
        self.args = args
        self.renderer_v3 = ImageRenderer(args) 
        if args.device is not None:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
        else:
            device = None
        self.device = device
        self.img_size = 416
        self.DATA_DIR = "./data"

        if args.arch == "rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        elif args.arch == "yolov3":
            self.model = YOLOv3Darknet().eval().to(device)
            self.model.load_darknet_weights('arch/weights/yolov3.weights')
        elif args.arch == "detr":
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).eval().to(
                device)
        elif args.arch == "deformable-detr":
            self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
        elif args.arch == "yolov2":
            self.model = Darknet('yolo2/cfg/yolov2.cfg').eval().to(device)
            self.model.load_weights('yolo2/yolov2.weights')
        elif args.arch == "yolov5":
            from detlib.HHDet.yolov5.api import HHYolov5
            cfg = ConfigParser("configs/baseline/v5.yaml")
            detector_cfg = cfg.DETECTOR 
            input_size = self.img_size
            self.model = HHYolov5(name="YOLOV5", 
                                cfg=detector_cfg,  
                                input_tensor_size=input_size,
                                device=device)
            model_weights = 'detlib/HHDet/yolov5/yolov5/weight/yolov5s.pt'
            model_config = 'detlib/HHDet/yolov5/yolov5/models_v5/yolov5s.yaml'
            self.model.load(model_weights, model_config=model_config)
            self.model.eval()
            self.model_parameters = self.model.parameters()
        elif args.arch == "mask_rcnn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        else:
            raise NotImplementedError

        for p in self.model.parameters():
            p.requires_grad = False

        self.batch_size = args.batch_size

        self.patch_transformer = PatchTransformer().to(device)
        if args.arch == "rcnn":
            self.prob_extractor = MaxProbExtractor(0, 80).to(device)
        elif args.arch == "yolov2":
            self.prob_extractor = YOLOv2MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "yolov3":
            self.prob_extractor = YOLOv3MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "deformable-detr":
            self.prob_extractor = DeformableDetrProbExtractor(0,80,self.img_size).to(device)
        elif args.arch == "yolov5":
            self.prob_extractor = YOLOv5MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        self.tv_loss = TotalVariation()

        self.train_loader = get_nuscenes_loader(
            img_dir='data/background_trans/background_train_resize',  # 根据您的目录结构修改
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            transform=transforms.ToTensor()
        )

        self.epoch_length = len(self.train_loader)


        color_transform = ColorTransform('color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        self.fig_size_H = 340
        self.fig_size_W = 864

        self.fig_size_H_t = 484
        self.fig_size_W_t = 700

        resolution = 4
        h, w, h_t, w_t = int(self.fig_size_H / resolution), int(self.fig_size_W / resolution), int(self.fig_size_H_t / resolution), int(self.fig_size_W_t / resolution)
        self.h, self.w, self.h_t, self.w_t = h, w, h_t, w_t

        # Set paths
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")

        self.coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
        self.coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t), torch.arange(w_t)), -1).to(device)

        self.colors = torch.load("data/camouflage4.pth").float().to(device)
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
        self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]

        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
        self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]


    def get_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(InriaDataset(img_dir, self.img_size, shuffle=shuffle),
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=4)
        return loader

    def init_tensorboard(self, name=None):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        print(time_str)
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        fname = self.args.save_path.split('/')[-1]
        return SummaryWriter(f'{self.args.patch_save_dir}/{TIMESTAMP}_{fname}')


    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        self.writer = self.init_tensorboard()
        args = self.args
        
        et0 = time.time()
        checkpoints = args.checkpoints
        cfg = ConfigParser(args.cfg)
        detector_attacker = UniversalAttacker(cfg, self.device)
        data_root = cfg.DATA.TRAIN.IMG_DIR
        data_loader_tsea, vlogger = init(detector_attacker, cfg, args=args, data_root=data_root)
        patch = detector_attacker.universal_patch
        patch.requires_grad_(True)
        optimizer = optim.Adam([patch], lr=args.lr, amsgrad=True)

        self.writer = self.init_tensorboard()
        for epoch in tqdm(range(checkpoints, args.nepoch)):
            if epoch % 100 == 90:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / args.lr_decay
                print("Updated learning rate:", optimizer.param_groups[0]['lr'])
            ep_3d_det_loss = 0
            ep_patch_loss = 0
            ep_patch_det_loss = 0
            ep_patch_tv_loss = 0
            ep_loss = 0
            eff_count = 0
            eff_count_patch = 0  
            data_iter_tsea = iter(data_loader_tsea)
            for i_batch, img_batch in enumerate(self.train_loader):
                optimizer.zero_grad()
                t0 = time.time()
                try:
                    
                    img_tensor_batch = next(data_iter_tsea)
                    
                except StopIteration:
                    data_iter_tsea = iter(data_loader_tsea)
                    img_tensor_batch = next(data_iter_tsea)

                detector_attacker.universal_patch.to(self.device)

                img_tensor_batch = img_tensor_batch.to(detector_attacker.device, non_blocking=True)

                all_preds = detector_attacker.detect_bbox(img_tensor_batch)

                target_nums = detector_attacker.get_patch_pos_batch(all_preds)

                if sum(target_nums) == 0: continue
                patch_loss, patch_tv_loss, patch_det_loss = detector_attacker.attack(img_tensor_batch, mode='optim')
                eff_count_patch += 1

                patch_c = patch.clone()
                patch_c = patch_c.clamp(0, 1)
                self.renderer_v3.set_adv_patch_texture(patch_c)
                all_composite_images = []
                all_gts = []
                
                for bg_idx, bg_image_tensor in enumerate(img_batch):
                    composite_images, gts = self.renderer_v3.generate_composite_image_tensor(bg_image_tensor)  # composite_images: List of tensors
                    all_composite_images.extend(composite_images) 
                    all_gts.extend(gts)
                p_img_batch = torch.stack(all_composite_images).to(self.device) 
                p_img_batch = p_img_batch[:, :3, :, :]
                gts_batch = torch.stack(all_gts).to(self.device)
                t1 = time.time()
                normalize = True
                if self.args.arch == "deformable-detr" and normalize:
                    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                    p_img_batch = normalize(p_img_batch)
                output = self.model(p_img_batch)
                t2 = time.time()

                det_loss, max_prob_list = self.prob_extractor(
                    output,
                    gts_batch,
                    loss_type=args.loss_type,
                    iou_thresh=args.train_iou
                )
                eff_count += 1
                loss = 0

                # if epoch < 100:
                #     loss = patch_det_loss + patch_tv_loss
                # else:
                #     loss = det_loss + patch_det_loss + patch_tv_loss

                loss += det_loss
                loss += patch_det_loss
                loss += patch_tv_loss


                ep_patch_loss += patch_loss.item()
                ep_patch_det_loss += patch_det_loss.item()
                ep_patch_tv_loss += patch_tv_loss.item()
                ep_3d_det_loss += det_loss.item()
                ep_loss += loss.item()
                
                loss.backward()
                optimizer.step()                
                patch.clamp(0, 1)
                    
                if i_batch % 10 == 0:
                    global_step = epoch * len(self.train_loader) + i_batch
                    self.writer.add_scalar('batch/3D_DET_loss', det_loss.item(), global_step)
                    self.writer.add_scalar('batch/Total_loss', loss.item(), global_step)
                    self.writer.add_scalar('batch/2D_DET_loss', patch_det_loss.item(), global_step)
                    self.writer.add_scalar('batch/2D_TV_loss', patch_tv_loss.item(), global_step)
                del patch_loss, patch_det_loss, patch_tv_loss, det_loss, img_batch, img_tensor_batch, all_composite_images, all_gts, p_img_batch, gts_batch, gts
                gc.collect()
                
            del patch_c, composite_images
            gc.collect()

            et1 = time.time()
            ep_patch_loss = ep_patch_loss / eff_count_patch
            ep_patch_det_loss = ep_patch_det_loss / eff_count_patch
            ep_patch_tv_loss = ep_patch_tv_loss / eff_count_patch
            ep_3d_det_loss = ep_3d_det_loss / eff_count
            ep_loss = ep_loss / eff_count
            
            print(' EPOCH: ', epoch),
            print("##################### AdvReal_2D #####################")
            print('2D DET LOSS: ', ep_patch_det_loss) 
            print(' 2D TV LOSS: ', ep_patch_tv_loss) 
            print("##################### AdvReal_3D #####################")
            print(' 3D DET LOSS: ', ep_3d_det_loss)
            print("#####################   AdvReal  #####################")
            print('  EPOCH TIME: ', et1 - et0)
            print('  EPOCH LOSS: ', ep_loss)
            self.writer.add_scalar('epoch/3D_DET_loss', ep_3d_det_loss, epoch)
            self.writer.add_scalar('epoch/2D_DET_loss', ep_patch_det_loss, epoch)
            self.writer.add_scalar('epoch/2D_TV_loss', ep_patch_tv_loss, epoch)
            self.writer.add_scalar('epoch/Total_loss', ep_loss, epoch)
            et0 = time.time()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    print('Version 2.0')
    print(torch.__version__)
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--device', default='cuda:0', help='')
    parser.add_argument('--lr', type=float, default=0.03, help='')
    parser.add_argument('--lr_seed', type=float, default=0.01, help='')
    parser.add_argument('--nepoch', type=int, default=800, help='')
    parser.add_argument('--checkpoints', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--save_path', default='results/demo', help='')
    parser.add_argument("--tv_loss", type=float, default=1, help='tv loss weight')
    parser.add_argument("--real_loss", type=float, default=0.5, help='real loss weight')
    parser.add_argument("--patch_loss", type=float, default=0.5, help='patch loss weight')
    parser.add_argument("--lr_decay", type=float, default=1.1, help='')
    parser.add_argument("--lr_decay_seed", type=float, default=2, help='')
    parser.add_argument("--arch", type=str, default="yolov2", help='deformable-detr')
    parser.add_argument("--seed_type", default='fixed', help='')
    parser.add_argument("--clamp_shift", type=float, default=0, help='')
    parser.add_argument("--resample_type", default=None, help='')
    parser.add_argument("--tps2d_range_t", type=float, default=50.0, help='')
    parser.add_argument("--tps2d_range_r", type=float, default=0.1, help='')
    parser.add_argument("--tps3d_range", type=float, default=0.15, help='')
    parser.add_argument("--disable_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--train_iou", type=float, default=0.45, help='')
    parser.add_argument("--mode", default='paper_obj', help='Patterns generated in adyolo')
    parser.add_argument("--patch_save_dir", default='demo', help='The generation path of the patch in adyolo')
    parser.add_argument('-cfg', '--cfg', type=str, default=os.path.join(os.getcwd(), 'configs/baseline/v2.yaml'), help="A relative path of the .yaml proj config file.")
    parser.add_argument('-p', '--patch', type=str, default='texture/heart.png', help="Start training with a given patch instead of random init. (for training from a break-point or for fine-tune)")
    parser.add_argument('-d', '--debugging', action='store_true', help="Will not start tensorboard process if debugging=True.")
    parser.add_argument('-sp', '--save_process', action='store_true', default=True, help="Save patches from intermediate epoches.")
    parser.add_argument('-n', '--board_name', type=str, default=None, help="Name of the Tensorboard as well as the patch name.")
    parser.add_argument('-np', '--new_process', action='store_true', default=False, help="Start new TensorBoard server process.")


    args = parser.parse_args()
    assert args.seed_type in ['fixed', 'random', 'variable', 'langevin']

    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Train info:", args)
    trainer = PatchTrainer(args)
    trainer.train()
