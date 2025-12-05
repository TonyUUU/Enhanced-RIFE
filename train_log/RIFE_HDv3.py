import warnings

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from train_log.IFNet_HDv3 import *
import torch.nn.functional as F
from model.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        # determine device for this instance (supports CPU, single-GPU and distributed GPU)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            if local_rank is not None and local_rank >= 0:
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # build network and move to device first
        self.flownet = IFNet()
        self.flownet.to(self.device)

        # wrap with DDP only when torch.distributed is initialized
        if dist.is_available() and dist.is_initialized() and local_rank is not None and local_rank >= 0:
            # ensure model is on the correct device before wrapping
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

        # optimizers and losses (parameters already on correct device)
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE().to(self.device)
        # self.vgg = VGGPerceptualLoss().to(self.device)
        self.sobel = SOBEL().to(self.device)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        # keep for compatibility: move model to stored device
        # if wrapped in DDP, .to will move the underlying module appropriately
        try:
            # handle DDP wrapper
            if isinstance(self.flownet, DDP):
                self.flownet.module.to(self.device)
            else:
                self.flownet.to(self.device)
        except Exception as e:
            warnings.warn(f"Failed to move model to device: {e}", RuntimeWarning)

    def load_model(self, path, rank=0):
        def convert(param):
            # remove 'module.' prefix if present (handles models saved from DDP)
            return {k.replace("module.", ""): v for k, v in param.items()}
        model_path = f'{path}/flownet.pkl'
        if rank <= 0:
            if torch.cuda.is_available() and 'cuda' in str(self.device):
                state = torch.load(model_path)
            else:
                state = torch.load(model_path, map_location='cpu')
            try:
                self_state = self.flownet.state_dict()
                state = convert(state)
                self.flownet.load_state_dict(state, strict=False)
            except Exception:
                # fallback: try direct load (may raise clearer error)
                self.flownet.load_state_dict(state)

    def save_model(self, path, rank=0):
        if rank == 0:
            # save the underlying module state_dict when using DDP
            if isinstance(self.flownet, DDP):
                torch.save(self.flownet.module.state_dict(), f'{path}/flownet.pkl')
            else:
                torch.save(self.flownet.state_dict(), f'{path}/flownet.pkl')

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, scale_list)
        return merged[2]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        scale = [4, 2, 1]
        flow, mask, merged = self.flownet(imgs, scale_list=scale)
        loss_l1 = (merged[2] - gt).abs().mean()
        loss_smooth = self.sobel(flow[2], flow[2]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'mask': mask,
            'flow': flow[2][:, :2],
            'loss_l1': loss_l1,
            # 'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }
