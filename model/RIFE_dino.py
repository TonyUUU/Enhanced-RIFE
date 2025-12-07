# Real-Time Intermediate Flow Estimation for Video Frame Interpolation
# Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou
# https://arxiv.org/abs/2011.06294

import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from model.loss import EPE, SOBEL, DinoLoss
from model.laplacian import LapLoss

from dino_config import DinoConfig
from model.dino_modules import DinoWrapper
from model.IFNet_dino import IFNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        self.dino_cfg = DinoConfig()
        self.dino = DinoWrapper(self.dino_cfg)
        self.flownet = IFNet(
            dino_in_channels=self.dino.embed_dim,
            dino_patch_size=self.dino.patch_size,
        )

        self.device()
        # use large weight decay may avoid NaN loss
        offset_params = []
        normal_params = []
        for name, param in self.flownet.named_parameters():
            if not param.requires_grad:
                continue
            if "conv_offset" in name:
                offset_params.append(param)
            else:
                normal_params.append(param)
        self.optimG = AdamW(
            [
                {"params": normal_params},
                {
                    "params": offset_params,
                    "lr_mult": self.dino_cfg.offset_lr_mult,
                },
            ],
            lr=1e-6,
            weight_decay=1e-3,
        )

        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        self.dino_loss = DinoLoss()
        if local_rank != -1:
            self.flownet = DDP(
                self.flownet, device_ids=[local_rank], output_device=local_rank
            )

    def train(self):
        self.flownet.train()
        self.dino.eval()

    def eval(self):
        self.flownet.eval()
        self.dino.eval()

    def device(self):
        self.flownet.to(device)
        self.dino.to(device)

    def load_model(self, path, rank=0):
        checkpoint = torch.load(
            "{}/flownet.pkl".format(path), map_location=device
        )
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        self.flownet.load_state_dict(new_state_dict)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(
                self.flownet.state_dict(), "{}/flownet.pkl".format(path)
            )

    def load_original_rife(self, path):
        checkpoint = torch.load(
            "{}/flownet.pkl".format(path), map_location=device
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint.items()
        }
        missing, unexpected = self.flownet.load_state_dict(
            state_dict, strict=False
        )
        print("missing: ", missing)
        print("unexpected: ", unexpected)

    def inference(
        self, img0, img1, scale=1, scale_list=None, TTA=False, timestep=0.5
    ):
        """
        Generates an intermediate frame (B, 3, H, W)
            between img0 and img1 (B, 3, H, W).
        """
        if scale_list is None:
            scale_list = [4, 2, 1]
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        feats0 = self.dino.get_features(img0, use_grad=False)
        feats1 = self.dino.get_features(img1, use_grad=False)
        dino_feats = (feats0, feats1)
        _, _, merged, _, _, _, _ = self.flownet(
            imgs, dino_feats, scale_list, timestep=timestep
        )
        if not TTA:
            return merged[2]
        else:
            dino_feats = (feats0.flip(2).flip(3), feats1.flip(2).flip(3))
            _, _, merged2, _, _, _, _ = self.flownet(
                imgs.flip(2).flip(3), dino_feats, scale_list, timestep=timestep
            )
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(
        self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None
    ):
        """
        Args:
            imgs (Tensor): Concatenated input pair. (B, 6, H, W)
            gt (Tensor): Ground Truth intermediate frame. (B, 3, H, W)
        Returns:
            tuple: (merged_image, info_dict)
        """
        for param_group in self.optimG.param_groups:
            mult = param_group.get("lr_mult", 1.0)
            param_group["lr"] = learning_rate * mult
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        feats0 = self.dino.get_features(img0, use_grad=False)
        feats1 = self.dino.get_features(img1, use_grad=False)
        dino_feats = (feats0, feats1)

        (
            flow,
            mask,
            merged,
            flow_teacher,
            merged_teacher,
            loss_distill,
            dcn_offsets,
        ) = self.flownet(torch.cat((imgs, gt), 1), dino_feats, scale=[4, 2, 1])

        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        loss_dino = self.dino_loss(merged[2], gt, self.dino)

        loss_dcn = 0
        for off in dcn_offsets:
            loss_dcn += torch.mean(torch.abs(off)) * 0.01

        if training:
            self.optimG.zero_grad()
            loss_G = (
                loss_l1
                + loss_tea
                + loss_distill * 0.01
                + loss_dino * self.dino_cfg.dino_loss_weight
                + loss_dcn * self.dino_cfg.dcn_loss_weight
            )
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            "merged_tea": merged_teacher,
            "mask": mask,
            "mask_tea": mask,
            "flow": flow[2][:, :2],
            "flow_tea": flow_teacher,
            "loss_l1": loss_l1,
            "loss_tea": loss_tea,
            "loss_distill": loss_distill,
            "loss_dino": loss_dino * self.dino_cfg.dino_loss_weight,
            "loss_dcn": loss_dcn * self.dino_cfg.dcn_loss_weight,
        }
