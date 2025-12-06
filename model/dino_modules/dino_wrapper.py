# Dino U-Net: Exploiting High-Fidelity Dense Features
# from Foundation Models for Medical Image Segmentation.
# Yifan Gao, Haoyue Li, Feng Yuan, Xiaosong Wang, and Xin Gao
# 1 University of Science and Technology of China
# 2 Shanghai Innovation Institute
# 3 Shanghai Artificial Intelligence Laboratory
# https://github.com/yifangao112/DinoUNet

import torch
import torch.nn as nn
from dino_config import DinoConfig
from torchvision.transforms import v2


class DinoWrapper(nn.Module):

    def __init__(self, config: DinoConfig):
        super().__init__()
        self.cfg = config

        self.model = torch.hub.load(
            self.cfg.dino_repo_dir,
            self.cfg.dino_model_name,
            source="local",
            weights=self.cfg.dino_checkpoint_path,
        )

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        self.num_registers = self.model.n_storage_tokens

        # -- Dino U-Net interaction indices --
        self.interaction_indices = self.cfg.interaction_indices

    def get_features(self, img, use_grad=False):
        """
        Extracts DINO semantic features from the input image.

        Args:
            img (Tensor): Input image (B, 3, H, W).
        Returns:
            list[Tensor]: List of Feature map [(B, embed_dim, H//p, W//p)].
        """
        img_rgb = img.flip(1)
        transform = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        img_dino = transform(img_rgb)

        context = torch.enable_grad() if use_grad else torch.no_grad()

        with context:
            features = self.model.get_intermediate_layers(
                img_dino, n=self.interaction_indices, reshape=True
            )
            if not use_grad:
                features = [f.detach() for f in features]

        return features
