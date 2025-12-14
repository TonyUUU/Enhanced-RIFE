import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32, data_root=None):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448

        if data_root is not None:
            self.data_root = data_root
        else:
            self.data_root = 'vimeo_triplet'

        self.image_root = os.path.join(self.data_root, "sequences")
        train_fn = os.path.join(self.data_root, "tri_trainlist.txt")
        test_fn = os.path.join(self.data_root, "tri_testlist.txt")
        with open(train_fn, "r") as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, "r") as f:
            self.testlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == "train":
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == "test":
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x : x + h, y : y + w, :]
        img1 = img1[x : x + h, y : y + w, :]
        gt = gt[x : x + h, y : y + w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [
            imgpath + "/im1.png",
            imgpath + "/im2.png",
            imgpath + "/im3.png",
        ]

        # Load images (BGR, 3 channels)
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep

    @staticmethod
    def _to_4ch_bgr_gray(img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # [H,W]
        gray = gray[..., None]                            # [H,W,1]
        img_4ch = np.concatenate([img_bgr, gray], axis=2) # [H,W,4]
        return img_4ch

    def __getitem__(self, index):
        img0, gt, img1, timestep = self.getimg(index)

        if self.dataset_name == "train":
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0_4 = self._to_4ch_bgr_gray(img0)
        img1_4 = self._to_4ch_bgr_gray(img1)
        gt_4 = self._to_4ch_bgr_gray(gt)

        img0_t = torch.from_numpy(img0_4.copy()).permute(2, 0, 1)  # [4,H,W]
        img1_t = torch.from_numpy(img1_4.copy()).permute(2, 0, 1)  # [4,H,W]
        gt_t = torch.from_numpy(gt_4.copy()).permute(2, 0, 1)      # [4,H,W]

        timestep = torch.tensor(timestep, dtype=torch.float32).reshape(1, 1, 1)

        # Concatenate along channel: [4+4+4, H, W] = [12,H,W]
        return torch.cat((img0_t, img1_t, gt_t), 0), timestep
