import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32, colab=False, data_folder=None):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448

        # Use Vimeo-Septuplet only
        if colab:
            # e.g. /content/drive/MyDrive/<data_folder>/vimeo_septuplet
            self.data_root = f'/content/drive/MyDrive/{data_folder}/vimeo_septuplet'
        else:
            self.data_root = 'vimeo_septuplet'

        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')

        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def _random_crop_many(self, frames, h, w):
        """
        frames: list of np.ndarray [H,W,3]
        returns: list of cropped frames (same crop applied to all)
        """
        ih, iw, _ = frames[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        out = [f[x:x + h, y:y + w, :] for f in frames]
        return out

    def _augment_many(self, frames):
        """
        Apply the same augmentations (flips/rotations) to all frames.
        frames: list of np.ndarray [H,W,3]
        """
        # channel flip
        if random.uniform(0, 1) < 0.5:
            frames = [f[:, :, ::-1] for f in frames]

        # vertical flip
        if random.uniform(0, 1) < 0.5:
            frames = [f[::-1] for f in frames]

        # horizontal flip
        if random.uniform(0, 1) < 0.5:
            frames = [f[:, ::-1] for f in frames]

        # random rotation
        p = random.uniform(0, 1)
        if p < 0.25:
            frames = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]
        elif p < 0.5:
            frames = [cv2.rotate(f, cv2.ROTATE_180) for f in frames]
        elif p < 0.75:
            frames = [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]

        return frames

    def getimg(self, index):
        """
        Vimeo-Septuplet:
        We want (f0,f1,f2,f4,f5,f6) -> f3

        Map:
            f0 = im1
            f1 = im2
            f2 = im3
            f3 = im4  (GT)
            f4 = im5
            f5 = im6
            f6 = im7
        """
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 8)]

        f0 = cv2.imread(imgpaths[0])  # im1
        f1 = cv2.imread(imgpaths[1])  # im2
        f2 = cv2.imread(imgpaths[2])  # im3
        f3 = cv2.imread(imgpaths[3])  # im4 (GT)
        f4 = cv2.imread(imgpaths[4])  # im5
        f5 = cv2.imread(imgpaths[5])  # im6
        f6 = cv2.imread(imgpaths[6])  # im7

        # endpoints are f0 and f6; f3 is in the middle: t = 3/6 = 0.5
        timestep = 0.5

        frames_in = [f0, f1, f2, f4, f5, f6]  # 6 input frames
        return frames_in, f3, timestep

    def __getitem__(self, index):
        frames_in, gt, timestep = self.getimg(index)

        if self.dataset_name == 'train':
            # crop all 6 inputs + gt together
            all_frames = frames_in + [gt]
            all_frames = self._random_crop_many(all_frames, 224, 224)
            frames_in = all_frames[:-1]
            gt = all_frames[-1]

            # shared augmentations
            all_frames = frames_in + [gt]
            all_frames = self._augment_many(all_frames)
            frames_in = all_frames[:-1]
            gt = all_frames[-1]

        # to tensors
        frames_t = [torch.from_numpy(f.copy()).permute(2, 0, 1) for f in frames_in]
        gt_t = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep, dtype=torch.float32).reshape(1, 1, 1)

        # imgs is concatenation of 6 input frames: [18, H, W]
        imgs = torch.cat(frames_t, dim=0)

        # Return imgs (6 frames), gt (mid frame), and timestep
        return imgs, gt_t, timestep
