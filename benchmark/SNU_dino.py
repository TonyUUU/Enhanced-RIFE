import sys
import argparse
from torch.nn import functional as F

sys.path.append(".")

from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
import numpy as np
import torch
import math
import cv2
import lpips

# Select which test file to run
parser = argparse.ArgumentParser()
parser.add_argument(
    "--testlist",
    type=str,
    default="test-easy.txt",
    help="Path to the SNU-FILM text list (e.g. test-easy.txt)",
)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_model("train_log")
model.eval()
model.device()

path = args.testlist
f = open(path, "r")
psnr_list = []
ssim_list = []

loss_fn_alex = lpips.LPIPS(net="alex").to(device)
lpips_list = []


for i, line in enumerate(f):
    paths = line.strip().split()
    print(f"../{paths[0]}")
    I0 = cv2.imread(f"./{paths[0]}")
    I1 = cv2.imread(f"./{paths[1]}")
    I2 = cv2.imread(f"./{paths[2]}")
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)

    n, c, h, w = I0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    I0 = F.pad(I0, padding)
    I2 = F.pad(I2, padding)

    mid = model.inference(I0, I2)[0]
    mid = mid[:, :h, :w]

    gt_tensor = (
        torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.0
    )
    pred_tensor = torch.round(mid * 255).unsqueeze(0) / 255.0
    ssim = ssim_matlab(gt_tensor, pred_tensor).detach().cpu().numpy()

    pred_rgb = pred_tensor.flip(1)
    gt_rgb = gt_tensor.flip(1)
    lpips_val = loss_fn_alex(pred_rgb * 2 - 1, gt_rgb * 2 - 1).item()
    lpips_list.append(lpips_val)

    mid = (
        np.round((mid * 255).detach().cpu().numpy())
        .astype("uint8")
        .transpose(1, 2, 0)
        / 255.0
    )
    I1 = I1 / 255.0
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print(
        "Avg PSNR: {} SSIM: {} LPIPS {}".format(
            np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)
        )
    )
