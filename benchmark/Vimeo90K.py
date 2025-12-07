import sys

sys.path.append(".")
import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np

from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
import lpips
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as ms_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_model("train_log")
model.eval()
model.device()

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "vimeo_triplet/"
f = open(path + "tri_testlist.txt", "r")


psnr_list = []
ssim_list = []
lpips_list = []
mse_list = []
ms_ssim_list = []

loss_fn_alex = lpips.LPIPS(net="alex").to(device)

ms_ssim_metric = ms_ssim(data_range=1.0).to(device)

for i in f:
    name = str(i).strip()
    if len(name) <= 1:
        continue
    print(path + "sequences/" + name + "/im1.png")

    I0 = cv2.imread(path + "sequences/" + name + "/im1.png")
    I1 = cv2.imread(path + "sequences/" + name + "/im2.png")
    I2 = cv2.imread(path + "sequences/" + name + "/im3.png")

    # to [0,1], shape [1, C, H, W]
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)

    mid = model.inference(I0, I2)[0]

    gt_tensor = (
        torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.0
    )
    # round to 0–255 then back to 0–1 like original code
    pred_tensor = torch.round(mid * 255).unsqueeze(0) / 255.0


    ssim_val = ssim_matlab(gt_tensor, pred_tensor).detach().cpu().numpy()


    pred_rgb = pred_tensor.flip(1)
    gt_rgb = gt_tensor.flip(1)
    lpips_val = loss_fn_alex(pred_rgb * 2 - 1, gt_rgb * 2 - 1).item()
    lpips_list.append(lpips_val)


    mid_img = (
        np.round((mid * 255).detach().cpu().numpy())
        .astype("uint8")
        .transpose(1, 2, 0)
        / 255.0
    )
    I1_norm = I1 / 255.0
    psnr = -10 * math.log10(((I1_norm - mid_img) * (I1_norm - mid_img)).mean())

    psnr_list.append(psnr)
    ssim_list.append(ssim_val)


    # pred_tensor, gt_tensor are already in [0,1]
    mse_val = F.mse_loss(pred_tensor, gt_tensor).item()
    mse_list.append(mse_val)

    ms_ssim_val = ms_ssim_metric(pred_tensor, gt_tensor).item()
    ms_ssim_list.append(ms_ssim_val)

    print(
        "Avg PSNR: {:.4f}  SSIM: {:.4f}  LPIPS: {:.4f}  MSE: {:.6f}  MS-SSIM: {:.4f}".format(
            np.mean(psnr_list),
            np.mean(ssim_list),
            np.mean(lpips_list),
            np.mean(mse_list),
            np.mean(ms_ssim_list),
        )
    )
