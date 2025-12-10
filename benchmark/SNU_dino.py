import shutil
import sys
import os
import argparse
from torch.nn import functional as F

sys.path.append(".")

from model.pytorch_msssim import ssim_matlab
from model.RIFE_dino import Model
import numpy as np
import torch
import math
import cv2
import lpips
from pytorch_fid import fid_score

# git clone https://github.com/danier97/flolpips.git on ./benchmark
# change evaluation.py line 275 return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# to return cupy.RawModule(code=strKernel).get_function(strFunction)
current_dir = os.path.dirname(os.path.abspath(__file__))
flolpips_path = os.path.join(current_dir, "flolpips")
sys.path.append(flolpips_path)
from flolpips import Flolpips

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
model.load_model("train_log_dino")
model.eval()
model.device()
flolpips_metric = Flolpips().to(device)

path = args.testlist
f = open(path, "r")
psnr_list = []
ssim_list = []
lpips_list = []
flolpips_list = []

loss_fn_alex = lpips.LPIPS(net="alex").to(device)
# --- Setup FID Folders ---
fid_pred_dir = "fid_outputs_snu-dino/pred"
fid_gt_dir = "fid_outputs_snu-dino/gt"
if os.path.exists("fid_outputs_snu-dino"):
    shutil.rmtree("fid_outputs_snu-dino")
os.makedirs(fid_pred_dir, exist_ok=True)
os.makedirs(fid_gt_dir, exist_ok=True)

for i, line in enumerate(f):
    paths = line.strip().split()
    I0 = cv2.imread(f"./{paths[0]}")
    I1 = cv2.imread(f"./{paths[1]}")
    I2 = cv2.imread(f"./{paths[2]}")
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)

    n, c, h, w = I0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    I0_unpad = I0.clone()
    I2_unpad = I2.clone()

    I0_padded = F.pad(I0, padding)
    I2_padded = F.pad(I2, padding)

    mid = model.inference(I0_padded, I2_padded)[0]
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

    flol_val = flolpips_metric(
        I0_unpad, I2_unpad, pred_tensor, gt_tensor
    ).item()
    flolpips_list.append(flol_val)

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

    # --- Save Images for FID ---
    save_name = paths[0].replace("/", "_").replace("\\", "_")

    cv2.imwrite(
        os.path.join(fid_pred_dir, save_name), (mid * 255).astype("uint8")
    )
    cv2.imwrite(
        os.path.join(fid_gt_dir, save_name), (I1 * 255).astype("uint8")
    )

num_preds = len(os.listdir(fid_pred_dir))
num_gts = len(os.listdir(fid_gt_dir))
if num_preds != num_gts:
    print(
        f"CRITICAL ERROR: Mismatch detected! Preds: {num_preds}, GTs: {num_gts}"
    )
    exit()
else:
    print(f"Success. Calculating FID on {num_preds} image pairs...")

fid_value = fid_score.calculate_fid_given_paths(
    paths=[fid_pred_dir, fid_gt_dir], batch_size=1, device=device, dims=2048
)
print(f"Final FID: {fid_value}")
print(f"Final Avg PSNR: {np.mean(psnr_list)}")
print(f"Final Avg SSIM: {np.mean(ssim_list)}")
print(f"Final Avg LPIPS: {np.mean(lpips_list)}")
print(f"Final Avg FloLPIPS: {np.mean(flolpips_list)}")
