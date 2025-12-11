import json
import os
import sys
from datetime import datetime
import shutil
import sys
import os

sys.path.append(".")
import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np

from model.pytorch_msssim import ssim_matlab
from model.RIFE_dino import Model
import lpips
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure as ms_ssim,
)
from pytorch_fid import fid_score

# git clone https://github.com/danier97/flolpips.git on benchmark
current_dir = os.path.dirname(os.path.abspath(__file__))
flolpips_path = os.path.join(current_dir, "flolpips")
sys.path.append(flolpips_path)
from flolpips import Flolpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_model("train_log_dino")
model.eval()
model.device()
flolpips_metric = Flolpips().to(device)

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
flolpips_list = []

# temporal metrics
temporal_pixel_list = []
temporal_flow_list = []

loss_fn_alex = lpips.LPIPS(net="alex").to(device)
ms_ssim_metric = ms_ssim(data_range=1.0).to(device)


def calculate_temporal_pixel_consistency(gt_minus1, pred0, gt_plus1, gt0):
    # Temporal consistency based on frame differences (MSE).
    # forward motion
    diff_gt_prev = gt0 - gt_minus1
    diff_pred_prev = pred0 - gt_minus1

    # backward motion
    diff_gt_next = gt_plus1 - gt0
    diff_pred_next = gt_plus1 - pred0

    prev_mse = F.mse_loss(diff_pred_prev, diff_gt_prev)
    next_mse = F.mse_loss(diff_pred_next, diff_gt_next)

    return 0.5 * (prev_mse + next_mse)


def compute_flow_cv2(frame_a, frame_b):
    # Compute optical flow using OpenCV Farneback.
    # Input: [1, C, H, W] tensors in [0, 1].
    # Output: numpy flow [H, W, 2].
    a = frame_a[0].detach().cpu().permute(1, 2, 0).numpy()
    b = frame_b[0].detach().cpu().permute(1, 2, 0).numpy()

    a_uint8 = (a * 255.0).astype(np.uint8)
    b_uint8 = (b * 255.0).astype(np.uint8)

    a_gray = cv2.cvtColor(a_uint8, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_uint8, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        a_gray,
        b_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow


def calculate_temporal_flow_consistency(gt_minus1, pred0, gt_plus1, gt0):
    # Temporal consistency based on optical flow (tOF-style).
    # forward motion flows
    flow_gt_prev = compute_flow_cv2(gt_minus1, gt0)  # gt-1 -> gt0
    flow_pred_prev = compute_flow_cv2(gt_minus1, pred0)  # gt-1 -> pred0

    # backward motion flows
    flow_gt_next = compute_flow_cv2(gt0, gt_plus1)  # gt0 -> gt1
    flow_pred_next = compute_flow_cv2(pred0, gt_plus1)  # pred0 -> gt1

    prev_diff = np.abs(flow_gt_prev - flow_pred_prev).mean()
    next_diff = np.abs(flow_gt_next - flow_pred_next).mean()

    return 0.5 * (prev_diff + next_diff)


# --- Setup FID Folders ---
fid_pred_dir = "fid_outputs_vimeo/pred"
fid_gt_dir = "fid_outputs_vimeo/gt"
if os.path.exists("fid_outputs_vimeo"):
    shutil.rmtree("fid_outputs_vimeo")
os.makedirs(fid_pred_dir, exist_ok=True)
os.makedirs(fid_gt_dir, exist_ok=True)

for i in f:
    name = str(i).strip()
    if len(name) <= 1:
        continue
    print(path + "sequences/" + name + "/im1.png")

    # read three frames as numpy (BGR, uint8)
    I0 = cv2.imread(path + "sequences/" + name + "/im1.png")  # gt-1
    I1 = cv2.imread(path + "sequences/" + name + "/im2.png")  # gt0
    I2 = cv2.imread(path + "sequences/" + name + "/im3.png")  # gt1

    # convert to tensors [1, C, H, W] in [0, 1]
    gt_minus1 = (
        torch.tensor(I0.transpose(2, 0, 1)).to(device).float() / 255.0
    ).unsqueeze(0)
    gt0 = (
        torch.tensor(I1.transpose(2, 0, 1)).to(device).float() / 255.0
    ).unsqueeze(0)
    gt_plus1 = (
        torch.tensor(I2.transpose(2, 0, 1)).to(device).float() / 255.0
    ).unsqueeze(0)

    # interpolation using Dino model
    mid = model.inference(gt_minus1, gt_plus1)[0]  # [C, H, W] in [0, 1]
    pred_tensor = torch.round(mid * 255).unsqueeze(0) / 255.0  # [1, C, H, W]
    gt_tensor = gt0  # alias for clarity

    # SSIM (Matlab-style)
    ssim_val = ssim_matlab(gt_tensor, pred_tensor).detach().cpu().numpy()

    # LPIPS
    pred_rgb = pred_tensor.flip(1)
    gt_rgb = gt_tensor.flip(1)
    lpips_val = loss_fn_alex(pred_rgb * 2 - 1, gt_rgb * 2 - 1).item()
    lpips_list.append(lpips_val)

    # flolpips
    flol_val = flolpips_metric(
        gt_minus1, gt_plus1, pred_tensor, gt_tensor
    ).item()
    flolpips_list.append(flol_val)

    # PSNR
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

    # MSE (spatial)
    mse_val = F.mse_loss(pred_tensor, gt_tensor).item()
    mse_list.append(mse_val)

    # MS-SSIM (spatial)
    ms_ssim_val = ms_ssim_metric(pred_tensor, gt_tensor).item()
    ms_ssim_list.append(ms_ssim_val)

    # temporal consistency (pixel difference)
    tc_pixel_val = calculate_temporal_pixel_consistency(
        gt_minus1=gt_minus1,
        pred0=pred_tensor,
        gt_plus1=gt_plus1,
        gt0=gt0,
    ).item()
    temporal_pixel_list.append(tc_pixel_val)

    # temporal consistency (optical flow)
    tc_flow_val = calculate_temporal_flow_consistency(
        gt_minus1=gt_minus1,
        pred0=pred_tensor,
        gt_plus1=gt_plus1,
        gt0=gt0,
    )
    temporal_flow_list.append(tc_flow_val)

    print(
        "Avg PSNR: {:.4f}  SSIM: {:.4f}  LPIPS: {:.4f}  "
        "MSE: {:.6f}  MS-SSIM: {:.4f}  "
        "TemporalPixel(MSE): {:.6f}  TemporalFlow(L1): {:.6f}".format(
            np.mean(psnr_list),
            np.mean(ssim_list),
            np.mean(lpips_list),
            np.mean(mse_list),
            np.mean(ms_ssim_list),
            np.mean(temporal_pixel_list),
            np.mean(temporal_flow_list),
        )
    )
    save_name = name.replace("/", "_") + ".png"
    cv2.imwrite(
        os.path.join(fid_pred_dir, save_name), (mid_img * 255).astype("uint8")
    )
    cv2.imwrite(
        os.path.join(fid_gt_dir, save_name), (I1_norm * 255).astype("uint8")
    )
fid_value = fid_score.calculate_fid_given_paths(
    paths=[fid_pred_dir, fid_gt_dir], batch_size=1, device=device, dims=2048
)

# Generate and store the final benchmark result
os.makedirs("benchmark/result", exist_ok=True)

result = {
    "timestamp": datetime.now().isoformat(),
    "num_samples": len(psnr_list),
    "psnr_mean": float(np.mean(psnr_list)) if psnr_list else None,
    "ssim_mean": float(np.mean(ssim_list)) if ssim_list else None,
    "lpips_mean": float(np.mean(lpips_list)) if lpips_list else None,
    "mse_mean": float(np.mean(mse_list)) if mse_list else None,
    "ms_ssim_mean": float(np.mean(ms_ssim_list)) if ms_ssim_list else None,
    "temporal_pixel_mean": (
        float(np.mean(temporal_pixel_list)) if temporal_pixel_list else None
    ),
    "temporal_flow_mean": (
        float(np.mean(temporal_flow_list)) if temporal_flow_list else None
    ),
    "FID": fid_value,
    "FloLPIPS": float(np.mean(flolpips_list)) if flolpips_list else None,
}

outfile = os.path.join(
    "benchmark",
    "result",
    f"vimeo90K_dino_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
)
with open(outfile, "w") as fh:
    json.dump(result, fh, indent=2)

print("Saved benchmark result to", outfile)
