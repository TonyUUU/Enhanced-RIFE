import json
import os
import sys
from datetime import datetime

sys.path.append(".")
import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np

from model.pytorch_msssim import ssim_matlab
from model.RIFE_4ch import Model
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

f = open(os.path.join(path, "tri_testlist.txt"), "r")

psnr_list = []
ssim_list = []
lpips_list = []
mse_list = []
ms_ssim_list = []
temporal_pixel_list = []
temporal_flow_list = []

loss_fn_alex = lpips.LPIPS(net="alex").to(device)
ms_ssim_metric = ms_ssim(data_range=1.0).to(device)


def to_4ch_tensor(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray[..., None]
    img_4 = np.concatenate([img_bgr, gray], axis=2)  # [H,W,4]
    ten = (
        torch.tensor(img_4.transpose(2, 0, 1))
        .to(device)
        .float()
        / 255.0
    ).unsqueeze(0)
    return ten


def calculate_temporal_pixel_consistency(gt_minus1, pred0, gt_plus1, gt0):
    diff_gt_prev = gt0 - gt_minus1
    diff_pred_prev = pred0 - gt_minus1
    diff_gt_next = gt_plus1 - gt0
    diff_pred_next = gt_plus1 - pred0
    prev_mse = F.mse_loss(diff_pred_prev, diff_gt_prev)
    next_mse = F.mse_loss(diff_pred_next, diff_gt_next)
    return 0.5 * (prev_mse + next_mse)


def compute_flow_cv2(frame_a, frame_b):
    a = frame_a[0].detach().cpu().permute(1, 2, 0).numpy()
    b = frame_b[0].detach().cpu().permute(1, 2, 0).numpy()
    a_uint8 = (a[:, :, :3] * 255.0).astype(np.uint8)
    b_uint8 = (b[:, :, :3] * 255.0).astype(np.uint8)
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
    flow_gt_prev = compute_flow_cv2(gt_minus1, gt0)
    flow_pred_prev = compute_flow_cv2(gt_minus1, pred0)
    flow_gt_next = compute_flow_cv2(gt0, gt_plus1)
    flow_pred_next = compute_flow_cv2(pred0, gt_plus1)
    prev_diff = np.abs(flow_gt_prev - flow_pred_prev).mean()
    next_diff = np.abs(flow_gt_next - flow_pred_next).mean()
    return 0.5 * (prev_diff + next_diff)


for line in f:
    name = line.strip()
    if len(name) <= 1:
        continue
    base = os.path.join(path, "sequences", name)
    I0 = cv2.imread(os.path.join(base, "im1.png"))
    I1 = cv2.imread(os.path.join(base, "im2.png"))
    I2 = cv2.imread(os.path.join(base, "im3.png"))

    gt_minus1 = to_4ch_tensor(I0)
    gt0 = to_4ch_tensor(I1)
    gt_plus1 = to_4ch_tensor(I2)

    mid = model.inference(gt_minus1, gt_plus1)[0]
    pred_tensor = torch.round(mid * 255).unsqueeze(0) / 255.0  # [1,4,H,W]
    gt_tensor = gt0

    pred_3 = pred_tensor[:, :3]
    gt_3 = gt_tensor[:, :3]

    ssim_val = ssim_matlab(gt_3, pred_3).detach().cpu().numpy()

    pred_rgb = pred_3.flip(1)
    gt_rgb = gt_3.flip(1)
    lpips_val = loss_fn_alex(pred_rgb * 2 - 1, gt_rgb * 2 - 1).item()
    lpips_list.append(lpips_val)

    mid_np = (
        np.round((mid * 255).detach().cpu().numpy())
        .astype("uint8")
    )
    mid_3 = mid_np[:3].transpose(1, 2, 0) / 255.0
    I1_norm = I1 / 255.0
    psnr = -10 * math.log10(((I1_norm - mid_3) * (I1_norm - mid_3)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim_val)

    mse_val = F.mse_loss(pred_3, gt_3).item()
    mse_list.append(mse_val)

    ms_ssim_val = ms_ssim_metric(pred_3, gt_3).item()
    ms_ssim_list.append(ms_ssim_val)

    tc_pixel_val = calculate_temporal_pixel_consistency(
        gt_minus1, pred_tensor, gt_plus1, gt_tensor
    ).item()
    temporal_pixel_list.append(tc_pixel_val)

    tc_flow_val = calculate_temporal_flow_consistency(
        gt_minus1, pred_tensor, gt_plus1, gt_tensor
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

os.makedirs("benchmark/result", exist_ok=True)

result = {
    "timestamp": datetime.now().isoformat(),
    "num_samples": len(psnr_list),
    "psnr_mean": float(np.mean(psnr_list)) if psnr_list else None,
    "ssim_mean": float(np.mean(ssim_list)) if ssim_list else None,
    "lpips_mean": float(np.mean(lpips_list)) if lpips_list else None,
    "mse_mean": float(np.mean(mse_list)) if mse_list else None,
    "ms_ssim_mean": float(np.mean(ms_ssim_list)) if ms_ssim_list else None,
    "temporal_pixel_mean": float(np.mean(temporal_pixel_list)) if temporal_pixel_list else None,
    "temporal_flow_mean": float(np.mean(temporal_flow_list)) if temporal_flow_list else None,
}

outfile = os.path.join(
    "benchmark",
    "result",
    f"vimeo90K_4ch_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
)
with open(outfile, "w") as fh:
    json.dump(result, fh, indent=2)

print("Saved benchmark result to", outfile)
