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
from model.RIFE_septuplet import Model
import lpips
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as ms_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load septuplet model ---
model = Model()
model.load_model("train_log")
model.eval()
model.device()

# --- dataset root: vimeo_septuplet + sep_testlist.txt ---
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "vimeo_septuplet/"

f = open(path + "sep_testlist.txt", "r")

psnr_list = []
ssim_list = []
lpips_list = []
mse_list = []
ms_ssim_list = []

# temporal metrics
temporal_pixel_list = []
temporal_flow_list = []

loss_fn_alex = lpips.LPIPS(net="alex").to(device)
ms_ssim_metric = ms_ssim(data_range=1.0).to(device)


def calculate_temporal_pixel_consistency(gt_minus1, pred0, gt_plus1, gt0):
    """
    Temporal consistency based on frame differences (MSE).
    Here:
        gt_minus1: frame before GT (e.g. im3)
        gt0:       GT middle frame (e.g. im4)
        gt_plus1:  frame after GT (e.g. im5)
        pred0:     predicted middle frame
    """
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
    """
    Compute optical flow using OpenCV Farneback.
    Input: [1, C, H, W] tensors in [0, 1].
    Output: numpy flow [H, W, 2].
    """
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
    """
    Temporal consistency based on optical flow (tOF-style).
    Returns mean L1 difference between GT flow and predicted flow.
    """
    # forward motion flows
    flow_gt_prev = compute_flow_cv2(gt_minus1, gt0)      # gt-1 -> gt0
    flow_pred_prev = compute_flow_cv2(gt_minus1, pred0)  # gt-1 -> pred0

    # backward motion flows
    flow_gt_next = compute_flow_cv2(gt0, gt_plus1)       # gt0 -> gt+1
    flow_pred_next = compute_flow_cv2(pred0, gt_plus1)   # pred0 -> gt+1

    prev_diff = np.abs(flow_gt_prev - flow_pred_prev).mean()
    next_diff = np.abs(flow_gt_next - flow_pred_next).mean()

    return 0.5 * (prev_diff + next_diff)


for i in f:
    name = str(i).strip()
    if len(name) <= 1:
        continue
    print(path + "sequences/" + name + "/im1.png")

    # --- read seven frames as numpy (BGR, uint8) ---
    base = path + "sequences/" + name + "/"
    I1 = cv2.imread(base + "im1.png")  # f0
    I2 = cv2.imread(base + "im2.png")  # f1
    I3 = cv2.imread(base + "im3.png")  # f2
    I4 = cv2.imread(base + "im4.png")  # f3 (GT target)
    I5 = cv2.imread(base + "im5.png")  # f4
    I6 = cv2.imread(base + "im6.png")  # f5
    I7 = cv2.imread(base + "im7.png")  # f6

    # convert to tensors [1, C, H, W] in [0, 1]
    def to_tensor(img):
        return (
            torch.tensor(img.transpose(2, 0, 1)).to(device).float() / 255.0
        ).unsqueeze(0)

    f0 = to_tensor(I1)
    f1 = to_tensor(I2)
    f2 = to_tensor(I3)
    f3 = to_tensor(I4)  # GT middle
    f4 = to_tensor(I5)
    f5 = to_tensor(I6)
    f6 = to_tensor(I7)

    # temporal GTs around the middle
    gt_minus1 = f2   # im3
    gt0 = f3         # im4 (GT)
    gt_plus1 = f4    # im5

    # --- interpolation via septuplet model ---
    # inputs: (f0, f1, f2, f4, f5, f6) -> predict f3
    inputs = [f0, f1, f2, f4, f5, f6]
    mid = model.inference(inputs, scale=1, timestep=0.5)[0]  # [C, H, W] in [0, 1]

    pred_tensor = torch.round(mid * 255).unsqueeze(0) / 255.0  # [1, C, H, W]
    gt_tensor = gt0  # alias for clarity

    # --- SSIM (Matlab-style) ---
    ssim_val = ssim_matlab(gt_tensor, pred_tensor).detach().cpu().numpy()

    # --- LPIPS ---
    # convert BGR->RGB by channel flip, then [-1,1] range
    pred_rgb = pred_tensor.flip(1)
    gt_rgb = gt_tensor.flip(1)
    lpips_val = loss_fn_alex(pred_rgb * 2 - 1, gt_rgb * 2 - 1).item()
    lpips_list.append(lpips_val)

    # --- PSNR ---
    mid_img = (
        np.round((mid * 255).detach().cpu().numpy())
        .astype("uint8")
        .transpose(1, 2, 0)
        / 255.0
    )
    I4_norm = I4 / 255.0
    psnr = -10 * math.log10(((I4_norm - mid_img) * (I4_norm - mid_img)).mean())

    psnr_list.append(psnr)
    ssim_list.append(ssim_val)

    # --- MSE (spatial) ---
    mse_val = F.mse_loss(pred_tensor, gt_tensor).item()
    mse_list.append(mse_val)

    # --- MS-SSIM (spatial) ---
    ms_ssim_val = ms_ssim_metric(pred_tensor, gt_tensor).item()
    ms_ssim_list.append(ms_ssim_val)

    # --- temporal consistency (pixel difference) ---
    tc_pixel_val = calculate_temporal_pixel_consistency(
        gt_minus1=gt_minus1,
        pred0=pred_tensor,
        gt_plus1=gt_plus1,
        gt0=gt0,
    ).item()
    temporal_pixel_list.append(tc_pixel_val)

    # --- temporal consistency (optical flow) ---
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

# Generate and store the final benchmark result
os.makedirs('benchmark/result', exist_ok=True)

result = {
    'timestamp': datetime.now().isoformat(),
    'num_samples': len(psnr_list),
    'psnr_mean': float(np.mean(psnr_list)) if psnr_list else None,
    'ssim_mean': float(np.mean(ssim_list)) if ssim_list else None,
    'lpips_mean': float(np.mean(lpips_list)) if lpips_list else None,
    'mse_mean': float(np.mean(mse_list)) if mse_list else None,
    'ms_ssim_mean': float(np.mean(ms_ssim_list)) if ms_ssim_list else None,
    'temporal_pixel_mean': float(np.mean(temporal_pixel_list)) if temporal_pixel_list else None,
    'temporal_flow_mean': float(np.mean(temporal_flow_list)) if temporal_flow_list else None,
}

outfile = os.path.join(
    'benchmark',
    'result',
    f"vimeo90K_septuplet_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
)
with open(outfile, 'w') as fh:
    json.dump(result, fh, indent=2)

print('Saved benchmark result to', outfile)
