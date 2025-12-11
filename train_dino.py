# Real-Time Intermediate Flow Estimation for Video Frame Interpolation
# Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou
# https://arxiv.org/abs/2011.06294

import os
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE_dino import Model
from dataset_dino import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_learning_rate(step):
    warmup_steps = args.warmup_epochs * args.step_per_epoch
    total_steps = args.epoch * args.step_per_epoch

    if step < warmup_steps:
        return args.lr_warmup
    else:
        steps_into_finetune = step - warmup_steps
        total_finetune_steps = total_steps - warmup_steps

        if (
            total_finetune_steps <= 0
            or steps_into_finetune > total_finetune_steps
        ):
            return args.lr_finetune

        progress = steps_into_finetune / total_finetune_steps
        mul = np.cos(progress * math.pi) * 0.5 + 0.5
        return (args.lr_finetune - 1e-6) * mul + 1e-6


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (
        normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1]
    )
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def to_np_img(tensor):
    # convert Tensor [B, C, H, W] to Numpy Image [B, H, W, C]
    return (tensor.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
        "uint8"
    )


def to_np(tensor):
    return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()


def train(model, local_rank):
    log_dir = args.log_path

    if local_rank == 0:
        loss_fn_alex = lpips.LPIPS(net="alex").to(device)
        loss_fn_alex.eval()
        for param in loss_fn_alex.parameters():
            param.requires_grad = False
    else:
        loss_fn_alex = None

    if local_rank == 0 and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if local_rank == 0:
        writer = SummaryWriter(os.path.join(log_dir, "train"))
        writer_val = SummaryWriter(os.path.join(log_dir, "validate"))
    else:
        writer = None
        writer_val = None

    if os.path.exists(os.path.join(log_dir, "flownet.pkl")):
        if local_rank == 0:
            print("Resumed from existing DINO checkpoint.")
        model.load_model(log_dir)
    elif args.resume_legacy:
        if local_rank == 0:
            print("Load original RIFE weights into DINO architecture")
        model.load_original_rife("train_log")
    else:
        if local_rank == 0:
            print("WARNING: Training from scratch")

    nr_eval = 0
    dataset = VimeoDataset("train")

    if args.world_size > 1:
        sampler = DistributedSampler(dataset)
        train_data = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
    else:
        sampler = None
        train_data = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            shuffle=True,
        )

    args.step_per_epoch = train_data.__len__()
    step = args.start_epoch * args.step_per_epoch
    print(f"Training starting from epoch {args.start_epoch}, step {step}...")
    dataset_val = VimeoDataset("validation")
    val_data = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        pin_memory=True,
        num_workers=8,
    )
    print("training...")
    time_stamp = time.time()
    for epoch in range(args.start_epoch, args.epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.0
            timestep = timestep.to(device, non_blocking=True)

            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * args.world_size / 4

            if args.world_size > 1:
                flownet_module = model.flownet.module
            else:
                flownet_module = model.flownet
            dino_adapters = [
                flownet_module.dino_compressor,
                flownet_module.dino_refiner,
                flownet_module.unet.fusion_s3,
                flownet_module.unet.fusion_s2,
                flownet_module.unet.adapt_s2,
            ]
            finetune_modules = [
                flownet_module.dino_compressor,
                flownet_module.dino_refiner,
                flownet_module.unet,
            ]
            # freeze backbone
            flownet_module.eval()
            flownet_module.requires_grad_(False)
            if epoch < args.warmup_epochs:
                for block in dino_adapters:
                    block.train()
                    block.requires_grad_(True)
            else:
                for block in finetune_modules:
                    block.train()
                    block.requires_grad_(True)

            pred, info = model.update(imgs, gt, learning_rate, training=True)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if step % 500 == 0 and local_rank == 0:
                print(f"Saved intermediate checkpoint at step {step}")
                torch.save(
                    model.flownet.state_dict(),
                    os.path.join(log_dir, f"flownet_step_{step}.pkl"),
                )

            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar("learning_rate", learning_rate, step)
                writer.add_scalar("loss/l1", info["loss_l1"], step)
                writer.add_scalar("loss/tea", info["loss_tea"], step)
                writer.add_scalar("loss/distill", info["loss_distill"], step)
                writer.add_scalar("loss/dino", info["loss_dino"], step)
                writer.add_scalar("loss/dcn", info["loss_dcn"], step)
                with torch.no_grad():
                    # Normalize [0, 1] -> [-1, 1]
                    train_pred_norm = (pred * 2) - 1
                    train_gt_norm = (gt * 2) - 1
                    train_lpips_val = (
                        loss_fn_alex(train_pred_norm, train_gt_norm)
                        .mean()
                        .item()
                    )
                writer.add_scalar("train/lpips", train_lpips_val, step)

            if step % 1000 == 1 and local_rank == 0:
                gt = to_np_img(gt)
                mask = to_np_img(
                    torch.cat((info["mask"], info["mask_tea"]), dim=3)
                )
                pred = to_np_img(pred)
                merged_img = to_np_img(info["merged_tea"])
                flow0 = to_np(info["flow"])
                flow1 = to_np(info["flow_tea"])

                for j in range(5):
                    imgs_BGR = np.concatenate(
                        (merged_img[j], pred[j], gt[j]), axis=1
                    )
                    imgs = imgs_BGR[:, :, ::-1]
                    rgb_flow_student = flow2rgb(flow0[j])
                    rgb_flow_teacher = flow2rgb(flow1[j])
                    combined_flow = np.concatenate(
                        (rgb_flow_student, rgb_flow_teacher), axis=1
                    )
                    writer.add_image(
                        str(j) + "/img", imgs, step, dataformats="HWC"
                    )
                    writer.add_image(
                        str(j) + "/flow",
                        combined_flow,
                        step,
                        dataformats="HWC",
                    )
                    writer.add_image(
                        str(j) + "/mask", mask[j], step, dataformats="HWC"
                    )
                writer.flush()
            if local_rank == 0:
                print(
                    f"epoch:{epoch} {i}/{args.step_per_epoch} "
                    f"time:{data_time_interval:.2f}+{train_time_interval:.2f} "
                    f"loss_l1:{info['loss_l1']:.4e} "
                    f"loss_dino:{info['loss_dino']:.4e} "
                    f"loss_dcn:{info['loss_dcn']:.4e}"
                )
            step += 1
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(
                model, val_data, step, local_rank, writer_val, loss_fn_alex
            )
        model.save_model(log_dir, local_rank)
        if args.world_size > 1:
            dist.barrier()


def evaluate(
    model, val_data, nr_eval, local_rank, writer_val, loss_fn_alex=None
):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    loss_dino_list = []
    psnr_list = []
    psnr_list_teacher = []
    lpips_list = []
    for i, data in enumerate(val_data):
        data_gpu, _ = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.0
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]

        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info["merged_tea"]

        loss_l1_list.append(info["loss_l1"].cpu().numpy())
        loss_tea_list.append(info["loss_tea"].cpu().numpy())
        loss_distill_list.append(info["loss_distill"].cpu().numpy())
        loss_dino_list.append(info["loss_dino"].cpu().numpy())

        for j in range(gt.shape[0]):
            mse = torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data
            psnr = -10 * math.log10(mse)
            psnr_list.append(psnr)

            mse_tea = (
                torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j]))
                .cpu()
                .data
            )
            psnr = -10 * math.log10(mse_tea)
            psnr_list_teacher.append(psnr)
            if loss_fn_alex is not None:
                # Normalize from [0, 1] to [-1, 1] for LPIPS
                pred_norm = (pred[j] * 2) - 1
                gt_norm = (gt[j] * 2) - 1

                # Calculate LPIPS (returns 1x1 tensor)
                lpips_val = loss_fn_alex(
                    pred_norm.unsqueeze(0), gt_norm.unsqueeze(0)
                ).item()
                lpips_list.append(lpips_val)

        gt = to_np_img(gt)
        pred = to_np_img(pred)
        merged_img = to_np_img(merged_img)
        flow0 = to_np(info["flow"])

        if i == 0 and local_rank == 0:
            for j in range(10):
                imgs_RGB = np.concatenate((merged_img[j], pred[j], gt[j]), 1)
                imgs = imgs_RGB[:, :, ::-1]
                writer_val.add_image(
                    str(j) + "/img", imgs.copy(), nr_eval, dataformats="HWC"
                )
                writer_val.add_image(
                    str(j) + "/flow",
                    flow2rgb(flow0[j][:, :, ::-1]),
                    nr_eval,
                    dataformats="HWC",
                )

    if local_rank != 0:
        return
    writer_val.add_scalar("psnr", np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar(
        "psnr_teacher", np.array(psnr_list_teacher).mean(), nr_eval
    )
    if len(lpips_list) > 0:
        writer_val.add_scalar("lpips", np.array(lpips_list).mean(), nr_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=60, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument(
        "--batch_size", default=128, type=int, help="minibatch size"
    )
    parser.add_argument("--local_rank", default=0, type=int, help="local rank")
    parser.add_argument("--world_size", default=1, type=int, help="world size")
    parser.add_argument(
        "--resume_legacy",
        action="store_true",
        help="Load original RIFE weights",
    )
    parser.add_argument(
        "--val_batch_size", default=16, type=int, help="validation batch size"
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs to freeze flow network",
    )
    parser.add_argument(
        "--lr_warmup",
        default=8e-4,
        type=float,
        help="Learning rate for stage 1",
    )
    parser.add_argument(
        "--lr_finetune",
        default=4e-4,
        type=float,
        help="Initial learning rate for stage 2 ",
    )
    parser.add_argument(
        "--log_path",
        default="train_log_dino",
        type=str,
        help="Path to save checkpoints and tensorboard logs",
    )

    args = parser.parse_args()

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", world_size=args.world_size
        )
        torch.cuda.set_device(args.local_rank)
        model = Model(args.local_rank)
    else:
        model = Model(local_rank=-1)
        args.local_rank = 0  # hack for logging
    torch.backends.cudnn.benchmark = True

    train(model, args.local_rank)
