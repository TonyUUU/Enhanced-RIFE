import argparse
import math
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset_4ch import VimeoDataset
from model.RIFE_4ch import Model

device = torch.device("cuda")
log_path = "train_log"


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.0
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) /
                     (args.epoch * args.step_per_epoch - 2000.0) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized = flow_map_np / (np.abs(flow_map_np).max() + 1e-6)

    rgb_map[:, :, 0] += normalized[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized[:, :, 0] + normalized[:, :, 1])
    rgb_map[:, :, 2] += normalized[:, :, 1]
    return rgb_map.clip(0, 1)


def train(model, local_rank, data_root=None):

    if local_rank == 0:
        writer = SummaryWriter("train")
        writer_val = SummaryWriter("validate")
    else:
        writer = None
        writer_val = None

    step = 0
    nr_eval = 0

    dataset = VimeoDataset("train", data_root=data_root)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
    args.step_per_epoch = len(train_data)

    dataset_val = VimeoDataset("validation", data_root=data_root)
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=2)

    print("training...")
    time_stamp = time.time()

    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)

        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            data_gpu, _ = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.0

            imgs = data_gpu[:, :8]     # 2 × 4ch
            gt = data_gpu[:, 8:12]     # 1 × 4ch

            learning_rate = get_learning_rate(step) * args.world_size / 4
            pred, info = model.update(imgs, gt, learning_rate, training=True)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # --- Logging ---
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar("learning_rate", learning_rate, step)
                writer.add_scalar("loss/l1", info["loss_l1"], step)
                writer.add_scalar("loss/tea", info["loss_tea"], step)
                writer.add_scalar("loss/distill", info["loss_distill"], step)

            if step % 1000 == 1 and local_rank == 0:
                gt_np = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype("uint8")
                pred_np = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype("uint8")
                merged_np = (info["merged_tea"].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype("uint8")

                mask_np = (
                    torch.cat((info["mask"], info["mask_tea"]), 3)
                    .permute(0, 2, 3, 1)
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).astype("uint8")

                flow0 = info["flow"].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info["flow_tea"].permute(0, 2, 3, 1).detach().cpu().numpy()

                for k in range(min(5, gt_np.shape[0])):
                    img_vis = np.concatenate((merged_np[k], pred_np[k], gt_np[k]), 1)
                    writer.add_image(f"{k}/img", img_vis[:, :, ::-1], step, dataformats="HWC")

                    writer.add_image(
                        f"{k}/flow",
                        np.concatenate((flow2rgb(flow0[k]), flow2rgb(flow1[k])), 1),
                        step,
                        dataformats="HWC",
                    )

                    writer.add_image(f"{k}/mask", mask_np[k], step, dataformats="HWC")

                writer.flush()

            if local_rank == 0:
                print(
                    f"epoch:{epoch} {i}/{args.step_per_epoch} "
                    f"time:{data_time_interval:.2f}+{train_time_interval:.2f} "
                    f"loss_l1:{info['loss_l1']:.4e}"
                )

            step += 1

        nr_eval += 1
        evaluate(model, val_data, step, local_rank, writer_val)

        model.save_model(log_path, local_rank)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_teacher_list = []

    for i, data in enumerate(val_data):
        data_gpu, _ = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.0
        imgs = data_gpu[:, :8]
        gt = data_gpu[:, 8:12]

        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged = info["merged_tea"]

        loss_l1_list.append(info["loss_l1"].cpu().numpy())
        loss_tea_list.append(info["loss_tea"].cpu().numpy())
        loss_distill_list.append(info["loss_distill"].cpu().numpy())

        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) ** 2).cpu().item())
            psnr_list.append(psnr)

            psnr_t = -10 * math.log10(torch.mean((merged[j] - gt[j]) ** 2).cpu().item())
            psnr_teacher_list.append(psnr_t)

        if i == 0 and local_rank == 0:
            gt_np = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            pred_np = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            merged_np = (merged.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

            for j in range(min(10, gt_np.shape[0])):
                img_vis = np.concatenate((merged_np[j], pred_np[j], gt_np[j]), 1)
                writer_val.add_image(f"{j}/img", img_vis[:, :, ::-1], nr_eval, dataformats="HWC")

    if local_rank == 0:
        writer_val.add_scalar("psnr", np.mean(psnr_list), nr_eval)
        writer_val.add_scalar("psnr_teacher", np.mean(psnr_teacher_list), nr_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--local-rank", "--local_rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--data_root", default=None, type=str)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # seeds
    seed = 1234
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(args.local_rank)
    train(model, args.local_rank, data_root=args.data_root)
