import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gauss_kernel(size=5, channels=4):
    kernel = torch.tensor([
        [1., 4., 6., 4., 1],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ])
    kernel /= 256.0
    kernel = kernel.repeat(channels, 1, 1, 1).to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(img, kernel, groups=img.shape[1])


def upsample(x):
    B, C, H, W = x.shape
    # nearest-neighbor-ish custom upsample
    x_up = torch.zeros(B, C, H * 2, W * 2, device=device)
    x_up[:, :, ::2, ::2] = x
    x_up = conv_gauss(x_up, 4 * gauss_kernel(channels=C))
    return x_up


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for _ in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, channels=4):
        super().__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)

    def forward(self, input, target):
        pyr_input = laplacian_pyramid(input, self.gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self.gauss_kernel, self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
