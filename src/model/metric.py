import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import torch.nn.functional as F
from math import exp
import numpy as np

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, device, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to(device)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3d(window_size, device, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().to(device)
    return window

def structural_similarity_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    # Code from: https://github.com/megvii-research/ECCV2022-RIFE/blob/15cb7f2389ccd93e8b8946546d4665c9b41541a3/benchmark/pytorch_msssim/__init__.py#L81
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, img1.device, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret    
    
    
def psnr(img, imclean):
    """
    compute psnr.
    """
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = []
    for i in range(Img.shape[0]):
        ps = peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=1.0)
        if np.isinf(ps):
            continue
        PSNR.append(ps)
    return sum(PSNR)/len(PSNR)


def ssim_python(img, imclean):
    """
    compute ssim using scikit-image
    """
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    Img = img.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    Iclean = imclean.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    SSIM = []
    for i in range(Img.shape[0]):
        # API changed for scikit-image 0.20.0.
        # To match the implementation of Wang et al. [1]_, set `gaussian_weights`
        # to True, `sigma` to 1.5, `use_sample_covariance` to False, and
        # specify the `data_range` argument.
        ss = structural_similarity(Iclean[i,:,:,:], Img[i,:,:,:], data_range=1.0, channel_axis=2, 
                                   gaussian_weights=True, use_sample_covariance=False)
        SSIM.append(ss)
    return sum(SSIM)/len(SSIM)


def ssim(img, imclean):
    """
    compute ssim consistent with Matlab ssim function.
    """
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    return structural_similarity_matlab(img, imclean).item()




