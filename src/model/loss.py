import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

mse = nn.MSELoss(reduction='mean')
l1_loss = nn.L1Loss(reduction='mean')

def get_covariance(x1, x2):
    """
    Compute covariance of 4D tensors x1 and x2.
    The first two dimensions are not touched. The rest dimmensions of x1 and x2 are 
    reshaped into one dimension, and stacked into a 2D matrix. Then, torch.cov is called
    to compute the covariance matrix (2 x 2). At last, the upper corner result is fetched.
    """
    B, C, H, W = x1.shape
    res = torch.zeros(B, C, 1).to(x1.device)
    x1 = x1.reshape(B, C, -1)
    x2 = x2.reshape(B, C, -1)
    for i in range(B):
        for j in range(C):
            x = torch.stack([x1[i][j], x2[i][j]], dim=0)
            #print("DEBUG x", x.shape)
            res[i][j] = torch.cov(x)[0][1]
    return res

def patched_cov_new3(tensor1, tensor2, block_size=3, stride=1, padding=1):
    """
    compute pixelwise covariance for 4D input tensor1 & tensor2 (B, C, H, W) in patches
    """
    
    B, C, H, W = tensor1.shape
    #mean_tensor1 = compute_mean(tensor1).unsqueeze(dim=2).repeat_interleave(H, dim=2).unsqueeze(dim=3).repeat_interleave(W, dim=3)
    #mean_tensor2 = compute_mean(tensor2).unsqueeze(dim=2).repeat_interleave(H, dim=2).unsqueeze(dim=3).repeat_interleave(W, dim=3)
    mean_tensor1 = F.avg_pool2d(tensor1, block_size, stride, padding)
    mean_tensor2 = F.avg_pool2d(tensor2, block_size, stride, padding)
    
    return (tensor1 - mean_tensor1)*(tensor2 - mean_tensor2)

def patched_var_new3(tensor, block_size=3, stride=1, padding=1):
    """
    compute pixelwise variance for 4D input tensor1 (B, C, H, W) in patches
    """
    return patched_cov_new3(tensor, tensor, block_size, stride, padding)


def patched_cov_new4(tensor1, tensor2, block_size, stride, padding, gaussian_filter):
    """
    compute pixelwise covariance for 4D input tensor1 & tensor2 (B, C, H, W) in patches
    """
    # using gaussian filter
    mean_tensor1 = gaussian_filter(tensor1)
    mean_tensor2 = gaussian_filter(tensor2)
    return (tensor1 - mean_tensor1)*(tensor2 - mean_tensor2)
    

def patched_var_new4(tensor, block_size, stride, padding, gaussian_filter):
    """
    compute variance for 4D input tensor1 (B, C, H, W) in patches
    """
    return patched_cov_new4(tensor, tensor, block_size, stride, padding, gaussian_filter)

def total_variation_loss(input_tensor, weight=0.1):
    """
    compute total variation loss for 4D input_tensor.
    """
    B, C, H, W = input_tensor.shape
    tv_horizontal = torch.nn.functional.mse_loss(input_tensor[:,:, 1:, :], input_tensor[:,:,:-1, :])
    tv_vertical = torch.nn.functional.mse_loss(input_tensor[:,:, :, 1:], input_tensor[:,:,:, :-1])
    return weight*(tv_horizontal + tv_vertical)

def loss_aug(sim_clean, clean):
    loss = mse(sim_clean, clean)
    return loss

def compute_std(input_tensor):
    """
    compute standard deviation for 4D input_tensor (B, C, H, W) at dimensions (H, W)
    """
    B, C, H, W = input_tensor.shape
    return torch.std(input_tensor.reshape(B, C, -1), -1)

def compute_mean(input_tensor):
    """
    compute mean value for 4D input_tensor (B, C, H, W) at dimensions (H, W)
    """
    B, C, H, W = input_tensor.shape
    return torch.mean(input_tensor.reshape(B, C, -1), -1)

def im2col(x, block_size, stride):
    """
    rearrange 4D tensor x (B, C, H, W) to 3D tensor (B, C*block_size*blocksize, L)
    """
    return torch.nn.functional.unfold(x, block_size, stride=stride)

def patched_std_loss(tensor1, tensor2, block_size=10, stride=10, topk=0, across_batch=False):
    """
    compute standard deviation loss  for 4D input tensor1 & tensor2 (B, 1, H, W) in patches
    """
    
    # After im2col, tensor1 & tensor2 become three dimensional (B, block_size*block_size, L)
    tensor1 = im2col(tensor1, block_size, stride)
    tensor2 = im2col(tensor2, block_size, stride)

    if topk > 0:
        # select topk smallest patches according to standard deviation
        std1 = torch.std(tensor1, dim=1)
        std2 = torch.std(tensor2, dim=1)
        #print("DEBUG: std1", std1.shape, 'tensor1', tensor1.shape)
        _, indices = torch.topk(std1, topk, dim=-1, largest=False)
        #std2 = std2[:,indices]
    else:
        indices = range(tensor1.shape[-1])
        
    if across_batch:
        # Combine the first two dimensions if across_batch == True
        # In this case, tensor1 & tensor1 become two dimensional (B*block_size*block_size, len(indicies))
        # 这里实际计算的是同一个像素位置的不同块之间像素的标准差 （在indicies维度上计算std）
        tensor1 = tensor1[:, :, indices].reshape(-1, len(indices))
        tensor2 = tensor2[:, :, indices].reshape(-1, len(indices))
        std1 = torch.std(tensor1, dim=1)
        std2 = torch.std(tensor2, dim=1)
        return mse(std1.mean(-1), std2.mean(-1)) # mse([1],[1])
    else:   
        # compute std [B, L]
        std1 = torch.std(tensor1, dim=1)
        std2 = torch.std(tensor2, dim=1)
        return mse(std1, std2)
    
def patched_mean_loss(tensor1, tensor2, block_size=8, stride=1):
    """
    compute mean loss  for 4D input tensor1 & tensor2 (B, 1, H, W) in patches
    """
    
    # After im2col, tensor1 & tensor2 become three dimensional (B, block_size*block_size, L)
    tensor1 = im2col(tensor1, block_size, stride)
    tensor2 = im2col(tensor2, block_size, stride)
        
    # compute std [B, L]
    mean1 = torch.mean(tensor1, dim=1)
    mean2 = torch.mean(tensor2, dim=1)
    return mse(mean1, mean2)


def patched_std(tensor, block_size=2, stride=1, across_batch=False):
    """
    compute standard deviation for 4D input tensor1 (B, 1, H, W) in patches
    """
    
    # After im2col, tensor becomes three dimensional (B, block_size*block_size, L)
    tensor = im2col(tensor, block_size, stride)
    if across_batch:
        # Combine the first two dimensions if across_batch == True
        # In this case, tensor1 & tensor1 become two dimensional (B*block_size*block_size, len(indicies))
        # 这里实际计算的是同一个像素位置的不同块之间像素的标准差 （在indicies维度上计算std）
        tensor = tensor.reshape(-1, tensor.shape[-1])
        std_values = torch.std(tensor, dim=1)
        return std_values.mean(-1)
    else:   
        # compute std [B, L]
        std_values = torch.std(tensor, dim=1)
        return std_values

def patched_mean(tensor, block_size=15, stride=1, padding=7):
    """
    compute mean for 4D input tensor1 (B, C, H, W) in patches
    """
    mean_tensor = F.avg_pool2d(tensor, kernel_size=block_size, stride=stride, padding=padding)
    return mean_tensor

    
def batch_cov(x, y, dim=1, indices=None):
    """
    compute covariance of two input tensors with size (B, F, L) at dimension dim.
    """
    F = x.shape[dim]
    if F <= 1:
        F = 2
        
    if indices is not None:
        x = x[:, indices, :]
        y = y[:, indices, :]
        F = len(indices)
    
    x_bar = x.mean(dim).unsqueeze(dim)
    y_bar = y.mean(dim).unsqueeze(dim)
    return ((x-x_bar)*(y-y_bar)).sum(dim) / (F - 1)
    
def patched_cov(tensor1, tensor2, block_size=4, stride=4, indices=None):
    """
    compute covariance for 4D input tensor1 & tensor2 (B, 1, H, W) in patches
    """
    
    # After im2col, tensor1 & tensor2 become three dimensional (B, block_size*block_size, L)
    tensor1 = im2col(tensor1, block_size, stride)
    tensor2 = im2col(tensor2, block_size, stride)
    return batch_cov(tensor1, tensor2, dim=1, indices=indices)

def patched_var(tensor, block_size, stride, indices=None):
    """
    compute variance for 4D input tensor (B, 1, H, W) in patches
    """
    return patched_cov(tensor, tensor, block_size, stride, indices)
    #return patched_std(tensor, block_size, stride) **2 # Faster using native API.


def loss_main(input_noisy, clean, noise, gaussian_filter):
    """
    input_noisy: 输入噪声图
    clean:input_noisy分离出的无噪图
    noise: 模型估计出的噪声
    gaussian_filter: 高斯滤波器
    """
    # 数据保真
    loss_data = mse(input_noisy, clean + noise) 
    
    # 均值约束：mean(input_noisy) == mean(clean)，防止出现模型输出恒为0解的平凡解
    loss_mean_value = mse(compute_mean(input_noisy), compute_mean(clean))

    # 方差正则化项来避免平凡解
    coeffi = 4.0
    block_size = 7
    stride = 1
    
    
    patched_var_input = patched_var_new3(input_noisy, block_size, stride, padding=block_size//2)
    patched_var_noise = patched_var_new3(noise, block_size, stride,  padding=block_size//2)
    patched_var_clean = patched_var_new3(clean, block_size, stride,  padding=block_size//2)
    patched_cov_res = patched_cov_new3(noise, clean, block_size, stride,  padding=block_size//2)

    std_diff = patched_var_input - patched_var_noise
    reg_diff = std_diff - (patched_var_clean + patched_cov_res)
    loss_var = torch.mean(std_diff**2) + coeffi * torch.mean(reg_diff**2)
        
    loss = loss_data + loss_mean_value + loss_var
    return loss


if __name__ == '__main__':
    print('loss')
