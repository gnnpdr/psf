from PIL import Image
import re
import numpy as np
import torch
import os
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from olimp.processing import fftshift
import matplotlib.pyplot as plt
from skimage.restoration import wiener

def img2float(img):
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_float = (img_array / 255).astype(np.float32)
    return img_float

def pad_psf(psf, img):
    h_img, w_img = img.shape
    h_psf, w_psf = psf.shape
    
    if isinstance(psf, np.ndarray):
        psf_tensor = torch.from_numpy(psf).float()
    else:
        psf_tensor = psf.clone()
    
    psf_shifted = fftshift(psf_tensor)
    
    pad_h = h_img - h_psf
    pad_w = w_img - w_psf
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    psf_padded = F.pad(psf_shifted, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    
    psf_padded = psf_padded / psf_padded.sum()
    return psf_padded


def tensor2np(tensor):
    if isinstance(tensor, torch.Tensor):
        if len(tensor.shape) == 4:
            np = tensor[0, 0].cpu().detach().numpy()
        elif len(tensor.shape) == 3:
            np = tensor[0].cpu().detach().numpy()
        elif len(tensor.shape) == 2:
            np = tensor.cpu().detach().numpy()
        else:
            np = tensor.cpu().detach().numpy()
    else:
        np = tensor.copy()

    return np


def get_img_params(file_name):
    name_without_ext = os.path.splitext(file_name)[0]

    match = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)', name_without_ext)
    if match:
        original_name = match.group(1)
        psf_ind = int(match.group(2))
        noise = float(match.group(3))

    return original_name, psf_ind, noise

def get_img_tensor(img):
    
    img_float = img2float(img)
    img_tensor = torch.from_numpy(img_float).float()

    return img_tensor


def make_trio(file_name, orig_np, blurred_np, res_np):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    axes[0].imshow(orig_np, cmap='gray')
    axes[0].set_title('original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(blurred_np, cmap='gray')
    axes[1].set_title('blurred', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(res_np, cmap='gray')
    axes[2].set_title('result', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'results/res_wiener_skimage_trios/{file_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

#--------------------------------------------------------------------------------------------------

def make_save_wiener(img_tensor, psf_tensor, noise, original_name):
    img_np = tensor2np(img_tensor)
    psf_np = tensor2np(psf_tensor)
        
    res_tensor = wiener(img_np, psf_np, noise)

    res_np = tensor2np(res_tensor)
        
    res_np = np.clip(res_np, 0, 1)
    result_img = (res_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(result_img).save(f'results/wiener_skimage/{original_name}_noise_{noise}_restored.png')
    return res_np


def make_wiener(img_tensor, psf_tensor, noise):
    img_np = tensor2np(img_tensor)
    psf_np = tensor2np(psf_tensor)
        
    res_tensor = wiener(img_np, psf_np, noise)

    return res_tensor



#------------------------------------------------------------------------------------------------------

def check_param(metric_func, blurred_tensor, psf_tensor, noise, orig_tensor):
    
    steps = [10, 5, 2]
    min_param = 1e-7
    max_param = 20.0

    cur_param = noise
    res_tensor = make_wiener(blurred_tensor, psf_tensor, cur_param)
    best_metric = metric_func(orig_tensor, res_tensor)
    best_param = cur_param

    for step in steps:
        test_param = best_param
        improved = True
        
        while improved:
            improved = False
            new_param = test_param // step
            if new_param < min_param:
                break
                
            res_tensor = make_wiener(blurred_tensor, psf_tensor, new_param)
            new_metric = metric_func(orig_tensor, res_tensor)
            
            if new_metric > best_metric:
                best_metric = new_metric
                best_param = new_param
                test_param = new_param
                improved = True
        
        test_param = best_param
        improved = True
        
        while improved:
            improved = False
            new_param = test_param * step
            if new_param > max_param:
                break
                
            res_tensor = make_wiener(blurred_tensor, psf_tensor, new_param)
            new_metric = metric_func(orig_tensor, res_tensor)
            
            if new_metric > best_metric:
                best_metric = new_metric
                best_param = new_param
                test_param = new_param
                improved = True

    return best_param, best_metric


def calc_ssim(orig_tensor, res_tensor):
    orig_np = tensor2np(orig_tensor)
    res_np = tensor2np(res_tensor)
    
    orig_np = np.clip(orig_np, 0, 1)
    res_np = np.clip(res_np, 0, 1)

    ssim_val = ssim(orig_np, res_np, data_range=1.0)
    return ssim_val


def calc_psnr(orig_tensor, res_tensor):
    orig_np = tensor2np(orig_tensor)
    res_np = tensor2np(res_tensor)
    
    orig_np = np.clip(orig_np, 0, 1)
    res_np = np.clip(res_np, 0, 1)
    
    psnr_val = psnr(orig_np, res_np, data_range=1.0)
    return psnr_val

#----------------------------------------------------------------------------------------------------------

os.makedirs('results/wiener_skimage', exist_ok = True)
os.makedirs('results/res_wiener_skimage_trios', exist_ok = True)

folder_path = 'results/blurred'

all_files = os.listdir(folder_path)

for filename in all_files:
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        
        file_name = os.path.basename(file_path)
        orig_name, psf_ind, noise = get_img_params(file_name)

        orig = Image.open(f'image/{orig_name}.tiff')
        orig_tensor = get_img_tensor(orig)

        blurred = Image.open(file_path)
        blurred_float = img2float(blurred)
        blurred_tensor = torch.from_numpy(blurred_float).float()

        psf_np = np.load(f'results/psf/{orig_name}_psf_{psf_ind}.npy')
        blurred_tensor = torch.from_numpy(blurred_float).float()
        if len(blurred_tensor.shape) == 2:
            blurred_tensor = blurred_tensor.unsqueeze(0).unsqueeze(0)

        psf = pad_psf(psf_np, blurred_float)
        psf_tensor = psf.unsqueeze(0).unsqueeze(0)

        init_psnr_val = calc_psnr(orig_tensor, blurred_tensor)
        init_ssim_val = calc_ssim(orig_tensor, blurred_tensor)
        best_psnr_noise, psnr_val = check_param(calc_psnr, blurred_tensor, psf_tensor, noise, orig_tensor)
        best_ssim_noise, ssim_val = check_param(calc_ssim, blurred_tensor, psf_tensor, noise, orig_tensor)
        with open('results/wiener_skimage_res.txt', 'a') as f:
            f.write(f"file {file_name}\ninitial psnr val {init_psnr_val}, psnr val after restoration {psnr_val}, noise {best_psnr_noise}\ninitial ssim val {init_ssim_val}, psnr val after restoration {ssim_val}, noise, {best_ssim_noise}\n---------------\n")
        
        psnr_name = f'{file_name}_best_psnr'
        ssim_name = f'{file_name}_best_ssim'
        res_np = make_save_wiener(blurred_tensor, psf_tensor, best_psnr_noise, psnr_name)
        res_np = make_save_wiener(blurred_tensor, psf_tensor, best_ssim_noise, ssim_name)
        orig_np = tensor2np(orig_tensor)
        blurred_np = tensor2np(blurred_tensor)
        make_trio(file_name, orig_np, blurred_np, res_np)