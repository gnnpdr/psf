from skimage.restoration import richardson_lucy
from PIL import Image
import re
import numpy as np
import os
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.ndimage import shift

def img2float(img):
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_float = (img_array / 255).astype(np.float32)

    return img_float

def get_img_params(file_name):
    match = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)', file_name)
    if match:
        original_name = match.group(1)
        psf_ind = int(match.group(2))
        noise = float(match.group(3))

    return original_name, psf_ind, noise


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
    plt.savefig(f'results/restored_trios_rich_lucy/{file_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

#--------------------------------------------------------------------------------------------------

def rich_lucy_save(img_float, psf_np, it, orig_name):
    psf_centered = center_psf(psf_np)
    res_np = richardson_lucy(img_float, psf_centered, it)
    result_img = (res_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(result_img).save(f'results/restored_rich_lusy/{orig_name}_it_{it}_restored.png')
    return res_np


def rich_lucy(img_float, psf_np, it):
    psf_centered = center_psf(psf_np)
    res_np = richardson_lucy(img_float, psf_centered, it)
    return res_np

#------------------------------------------------------------------------------------------------------


def check_iterations(metric_func, blurred_float, psf_np, orig_float):
    
    steps = [10, 5, 2]
    min_it = 1
    max_it = 1000

    cur_it = 50
    res_np = rich_lucy(blurred_float, psf_np, cur_it)
    best_metric = metric_func(orig_float, res_np)
    best_it = cur_it

    for step in steps:
        test_it = best_it
        improved = True
        
        while improved:
            improved = False
            new_it = test_it // step
            if new_it < min_it:
                break
                
            res_np = rich_lucy(blurred_float, psf_np, new_it)

            new_metric = metric_func(orig_float, res_np)
            
            if new_metric > best_metric:
                best_metric = new_metric
                best_it = new_it
                test_it = new_it
                improved = True
        
        test_it = best_it
        improved = True
        
        while improved:
            improved = False
            new_it = test_it * step
            if new_it > max_it:
                break
                
            res_np = rich_lucy(blurred_float, psf_np, new_it)

            new_metric = metric_func(orig_float, res_np)
            
            if new_metric > best_metric:
                best_metric = new_metric
                best_it = new_it
                test_it = new_it
                improved = True

    return best_it, best_metric


def center_psf(psf):
    h, w = psf.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.indices(psf.shape)
    center_y = np.sum(y * psf) / psf.sum()
    center_x = np.sum(x * psf) / psf.sum()
    
    shift_y = cy - center_y
    shift_x = cx - center_x
    
    if abs(shift_y) > 0.5 or abs(shift_x) > 0.5:
        psf = shift(psf, (shift_y, shift_x), mode='constant')
    
    return psf

def calc_ssim(orig_float, res_np):
    ssim_val = ssim(orig_float, res_np, data_range=1.0)
    return ssim_val



def calc_psnr(orig_float, res_np):
    metric_val = psnr(orig_float, res_np, data_range=1.0)
    return metric_val

#----------------------------------------------------------------------------------------------------------

os.makedirs('results/restored_rich_lusy', exist_ok = True)
os.makedirs('results/restored_trios_rich_lucy', exist_ok = True)

folder_path = 'results/blurred'

all_files = os.listdir(folder_path)

for filename in all_files:
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        orig_name, psf_ind, noise = get_img_params(file_name)

        orig = Image.open(f'image/{orig_name}.tiff')
        orig_float = img2float(orig)

        blurred = Image.open(file_path)
        blurred_float = img2float(blurred)

        psf_np = np.load(f'results/psf/{orig_name}_psf_{psf_ind}.npy')
        if abs(psf_np.sum() - 1.0) > 1e-6:
            psf_np = psf_np / psf_np.sum()

        init_psnr_val = calc_psnr(orig_float, blurred_float)
        init_ssim_val = calc_ssim(orig_float, blurred_float)

        #best_psnr_iterations, psnr_val = check_iterations(calc_psnr, blurred_float, psf_np, orig_float)
        #best_ssim_iterations, ssim_val = check_iterations(calc_ssim, blurred_float, psf_np, orig_float)
        #file_name_psnr = f'{file_name}_best_psnr'
        #file_name_ssim = f'{file_name}_best_ssim'

        #with open('results/rich_lucy_res.txt', 'a') as f:
        #    f.write(f"file {file_name_psnr}\ninitial psnr val {init_psnr_val}, psnr val after restoration {psnr_val}, iterations {best_psnr_iterations}\nfile {file_name_ssim}\ninitial ssim val {init_ssim_val}, ssim val after restoration {ssim_val}, iterations {best_ssim_iterations}\n---------------\n")
        
        #res_np_psnr = rich_lucy_save(blurred_float, psf_np, best_psnr_iterations, file_name_psnr)
        #res_np_ssim = rich_lucy_save(blurred_float, psf_np, best_ssim_iterations, file_name_ssim)
        #make_trio(file_name_psnr, orig_float, blurred_float, res_np_psnr)
        #make_trio(file_name_ssim, orig_float, blurred_float, res_np_ssim)

        

        res_np = rich_lucy_save(blurred_float, psf_np, 5, file_name)
        res_psnr_val = calc_psnr(orig_float, res_np)
        res_ssim_val = calc_ssim(orig_float, res_np)
        with open('results/rich_lucy_res_20.txt', 'a') as f:
            f.write(f"file {file_name}\ninitial psnr val {init_psnr_val}, psnr val after restoration {res_psnr_val}\ninitial ssim val {init_ssim_val}, ssim val after restoration {res_ssim_val}\n---------------\n")
        make_trio(file_name, orig_float, blurred_float, res_np)
