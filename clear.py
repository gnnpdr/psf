from skimage.restoration import wiener, richardson_lucy
from olimp.precompensation.optimization.montalto import montalto, MontaltoParameters
from PIL import Image
import re
import numpy as np
import torch
import os
import torch.nn.functional as F

def img2float(img):
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_float = (img_array / 255).astype(np.float32)
    return img_float

#здесь уже достаточно просто дополнить
def pad_psf(psf, img):
    h_img, w_img = img.shape
    h_psf, w_psf = psf.shape
    
    pad_h = h_img - h_psf
    pad_w = w_img - w_psf
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf).float()
    
    psf_padded = F.pad(psf, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    
    psf_padded = psf_padded / psf_padded.sum()
    return psf_padded




os.makedirs('results/wiener', exist_ok = True)
os.makedirs('results/rich_lucy', exist_ok = True)
os.makedirs('results/tv', exist_ok = True)

file_path = 'results/blurred/1_psf_0_noise_0.1.png'
file_name = os.path.basename(file_path)
img = Image.open(file_path)
name_without_ext = os.path.splitext(file_name)[0]
match = re.search(r'(\d+)_psf_(\d+)_noise_([\d.]+)', name_without_ext)
if match:
    original_name = match.group(1)
    psf_idx = int(match.group(2))
    noise = float(match.group(3))

noise = 0.1

img_float = img2float(img)


psf_np = np.load(f'results/psf/{original_name}_psf_{psf_idx}.npy')


result = wiener(img_float, psf_np, noise)
result_img = (result * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(result_img).save(f'results/wiener/{original_name}_clear_noise_{noise}_restored.png')

#num_iter = 4
#result = richardson_lucy(img_float, psf_np, num_iter)
#result_img = (result * 255).clip(0, 255).astype(np.uint8)
#Image.fromarray(result_img).save(f'results/rich_lucy/{name_without_ext}_{num_iter}_restored.png')

#psf = pad_psf(psf_np, img_float)
#img_tensor = torch.from_numpy(img_float).float()
#
#if len(img_tensor.shape) == 2:
#    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 
#if len(psf.shape) == 2:
#    psf_tensor = psf.unsqueeze(0).unsqueeze(0)
#
#params = MontaltoParameters(
#    lr=1e-2,           
#    theta=1e-6,  
#    tau=2e-5,   
#    Lambda=65.0,       
#    c_high=1.0,        
#    c_low=0.0,        
#    gap=0.01,
#)
#
#result_tensor = montalto(img_tensor, psf_tensor, params)
#result_tensor = torch.fft.ifftshift(result_tensor, dim = (-2, -1)) 
#result = result_tensor[0, 0].cpu().detach().numpy()
#result_img = (result * 255).clip(0, 255).astype(np.uint8)
#Image.fromarray(result_img).save(f'results/tv/{name_without_ext}_restored.png')