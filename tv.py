from skimage.restoration import wiener, richardson_lucy
from olimp.precompensation.optimization.montalto import montalto, MontaltoParameters
from PIL import Image
import re
import numpy as np
import torch
import os
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from olimp.processing import fftshift

def img2float(img):
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_float = (img_array / 255).astype(np.float32)
    return img_float

#здесь уже достаточно просто допролнить
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

def tv(orig_img_np, img, psf):
    #возьмем короче каждое клевое маленькое изображение, посчитаем коэффициенты для исходного, искаженного и восстановленного для разных параметров
    #получившиеся изображения можно не сохранять, достаточно оставить параметры, а потом сгенерировать заново.
    theta = check_theta(orig_img_np, img, psf)
    params = MontaltoParameters(
        lr=1e-2,
        theta=theta,
        tau=2e-5,
        Lambda=65.0,
        gap=0.01
    )

    img_shifted = fftshift(img)
    result_tensor = montalto(img_shifted, psf, params)
    result = result_tensor[0, 0].cpu().detach().numpy()
    result_img = (result * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(result_img).save(f'results/tv/{name_without_ext}_theta_{theta}_restored.png')
    

def check_theta(orig_img_np, img, psf):
    theta_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    results = []

    for theta in theta_values:

        params = MontaltoParameters(
            lr=1e-2,
            theta=theta,
            tau=2e-5,
            Lambda=65.0,
            gap=0.01
        )

        img_shifted = fftshift(img)
        result_tensor = montalto(img_shifted, psf, params)
        result_np = result_tensor[0, 0].cpu().detach().numpy()
        psnr_value = psnr(orig_img_np, result_np)
        results.append((theta, psnr_value, result_np))

    best_theta, best_psnr, best_result = max(results, key=lambda x: x[1])
    return best_theta



#------------------------------------

os.makedirs('results/tv', exist_ok = True)


#folder_path = 'results/blured'
#
#all_files = os.listdir(folder_path)
#
#for filename in all_files:
#    file_path = os.path.join(folder_path, filename)
#    if os.path.isfile(file_path):
file_path = 'results/blurred/1_psf_0_noise_0.1.png'
file_name = os.path.basename(file_path)
img = Image.open(file_path)
name_without_ext = os.path.splitext(file_name)[0]
match = re.search(r'(\d+)_psf_(\d+)_noise_', name_without_ext)
if match:
    original_name = match.group(1)
    psf_idx = int(match.group(2))
orig_img = Image.open(f'image/{original_name}.png')
orig_img_np = np.array(orig_img)
img_float = img2float(img)
psf_np = np.load(f'results/psf/{original_name}_psf_{psf_idx}.npy')
psf = pad_psf(psf_np, img_float)
img_tensor = torch.from_numpy(img_float).float()
if len(img_tensor.shape) == 2:
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 
if len(psf.shape) == 2:
    psf_tensor = psf.unsqueeze(0).unsqueeze(0)
tv(orig_img_np, img_tensor, psf_tensor)