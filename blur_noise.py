from PIL import Image
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from olimp.simulate.psf_gauss import PSFGauss
from olimp.precompensation.nn.dataset.psf_gauss import PsfGaussDataset
from olimp.simulate.refraction_distortion import RefractionDistortion
from olimp.processing import fft_conv, fftshift
from ballfish import DistributionParams 
import matplotlib.pyplot as plt
import os
from skimage.util import random_noise

#----------------------------------------------------------------------

def get_file_name(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    return name

def check_color(img_array):
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        return 'grayscale'
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 3:
            return 'rgb'
        elif img_array.shape[2] == 4:
            return 'rgba'

def tiff2float(img):
    img_gray = img.convert('L')
    img_array = numpy.array(img_gray)
    img_float = (img_array / 255).astype(numpy.float32)
    return img_float

def pad_psf(psf):
    psf_shifted = fftshift(psf)
    
    h_psf, w_psf = psf_shifted.shape
    
    pad_h = h_img - h_psf
    pad_w = w_img - w_psf
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    psf_padded = F.pad(psf_shifted, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    
    psf_padded = psf_padded / psf_padded.sum()
    return psf_padded

def blur(img, psf):
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    psf_tensor = psf.unsqueeze(0).unsqueeze(0)
    
    blurred_tensor = fft_conv(img_tensor, psf_tensor)

    blurred_tensor = torch.fft.ifftshift(blurred_tensor, dim = (-2, -1)) 
    return blurred_tensor

def make_trio(img_name, img_np, noisy_np, psf_2d, psf_idx, noise_level):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    type = check_color(img_np)
    if type == 'grayscale':
        axes[0].imshow(img_np, cmap = 'gray')
    else: 
        axes[0].imshow(img_np)
    axes[0].set_title('Оригинал', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(noisy_np, cmap='gray')
    axes[1].set_title(f'Размытое (шум {noise_level})', fontsize=12)
    axes[1].axis('off')

    psf_show = psf_2d.cpu().numpy()
    from scipy.ndimage import zoom
    psf_big = zoom(psf_show, 10, order=3)

    axes[2].imshow(psf_big, cmap='hot')
    axes[2].set_title(f'ФРТ {psf_idx} (увеличено)', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'results/{img_name}_psf_{psf_idx}_noise_{noise_level}.png', dpi=150, bbox_inches='tight')
    plt.close()

#---------------------------------------------------------------

os.makedirs('results', exist_ok = True)

image_paths = [
    'image/1.png',
    'image/4.1.08.tiff', 
    'image/4.2.05.tiff',
    'image/7.1.01.tiff',
    'image/7.1.04.tiff',
    'image/7.2.01.tiff'
]

psf_dataset = PsfGaussDataset(
    width=15,
    height=15,
    center_x = ({'name': 'constant', 'value': 0}),
    center_y = ({'name': 'constant', 'value': 0}),
    theta = ({'name': 'uniform', 'a': 0, 'b': 180}),
    sigma_x = ({'name': 'uniform', 'a': 1.0, 'b': 5.0}),
    sigma_y = ({'name': 'uniform', 'a': 1.0, 'b': 5.0}),
    seed=42,
    size=5
)

noise_params = [0.01, 0.05, 0.1]
dataset = []

#----------------------------------------------------------

for img_path in image_paths:
    orig_img = Image.open(img_path) 
    orig_img_array = numpy.array(orig_img)
    img = tiff2float(orig_img)

    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    h_img, w_img = img.shape

    for psf_idx in range(5):
        psf = psf_dataset[psf_idx]
        psf_2d = psf.squeeze(0)
        psf_padded = pad_psf(psf_2d)

        blurred_tensor = blur(img, psf_padded)
        blurred_np = blurred_tensor[0, 0].cpu().numpy()

        for noise_level in noise_params:
            noisy_np = random_noise(blurred_np, mode = 'gaussian', mean = 0, var = noise_level**2, clip = True)
            img_name = get_file_name(img_path)
            make_trio(img_name, orig_img_array, noisy_np, psf_2d, psf_idx, noise_level)