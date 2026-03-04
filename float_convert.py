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

def tiff2float(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    img_arr = numpy.array(img_gray)
    img_float = (img_arr / 255).astype(numpy.float32)
    return img_float

def ifftshift(tensor, dim = (-2, -1)):
    return torch.fft.ifftshift(tensor, dim=dim)

#---------------------------------------------------------------

os.makedirs('results', exist_ok=True)
os.makedirs('results/images', exist_ok=True)

img = tiff2float('image/1.png')

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

img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
h_img, w_img = img.shape

for psf_idx in range(5):
    psf = psf_dataset[psf_idx]
    psf_2d = psf.squeeze(0)
    psf_shifted = fftshift(psf_2d)
    
    h_psf, w_psf = psf_shifted.shape
    
    pad_h = h_img - h_psf
    pad_w = w_img - w_psf
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    psf_padded = F.pad(psf_shifted, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    
    psf_padded = psf_padded / psf_padded.sum()
    
    psf_tensor = psf_padded.unsqueeze(0).unsqueeze(0)
    
#    
#    # ПРОВЕРЯЕМ, ЧТО ФРТ В ЦЕНТРЕ
#    # Создаем визуализацию ФРТ для проверки
#    plt.figure(figsize=(10, 4))
#    
#    plt.subplot(1, 2, 1)
#    plt.imshow(psf_padded.cpu().numpy(), cmap='hot')
#    plt.title(f'ФРТ {psf_idx} (полная)')
#    plt.colorbar()
#    
#    plt.subplot(1, 2, 2)
#    # Показываем центр ФРТ (увеличено)
#    center = psf_padded[120:140, 120:140].cpu().numpy()
#    plt.imshow(center, cmap='hot')
#    plt.title('Центр ФРТ (20x20)')
#    plt.colorbar()
#    
#    plt.savefig(f'results/psf_check_{psf_idx}.png', dpi=150, bbox_inches='tight')
#    plt.close()
    
    blurred_tensor = fft_conv(img_tensor, psf_tensor)

    blurred_tensor = ifftshift(blurred_tensor, dim = (-2, -1)) 
    
    blurred_np = blurred_tensor[0, 0].cpu().numpy()
    
#    blurred_img = (blurred_np * 255).astype(np.uint8)
#    Image.fromarray(blurred_img).save(f'results/blurred_{psf_idx}.png')
    
    # Добавляем шум
    for noise_level in noise_params:
        noisy_np = random_noise(blurred_np, mode='gaussian', mean=0, var=noise_level**2, clip=True)
        
        noisy_img = (noisy_np * 255).astype(np.uint8)
        Image.fromarray(noisy_img).save(f'results/images/blurred_{psf_idx}_noise{noise_level}.png')
        
        dataset.append({
            'original': img_tensor.clone(),
            'psf': psf_2d.clone(),
            'blurred_noisy': torch.from_numpy(noisy_np).float().unsqueeze(0).unsqueeze(0),
            'psf_idx': psf_idx,
            'noise_level': noise_level
        })

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

## Первая строка - "резаное" (без ifftshift)
#axes[0, 0].imshow(img, cmap='gray')
#axes[0, 0].set_title('Оригинал')
#axes[0, 0].axis('off')
#
## Вторая строка - исправленное (с ifftshift)
#axes[1, 0].imshow(img, cmap='gray')
#axes[1, 0].set_title('Оригинал')
#axes[1, 0].axis('off')
#
#axes[1, 1].imshow(blurred_np, cmap='gray')
#axes[1, 1].set_title('С ifftshift (нормальное)')
#axes[1, 1].axis('off')
#
#axes[1, 2].imshow(blurred_np[128:148, 128:148], cmap='gray')
#axes[1, 2].set_title('Уголок нормального')
#axes[1, 2].axis('off')
#
#plt.tight_layout()
#plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
#plt.close()