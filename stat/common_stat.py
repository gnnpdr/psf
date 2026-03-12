import re
import numpy as np
import matplotlib.pyplot as plt

def parse_richardson_file_psftype(filepath, psf_type):
    results = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    blocks = content.split('---------------')
    
    for block in blocks:
        if not block.strip():
            continue
        
        psnr_match = re.search(r'file (.+?_best_psnr).*?initial psnr val ([\d.]+), psnr val after restoration ([\d.]+), iterations (\d+)', block, re.DOTALL)
        if psnr_match:
            filename = psnr_match.group(1)
            params = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)_best_psnr', filename)
            if params:
                results.append({
                    'method': 'Richardson-Lucy',
                    'psf_type': psf_type,
                    'type': 'psnr',
                    'original_name': params.group(1),
                    'psf_idx': int(params.group(2)),
                    'true_noise': float(params.group(3)),
                    'init_val': float(psnr_match.group(2)),
                    'restored_val': float(psnr_match.group(3)),
                    'param': int(psnr_match.group(4)),
                    'improvement': float(psnr_match.group(3)) - float(psnr_match.group(2))
                })
        
        ssim_match = re.search(r'file (.+?_best_ssim).*?initial ssim val ([\d.]+), ssim val after restoration ([\d.]+), iterations (\d+)', block, re.DOTALL)
        if ssim_match:
            filename = ssim_match.group(1)
            params = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)_best_ssim', filename)
            if params:
                results.append({
                    'method': 'Richardson-Lucy',
                    'psf_type': psf_type,
                    'type': 'ssim',
                    'original_name': params.group(1),
                    'psf_idx': int(params.group(2)),
                    'true_noise': float(params.group(3)),
                    'init_val': float(ssim_match.group(2)),
                    'restored_val': float(ssim_match.group(3)),
                    'param': int(ssim_match.group(4)),
                    'improvement': float(ssim_match.group(3)) - float(ssim_match.group(2))
                })
    
    return results

def parse_wiener_file_psftype(filepath, method_name, psf_type):
    results = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    blocks = content.split('---------------')
    
    for block in blocks:
        if not block.strip():
            continue
        
        psnr_match = re.search(r'file (.+?_best_psnr).*?initial psnr val ([\d.]+), psnr val after restoration ([\d.]+), noise ([\d.]+)', block, re.DOTALL)
        if psnr_match:
            filename = psnr_match.group(1)
            params = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)_best_psnr', filename)
            if params:
                results.append({
                    'method': method_name,
                    'psf_type': psf_type,
                    'type': 'psnr',
                    'original_name': params.group(1),
                    'psf_idx': int(params.group(2)),
                    'true_noise': float(params.group(3)),
                    'init_val': float(psnr_match.group(2)),
                    'restored_val': float(psnr_match.group(3)),
                    'param': float(psnr_match.group(4)),
                    'improvement': float(psnr_match.group(3)) - float(psnr_match.group(2))
                })
        
        ssim_match = re.search(r'file (.+?_best_ssim).*?initial ssim val ([\d.]+), ssim val after restoration ([\d.]+), noise,? ([\d.]+)', block, re.DOTALL)
        if ssim_match:
            filename = ssim_match.group(1)
            params = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)_best_ssim', filename)
            if params:
                results.append({
                    'method': method_name,
                    'psf_type': psf_type,
                    'type': 'ssim',
                    'original_name': params.group(1),
                    'psf_idx': int(params.group(2)),
                    'true_noise': float(params.group(3)),
                    'init_val': float(ssim_match.group(2)),
                    'restored_val': float(ssim_match.group(3)),
                    'param': float(ssim_match.group(4)),
                    'improvement': float(ssim_match.group(3)) - float(ssim_match.group(2))
                })
    
    return results

def plot_methods_comparison_by_psf(all_results):
    psnr_results = [r for r in all_results if r['type'] == 'psnr']
    
    methods = sorted(set([r['method'] for r in psnr_results]))
    noise_levels = sorted(set([r['true_noise'] for r in psnr_results]))
    psf_types = sorted(set([r['psf_type'] for r in psnr_results]))
    
    # Создаем фигуру с тремя подграфиками (по одному на уровень шума)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('PSNR: Comparison of methods for different PSF', fontsize=16)
    
    colors = {'Richardson-Lucy': 'blue', 'Wiener (olimp)': 'green', 'Wiener (skimage)': 'red'}
    psf_colors = {'тип1': 'light', 'тип2': 'dark'}
    
    bar_width = 0.35
    psf_offset = {'тип1': -bar_width/2, 'тип2': bar_width/2}

    legend_elements = []

    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                       label='No improvement'))
    
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='lightblue', alpha=0.5,
                                          edgecolor='black', label='PSF type 1 (weak blur)'))
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='darkblue', alpha=0.5,
                                          edgecolor='black', label='PSF type 2 (strong blur)'))
    
    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        
        x_pos = np.arange(len(methods))
        
        for psf_type in psf_types:
            means = []
            errors = []
            valid_methods = []
            
            for i, method in enumerate(methods):
                impr = [r['improvement'] for r in psnr_results 
                        if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
                if impr:
                    means.append(np.mean(impr))
                    errors.append(np.std(impr))
                    valid_methods.append(i)
                else:
                    means.append(0)
                    errors.append(0)
                        
            if any(m != 0 for m in means):
                offset = psf_offset[psf_type]

                if psf_type == 'тип1':
                    bar_color = 'lightblue'
                else:
                    bar_color = 'blue'

                bars = ax.bar(x_pos + offset, means, bar_width,
                             yerr=errors, capsize=3,
                             color=bar_color,
                             alpha=0.5, 
                             edgecolor='black', linewidth=0.5)

                for i, method in enumerate(methods):
                    impr = [r['improvement'] for r in psnr_results 
                            if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
                    if impr:
                        x_vals = [i + offset] * len(impr)
                        point_color = 'blue' if method == 'Richardson-Lucy' else \
                                      'green' if method == 'Wiener (olimp)' else \
                                      'red'
                        ax.scatter(x_vals, impr, alpha=0.7, 
                                  color=point_color,
                                  s=30, zorder=10)
        
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='no improvement')
        ax.set_xlabel('Method')
        ax.set_ylabel('PSNR improvement (dB)')
        ax.set_title(f'Noise level {noise}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('Wiener (', '').replace(')', '') for m in methods])
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(handles=legend_elements, title='PSF', loc='upper left')
        
        for i, method in enumerate(methods):
            y_max = ax.get_ylim()[1]
            for psf_type in psf_types:
                impr = [r['improvement'] for r in psnr_results 
                        if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
    
    plt.tight_layout()
    plt.savefig('results/methods_comparison_by_psf.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("="*80)
    print(f"{'Метод':20} {'Тип ФРТ':8} {'Шум':8} {'Улучшение PSNR (dB)':25} {'n':5}")
    print("-"*80)
    
    for method in methods:
        for psf_type in psf_types:
            for noise in noise_levels:
                impr = [r['improvement'] for r in psnr_results 
                        if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
                if impr:
                    mean = np.mean(impr)
                    std = np.std(impr)
                    print(f"{method:20} {psf_type:8} {noise:8} {mean:6.2f} ± {std:5.2f} dB    n={len(impr):2}")


def plot_methods_comparison_by_psf_ssim(all_results):
    ssim_results = [r for r in all_results if r['type'] == 'ssim']
    
    methods = sorted(set([r['method'] for r in ssim_results]))
    noise_levels = sorted(set([r['true_noise'] for r in ssim_results]))
    psf_types = sorted(set([r['psf_type'] for r in ssim_results]))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SSIM: Comparison of methods for different PSF types', fontsize=16)
    
    bar_width = 0.35
    psf_offset = {'тип1': -bar_width/2, 'тип2': bar_width/2}

    legend_elements = []

    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                       label='No improvement'))
    
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='lightblue', alpha=0.5,
                                          edgecolor='black', label='PSF type 1 (weak blur)'))
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='darkblue', alpha=0.5,
                                          edgecolor='black', label='PSF type 2 (strong blur)'))
    
    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        
        x_pos = np.arange(len(methods))
        
        for psf_type in psf_types:
            means = []
            errors = []
            
            for i, method in enumerate(methods):
                impr = [r['improvement'] for r in ssim_results 
                        if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
                if impr:
                    means.append(np.mean(impr))
                    errors.append(np.std(impr))
                else:
                    means.append(0)
                    errors.append(0)
            
            if any(m != 0 for m in means):
                offset = psf_offset[psf_type]

                if psf_type == 'тип1':
                    bar_color = 'lightblue'
                else:
                    bar_color = 'darkblue'

                bars = ax.bar(x_pos + offset, means, bar_width,
                             yerr=errors, capsize=3,
                             color=bar_color,
                             alpha=0.5,
                             edgecolor='black', linewidth=0.5)

                for i, method in enumerate(methods):
                    impr = [r['improvement'] for r in ssim_results 
                            if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
                    if impr:
                        x_vals = [i + offset] * len(impr)
                        point_color = 'blue' if method == 'Richardson-Lucy' else \
                                      'green' if method == 'Wiener (olimp)' else \
                                      'red'
                        ax.scatter(x_vals, impr, alpha=0.7, 
                                  color=point_color,
                                  s=30, zorder=10)
        
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Method')
        ax.set_ylabel('SSIM improvement')
        ax.set_title(f'Noise level: {noise}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('Wiener (', '').replace(')', '') for m in methods])
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(handles=legend_elements, title='PSF type', loc='upper left')
        
        for i, method in enumerate(methods):
            y_max = ax.get_ylim()[1]
            for psf_type in psf_types:
                impr = [r['improvement'] for r in ssim_results 
                        if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
    
    plt.tight_layout()
    plt.savefig('results/methods_comparison_ssim_by_psf.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Статистика по SSIM
    print("\n📊 SSIM")
    print("="*80)
    print(f"{'Method':20} {'PSF type':8} {'Noise':8} {'SSIM improvement':25} {'n':5}")
    print("-"*80)
    
    for method in methods:
        for psf_type in psf_types:
            for noise in noise_levels:
                impr = [r['improvement'] for r in ssim_results 
                        if r['method'] == method and r['true_noise'] == noise and r['psf_type'] == psf_type]
                if impr:
                    mean = np.mean(impr)
                    std = np.std(impr)
                    print(f"{method:20} {psf_type:8} {noise:8} {mean:6.3f} ± {std:5.3f}    n={len(impr):2}")


rl_1 = parse_richardson_file_psftype('results/rich_lucy_res_1.txt', 'тип1')
rl_2 = parse_richardson_file_psftype('results/rich_lucy_res_2.txt', 'тип2')
wo_1 = parse_wiener_file_psftype('results/wiener_olimp_res_1.txt', 'Wiener (olimp)', 'тип1')
wo_2 = parse_wiener_file_psftype('results/wiener_olimp_res_2.txt', 'Wiener (olimp)', 'тип2')
ws_1 = parse_wiener_file_psftype('results/wiener_skimage_res_1.txt', 'Wiener (skimage)', 'тип1')
ws_2 = parse_wiener_file_psftype('results/wiener_skimage_res_2.txt', 'Wiener (skimage)', 'тип2')

all_results = rl_1 + rl_2 + wo_1 + wo_2 + ws_1 + ws_2

print(f" {len(all_results)}")
print(f"Richardson-Lucy: тип1={len(rl_1)}, тип2={len(rl_2)}")
print(f"Wiener (olimp): тип1={len(wo_1)}, тип2={len(wo_2)}")
print(f"Wiener (skimage): тип1={len(ws_1)}, тип2={len(ws_2)}")

plot_methods_comparison_by_psf(all_results)
plot_methods_comparison_by_psf_ssim(all_results)