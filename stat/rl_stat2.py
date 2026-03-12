import re
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_rl_results(filepath):
    results = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('file '):
            data = {}
            
            filename = line[5:].strip() 
            data['filename'] = filename
            
            params_match = re.search(r'(.+?)_psf_(\d+)_noise_([\d.]+)', filename)
            if params_match:
                data['original_name'] = params_match.group(1)
                data['psf_idx'] = int(params_match.group(2))
                data['true_noise'] = float(params_match.group(3))
            
            i += 1
            if i < len(lines):
                psnr_line = lines[i].strip()
                psnr_match = re.search(r'initial psnr val ([\d.]+), psnr val after restoration ([\d.]+)', psnr_line)
                if psnr_match:
                    data['init_psnr'] = float(psnr_match.group(1))
                    data['restored_psnr'] = float(psnr_match.group(2))
                    data['psnr_improvement'] = data['restored_psnr'] - data['init_psnr']
            
            i += 1
            if i < len(lines):
                ssim_line = lines[i].strip()
                ssim_match = re.search(r'initial ssim val ([\d.]+), ssim val after restoration ([\d.]+)', ssim_line)
                if ssim_match:
                    data['init_ssim'] = float(ssim_match.group(1))
                    data['restored_ssim'] = float(ssim_match.group(2))
                    data['ssim_improvement'] = data['restored_ssim'] - data['init_ssim']
            
            results.append(data)
        
        i += 1
    
    return results
            
def plot_rl_analysis(results, res_name):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    
    noise_levels = sorted(set([r['true_noise'] for r in results if 'true_noise' in r]))
    
    means_psnr = []
    stds_psnr = []
    
    for noise in noise_levels:
        impr = [r['psnr_improvement'] for r in results 
                if r.get('true_noise') == noise and 'psnr_improvement' in r]
        if impr:
            means_psnr.append(np.mean(impr))
            stds_psnr.append(np.std(impr))
        else:
            means_psnr.append(0)
            stds_psnr.append(0)
    
    x_pos = np.arange(len(noise_levels))
    
    bars = ax.bar(x_pos, means_psnr, yerr=stds_psnr, capsize=5, alpha=0.3, color='blue', zorder=1)
    
    jitter = 0.05
    for noise in noise_levels:
        impr = [r['psnr_improvement'] for r in results 
                if r.get('true_noise') == noise and 'psnr_improvement' in r]
        if impr:
            x = [noise_levels.index(noise)] * len(impr)
            ax.scatter(x, impr, alpha=0.5, color='blue', zorder=5)
    
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='no improvement')
    ax.set_xlabel('noise')
    ax.set_ylabel('PSNR improvement (dB)')
    ax.set_title('dependence of PSNR on the noise level')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n}' for n in noise_levels])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[0, 1]
    
    means_ssim = []
    stds_ssim = []
    
    for noise in noise_levels:
        impr = [r['ssim_improvement'] for r in results 
                if r.get('true_noise') == noise and 'ssim_improvement' in r]
        if impr:
            means_ssim.append(np.mean(impr))
            stds_ssim.append(np.std(impr))
        else:
            means_ssim.append(0)
            stds_ssim.append(0)
    
    bars = ax.bar(x_pos, means_ssim, yerr=stds_ssim, capsize=5, alpha=0.3, color='green', zorder=1)
    
    for noise in noise_levels:
        impr = [r['ssim_improvement'] for r in results 
                if r.get('true_noise') == noise and 'ssim_improvement' in r]
        if impr:
            x = [noise_levels.index(noise)] * len(impr)
            ax.scatter(x, impr, alpha=0.5, color='blue', zorder=5)
    
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='no improvement')
    ax.set_xlabel('noise')
    ax.set_ylabel('PSNR improvement (dB)')
    ax.set_title('dependence of SSIM on the noise level')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n}' for n in noise_levels])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1, 0]
    
    psf_indices = sorted(set([r['psf_idx'] for r in results if 'psf_idx' in r]))
    
    means_psnr_psf = []
    stds_psnr_psf = []
    counts_psnr = []
    
    for psf_idx in psf_indices:
        impr = [r['psnr_improvement'] for r in results 
                if r.get('psf_idx') == psf_idx and 'psnr_improvement' in r]
        if impr:
            means_psnr_psf.append(np.mean(impr))
            stds_psnr_psf.append(np.std(impr))
            counts_psnr.append(len(impr))
        else:
            means_psnr_psf.append(0)
            stds_psnr_psf.append(0)
            counts_psnr.append(0)
    
    x_pos = np.arange(len(psf_indices))
    
    bars = ax.bar(x_pos, means_psnr_psf, yerr=stds_psnr_psf, capsize=5, alpha=0.3, color='blue', zorder=1)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    for psf_idx in psf_indices:
        impr = [r['psnr_improvement'] for r in results 
                if r.get('psf_idx') == psf_idx and 'psnr_improvement' in r]
        if impr:
            x = [psf_indices.index(psf_idx)] * len(impr)
            ax.scatter(x, impr, alpha=0.5, color='blue', zorder=5)

    
    
    ax.set_xlabel('PSF index')
    ax.set_ylabel('PSNR improvement (dB)')
    ax.set_title('dependence of PSNR on the PSF')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'PSF {i}' for i in psf_indices])
    ax.grid(True, alpha=0.3, axis='y')
    
    
    ax = axes[1, 1]
    
    means_ssim_psf = []
    stds_ssim_psf = []
    counts_ssim = []
    
    for psf_idx in psf_indices:
        impr = [r['ssim_improvement'] for r in results 
                if r.get('psf_idx') == psf_idx and 'ssim_improvement' in r]
        if impr:
            means_ssim_psf.append(np.mean(impr))
            stds_ssim_psf.append(np.std(impr))
            counts_ssim.append(len(impr))
        else:
            means_ssim_psf.append(0)
            stds_ssim_psf.append(0)
            counts_ssim.append(0)
    
    bars = ax.bar(x_pos, means_ssim_psf, yerr=stds_ssim_psf, capsize=5, alpha=0.3, color='green', zorder=1)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    for psf_idx in psf_indices:
        impr = [r['ssim_improvement'] for r in results 
                if r.get('psf_idx') == psf_idx and 'ssim_improvement' in r]
        if impr:
            x = [psf_indices.index(psf_idx)] * len(impr)
            ax.scatter(x, impr, alpha=0.5, color='blue', zorder=5)
    
    ax.set_xlabel('PSF index')
    ax.set_ylabel('SSIM improvement (dB)')
    ax.set_title('dependence of SSIM on the PSF')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'PSF {i}' for i in psf_indices])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'results/{res_name}.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    results_file = "results/rich_lucy_res_20.txt"
    results = parse_rl_results(results_file)
    res_name = 'rich_lucy_20'
    plot_rl_analysis(results, res_name)