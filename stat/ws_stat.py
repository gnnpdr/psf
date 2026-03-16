import re
import numpy as np
import matplotlib.pyplot as plt

def parse_results_file(filepath):
    results_psnr = []
    results_ssim = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    blocks = content.split('---------------')
    
    for i, block in enumerate(blocks):
        if not block.strip():
            continue
        
        psnr_file_match = re.search(r'file (.+?)_best_psnr', block)
        if psnr_file_match:
            filename = psnr_file_match.group(1)
            
            noise_match = re.search(r'.+?_psf_\d+_noise_([\d.]+)', filename)
            true_noise = float(noise_match.group(1)) if noise_match else None
            
            psnr_match = re.search(r'initial psnr val ([\d.]+), psnr val after restoration ([\d.]+), noise ([\d.]+)', block)
            if psnr_match:
                results_psnr.append({
                    'filename': filename,
                    'true_noise': true_noise,
                    'init_psnr': float(psnr_match.group(1)),
                    'restored_psnr': float(psnr_match.group(2)),
                    'opt_noise': float(psnr_match.group(3)),
                    'psnr_improvement': float(psnr_match.group(2)) - float(psnr_match.group(1))
                })
        
        ssim_file_match = re.search(r'file (.+?)_best_ssim', block)
        if ssim_file_match:
            filename = ssim_file_match.group(1)
            noise_match = re.search(r'.+?_psf_\d+_noise_([\d.]+)', filename)
            true_noise = float(noise_match.group(1)) if noise_match else None
            ssim_match = re.search(r'initial ssim val ([\d.]+), ssim val after restoration ([\d.]+), noise ([\d.]+)', block)
            if ssim_match:
                results_ssim.append({
                    'filename': filename,
                    'true_noise': true_noise,
                    'init_ssim': float(ssim_match.group(1)),
                    'restored_ssim': float(ssim_match.group(2)),
                    'opt_noise': float(ssim_match.group(3)),
                    'ssim_improvement': float(ssim_match.group(2)) - float(ssim_match.group(1))
                })
    
    return results_psnr, results_ssim


def plot_results(results_psnr, results_ssim):
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    if results_psnr:
        ax = axes[0, 0]
        
        true_noises_psnr = [r['true_noise'] for r in results_psnr if r['true_noise'] is not None]
        psnr_improvements = [r['psnr_improvement'] for r in results_psnr if r['true_noise'] is not None]
        
        unique_noises = sorted(set(true_noises_psnr))
        
        if unique_noises:
            means = []
            stds = []
            for noise in unique_noises:
                improvements = [r['psnr_improvement'] for r in results_psnr 
                               if r['true_noise'] == noise]
                means.append(np.mean(improvements))
                stds.append(np.std(improvements))
            
            ax.errorbar(unique_noises, means, yerr=stds, marker='o', capsize=5)
            ax.scatter(true_noises_psnr, psnr_improvements, alpha=0.3, color='blue')
            
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='no improvement')
            ax.set_xlabel('init noise')
            ax.set_ylabel('PSNR improvement (dB)')
            ax.set_title('dependence of PSNR on the noise level')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    if results_ssim:
        ax = axes[0, 1]
        
        true_noises_ssim = [r['true_noise'] for r in results_ssim if r['true_noise'] is not None]
        ssim_improvements = [r['ssim_improvement'] for r in results_ssim if r['true_noise'] is not None]
        
        unique_noises = sorted(set(true_noises_ssim))
        
        if unique_noises:
            means = []
            stds = []
            for noise in unique_noises:
                improvements = [r['ssim_improvement'] for r in results_ssim 
                               if r['true_noise'] == noise]
                means.append(np.mean(improvements))
                stds.append(np.std(improvements))
            
            ax.errorbar(unique_noises, means, yerr=stds, marker='s', color='green', capsize=5)
            ax.scatter(true_noises_ssim, ssim_improvements, alpha=0.3, color='green')
            
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='no improvement')
            ax.set_xlabel('init noise')
            ax.set_ylabel('SSIM improvement (dB)')
            ax.set_title('dependence of SSIM on the noise level')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    if results_psnr:
        ax = axes[1, 0]
        
        true_noises_psnr = [r['true_noise'] for r in results_psnr if r['true_noise'] is not None]
        opt_noises_psnr = [r['opt_noise'] for r in results_psnr if r['true_noise'] is not None]
        
        if true_noises_psnr and opt_noises_psnr:
            ax.scatter(true_noises_psnr, opt_noises_psnr, alpha=0.6)
            
            min_noise = min(true_noises_psnr + opt_noises_psnr)
            max_noise = max(true_noises_psnr + opt_noises_psnr)
            ax.plot([min_noise, max_noise], [min_noise, max_noise], 
                    'r--', label='balance = init noise', alpha=0.5)
            
            ax.set_xlabel('init noise')
            ax.set_ylabel('balance (PSNR)')
            ax.set_title('balance and init noise correlation (PSNR)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    if results_ssim:
        ax = axes[1, 1]
        
        true_noises_ssim = [r['true_noise'] for r in results_ssim if r['true_noise'] is not None]
        opt_noises_ssim = [r['opt_noise'] for r in results_ssim if r['true_noise'] is not None]
        
        if true_noises_ssim and opt_noises_ssim:
            ax.scatter(true_noises_ssim, opt_noises_ssim, alpha=0.6, color='green')
            
            min_noise = min(true_noises_ssim + opt_noises_ssim)
            max_noise = max(true_noises_ssim + opt_noises_ssim)
            ax.plot([min_noise, max_noise], [min_noise, max_noise], 
                    'r--', label='balance = init noise', alpha=0.5)
            
            ax.set_xlabel('init noise')
            ax.set_ylabel('balance (SSIM)')
            ax.set_title('balance and init noise correlation (SSIM)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/wiener_skimage_analysis.png', dpi=150, bbox_inches='tight')
                

def plot_results_by_noise_psf(results_psnr, results_ssim):
    noise_levels = sorted(set([r['true_noise'] for r in results_psnr if r['true_noise'] is not None]))
    
    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(14, 5*len(noise_levels)))
    fig.suptitle('dependence of metrics on the PSF', fontsize=16)
    
    if len(noise_levels) == 1:
        axes = axes.reshape(1, -1)
    
    for row, noise in enumerate(noise_levels):
        psnr_noise = [r for r in results_psnr if r.get('true_noise') == noise]
        ssim_noise = [r for r in results_ssim if r.get('true_noise') == noise]
        
        ax = axes[row, 0]
        if psnr_noise:
            psf_indices = []
            improvements = []
            
            for r in psnr_noise:
                psf_match = re.search(r'_psf_(\d+)_', r['filename'])
                if psf_match:
                    psf_indices.append(int(psf_match.group(1)))
                    improvements.append(r['psnr_improvement'])
            
            if psf_indices:
                unique_psf = sorted(set(psf_indices))
                means = []
                stds = []
                for psf_idx in unique_psf:
                    impr = [improvements[i] for i, idx in enumerate(psf_indices) if idx == psf_idx]
                    means.append(np.mean(impr))
                    stds.append(np.std(impr))
                
                x_pos = np.arange(len(unique_psf))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.3, color='blue', edgecolor='black')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                for i, psf_idx in enumerate(unique_psf):
                    y_vals = [improvements[j] for j, idx in enumerate(psf_indices) if idx == psf_idx]
                    x_vals = [i] * len(y_vals)
                    ax.scatter(x_vals, y_vals, alpha=0.5, color='darkblue', s=30, zorder=10)
                
                ax.set_title(f'PSNR, noise {noise}')
                ax.set_xlabel('PSF index')
                ax.set_ylabel('PSNR improvement (dB)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'PSF {i}' for i in unique_psf])
                ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[row, 1]
        if ssim_noise:
            psf_indices = []
            improvements = []
            
            for r in ssim_noise:
                psf_match = re.search(r'_psf_(\d+)_', r['filename'])
                if psf_match:
                    psf_indices.append(int(psf_match.group(1)))
                    improvements.append(r['ssim_improvement'])
            
            if psf_indices:
                unique_psf = sorted(set(psf_indices))
                means = []
                stds = []
                for psf_idx in unique_psf:
                    impr = [improvements[i] for i, idx in enumerate(psf_indices) if idx == psf_idx]
                    means.append(np.mean(impr))
                    stds.append(np.std(impr))
                
                x_pos = np.arange(len(unique_psf))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.3, color='green', edgecolor='black')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                for i, psf_idx in enumerate(unique_psf):
                    y_vals = [improvements[j] for j, idx in enumerate(psf_indices) if idx == psf_idx]
                    x_vals = [i] * len(y_vals)
                    ax.scatter(x_vals, y_vals, alpha=0.5, color='darkgreen', s=30, zorder=10)
                
                ax.set_title(f'SSIM, noise {noise}')
                ax.set_xlabel('PSF index')
                ax.set_ylabel('SSIM improvement (dB)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'PSF {i}' for i in unique_psf])
                ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/wiener_skimage_analysis_by_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
              

results_file = 'results/wiener_skimage_res.txt'
results_psnr, results_ssim = parse_results_file(results_file)
plot_results(results_psnr, results_ssim)
plot_results_by_noise_psf(results_psnr, results_ssim)  