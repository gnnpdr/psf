import re
import numpy as np
import matplotlib.pyplot as plt

def parse_richardson_results(filepath):
    results_psnr = []
    results_ssim = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    blocks = content.split('---------------')
    
    for block in blocks:
        if not block.strip():
            continue
        
        psnr_file_match = re.search(r'file (.+?)_best_psnr', block)
        if psnr_file_match:
            filename = psnr_file_match.group(1)
            
            noise_match = re.search(r'.+?_psf_(\d+)_noise_([\d.]+)', filename)
            if noise_match:
                psf_idx = int(noise_match.group(1))
                true_noise = float(noise_match.group(2))
            else:
                psf_idx = None
                true_noise = None
            
            psnr_match = re.search(r'initial psnr val ([\d.]+), psnr val after restoration ([\d.]+), iterations ([\d]+)', block)
            if psnr_match:
                results_psnr.append({
                    'filename': filename,
                    'true_noise': true_noise,
                    'psf_idx': psf_idx,
                    'init_psnr': float(psnr_match.group(1)),
                    'restored_psnr': float(psnr_match.group(2)),
                    'iterations': int(psnr_match.group(3)),
                    'psnr_improvement': float(psnr_match.group(2)) - float(psnr_match.group(1))
                })
        
        ssim_file_match = re.search(r'file (.+?)_best_ssim', block)
        if ssim_file_match:
            filename = ssim_file_match.group(1)
            
            noise_match = re.search(r'.+?_psf_(\d+)_noise_([\d.]+)', filename)
            if noise_match:
                psf_idx = int(noise_match.group(1))
                true_noise = float(noise_match.group(2))
            else:
                psf_idx = None
                true_noise = None
            
            ssim_match = re.search(r'initial ssim val ([\d.]+), ssim val after restoration ([\d.]+), iterations ([\d]+)', block)
            if ssim_match:
                results_ssim.append({
                    'filename': filename,
                    'true_noise': true_noise,
                    'psf_idx': psf_idx,
                    'init_ssim': float(ssim_match.group(1)),
                    'restored_ssim': float(ssim_match.group(2)),
                    'iterations': int(ssim_match.group(3)),
                    'ssim_improvement': float(ssim_match.group(2)) - float(ssim_match.group(1))
                })
    
    return results_psnr, results_ssim


def plot_richardson_results(results_psnr, results_ssim):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    if results_psnr:
        ax = axes[0, 0]
        
        true_noises = [r['true_noise'] for r in results_psnr if r['true_noise'] is not None]
        improvements = [r['psnr_improvement'] for r in results_psnr if r['true_noise'] is not None]
        
        unique_noises = sorted(set(true_noises))
        
        if unique_noises:
            means = []
            stds = []
            for noise in unique_noises:
                impr = [r['psnr_improvement'] for r in results_psnr if r['true_noise'] == noise]
                means.append(np.mean(impr))
                stds.append(np.std(impr))
            
            ax.errorbar(unique_noises, means, yerr=stds, marker='o', capsize=5)
            ax.scatter(true_noises, improvements, alpha=0.5, color='blue')
            
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='no improvements')
            ax.set_xlabel('init noise')
            ax.set_ylabel('PSNR improvement (dB)')
            ax.set_title('dependence of PSNR on the noise level')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    if results_ssim:
        ax = axes[0, 1]
        
        true_noises = [r['true_noise'] for r in results_ssim if r['true_noise'] is not None]
        improvements = [r['ssim_improvement'] for r in results_ssim if r['true_noise'] is not None]
        
        unique_noises = sorted(set(true_noises))
        
        if unique_noises:
            means = []
            stds = []
            for noise in unique_noises:
                impr = [r['ssim_improvement'] for r in results_ssim if r['true_noise'] == noise]
                means.append(np.mean(impr))
                stds.append(np.std(impr))
            
            ax.errorbar(unique_noises, means, yerr=stds, marker='s', color='green', capsize=5)
            ax.scatter(true_noises, improvements, alpha=0.5, color='green')
            
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='no improvements')
            ax.set_xlabel('init noise')
            ax.set_ylabel('SSIM improvement (dB)')
            ax.set_title('dependence of SSIM on the noise level')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    if results_psnr:
        ax = axes[1, 0]
        
        true_noises = [r['true_noise'] for r in results_psnr if r['true_noise'] is not None]
        iterations = [r['iterations'] for r in results_psnr if r['true_noise'] is not None]
        
        unique_noises = sorted(set(true_noises))
        
        if unique_noises:
            data_by_noise = []
            positions = []
            for i, noise in enumerate(unique_noises):
                iters = [r['iterations'] for r in results_psnr if r['true_noise'] == noise]
                data_by_noise.append(iters)
                positions.append(i)
            
            bp = ax.boxplot(data_by_noise, positions=positions, widths=0.5, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_alpha(0.5)
                patch.set_facecolor('lightblue')
            
            for i, noise in enumerate(unique_noises):
                iters = [r['iterations'] for r in results_psnr if r['true_noise'] == noise]
                x_vals = [i] * len(iters)
                ax.scatter(x_vals, iters, alpha=0.5, color='blue')
            
            ax.set_xlabel('init noise')
            ax.set_ylabel('iterations amount')
            ax.set_title('dependence of PSNR on the iterations amount')
            ax.set_xticks(positions)
            ax.set_xticklabels([f'{n}' for n in unique_noises])
            ax.grid(True, alpha=0.3, axis='y')
    
    if results_ssim:
        ax = axes[1, 1]
        
        true_noises = [r['true_noise'] for r in results_ssim if r['true_noise'] is not None]
        iterations = [r['iterations'] for r in results_ssim if r['true_noise'] is not None]
        
        unique_noises = sorted(set(true_noises))
        
        if unique_noises:
            data_by_noise = []
            positions = []
            for i, noise in enumerate(unique_noises):
                iters = [r['iterations'] for r in results_ssim if r['true_noise'] == noise]
                data_by_noise.append(iters)
                positions.append(i)
            
            bp = ax.boxplot(data_by_noise, positions=positions, widths=0.5, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_alpha(0.5)
                patch.set_facecolor('lightblue')
            
            for i, noise in enumerate(unique_noises):
                iters = [r['iterations'] for r in results_ssim if r['true_noise'] == noise]
                x_vals = [i] * len(iters)
                ax.scatter(x_vals, iters, alpha=0.5, color='green')
            
            ax.set_xlabel('init noise')
            ax.set_ylabel('iterations amount')
            ax.set_title('dependence of SSIM on the iterations amount')
            ax.set_xticks(positions)
            ax.set_xticklabels([f'{n}' for n in unique_noises])
            ax.grid(True, alpha=0.3, axis='y')
    
    if results_psnr:
        ax = axes[2, 0]
        
        psf_indices = [r['psf_idx'] for r in results_psnr if r['psf_idx'] is not None]
        improvements = [r['psnr_improvement'] for r in results_psnr if r['psf_idx'] is not None]
        
        unique_psf = sorted(set(psf_indices))
        
        if unique_psf:
            means = []
            stds = []
            for psf_idx in unique_psf:
                impr = [r['psnr_improvement'] for r in results_psnr if r['psf_idx'] == psf_idx]
                means.append(np.mean(impr))
                stds.append(np.std(impr))
            
            x_pos = np.arange(len(unique_psf))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            for i, psf_idx in enumerate(unique_psf):
                y_vals = [r['psnr_improvement'] for r in results_psnr if r['psf_idx'] == psf_idx]
                x_vals = [i] * len(y_vals)
                ax.scatter(x_vals, y_vals, alpha=0.5, color='blue', zorder=5)
            
            ax.set_xlabel('PSF index')
            ax.set_ylabel('PSNR improvement (dB)')
            ax.set_title('dependence of PSNR on the PSF')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'PSF {i}' for i in unique_psf])
            ax.grid(True, alpha=0.3, axis='y')
    
    if results_ssim:
        ax = axes[2, 1]
        
        psf_indices = [r['psf_idx'] for r in results_ssim if r['psf_idx'] is not None]
        improvements = [r['ssim_improvement'] for r in results_ssim if r['psf_idx'] is not None]
        
        unique_psf = sorted(set(psf_indices))
        
        if unique_psf:
            means = []
            stds = []
            for psf_idx in unique_psf:
                impr = [r['ssim_improvement'] for r in results_ssim if r['psf_idx'] == psf_idx]
                means.append(np.mean(impr))
                stds.append(np.std(impr))
            
            x_pos = np.arange(len(unique_psf))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.3, color='green')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            for i, psf_idx in enumerate(unique_psf):
                y_vals = [r['ssim_improvement'] for r in results_ssim if r['psf_idx'] == psf_idx]
                x_vals = [i] * len(y_vals)
                ax.scatter(x_vals, y_vals, alpha=0.5, color='green', zorder=5)
            
            ax.set_xlabel('PSF index')
            ax.set_ylabel('SSIM improvement (dB)')
            ax.set_title('dependence of SSIM on the PSF')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'PSF {i}' for i in unique_psf])
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/richardson_lucy_analysis.png', dpi=150, bbox_inches='tight')
    
    if results_psnr:
        unique_noises = sorted(set([r['true_noise'] for r in results_psnr if r['true_noise'] is not None]))
        for noise in unique_noises:
            group = [r for r in results_psnr if r['true_noise'] == noise]
            impr = [r['psnr_improvement'] for r in group]
            iters = [r['iterations'] for r in group]
            print(f"noise {noise}: improvement = {np.mean(impr):.2f}±{np.std(impr):.2f} dB, "
                  f"iterations = {np.mean(iters):.1f}±{np.std(iters):.1f}")
    
    if results_ssim:
        unique_noises = sorted(set([r['true_noise'] for r in results_ssim if r['true_noise'] is not None]))
        for noise in unique_noises:
            group = [r for r in results_ssim if r['true_noise'] == noise]
            impr = [r['ssim_improvement'] for r in group]
            iters = [r['iterations'] for r in group]
            print(f"noise {noise}: улучшение = {np.mean(impr):.4f}±{np.std(impr):.4f}, "
                  f"iterations = {np.mean(iters):.1f}±{np.std(iters):.1f}")
    
    if results_psnr:
        unique_psf = sorted(set([r['psf_idx'] for r in results_psnr if r['psf_idx'] is not None]))
        for psf_idx in unique_psf:
            group = [r for r in results_psnr if r['psf_idx'] == psf_idx]
            impr = [r['psnr_improvement'] for r in group]
            print(f"PSF {psf_idx}: improvement = {np.mean(impr):.2f}±{np.std(impr):.2f} dB, n={len(group)}")
    
    if results_ssim:
        unique_psf = sorted(set([r['psf_idx'] for r in results_ssim if r['psf_idx'] is not None]))
        for psf_idx in unique_psf:
            group = [r for r in results_ssim if r['psf_idx'] == psf_idx]
            impr = [r['ssim_improvement'] for r in group]
            print(f"PSF {psf_idx}: improvement = {np.mean(impr):.4f}±{np.std(impr):.4f}, n={len(group)}")


results_file = 'results/rich_lucy_res.txt'
results_psnr, results_ssim = parse_richardson_results(results_file)
plot_richardson_results(results_psnr, results_ssim)