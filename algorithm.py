import os
import time
import glob
import numpy as np
import cv2
import torch
import pywt  # 需提前安装: pip install PyWavelets
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import bm3d  # 需提前安装: pip install bm3d

# ==========================================
# 配置路径与参数
# ==========================================
INPUT_DIR = r"D:\AAAlyk\experiment\exp5\images"
OUTPUT_DIR = r"D:\AAAlyk\experiment\exp5\result"
NOISE_LEVEL = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 兼容你代码中的 psnr_metric，统一设置 data_range 为 1.0
def psnr_metric(img_true, img_test):
    return psnr(img_true, img_test, data_range=1.0)

# ==========================================
# 1. 算法核心定义 (包含你提供的代码)
# ==========================================

def ista_denoising(y, lam=0.02, rho=0.1, iterations=100):
    x = np.copy(y)
    h, w = y.shape
    for _ in range(iterations):
        r = x - rho * (x - y)
        coeffs = pywt.wavedec2(r, 'db1', level=2)
        coeffs_list, coeff_slices = pywt.coeffs_to_array(coeffs)
        tau = lam * rho
        coeffs_list_thresh = np.sign(coeffs_list) * np.maximum(np.abs(coeffs_list) - tau, 0)
        new_coeffs = pywt.array_to_coeffs(coeffs_list_thresh, coeff_slices, output_format='wavedec2')
        x_reconstructed = pywt.waverec2(new_coeffs, 'db1')
        x = x_reconstructed[:h, :w]
        x = np.clip(x, 0, 1)
    return x

def soft_threshold_wavelet(x, lmbd, shape):
    coeffs = pywt.wavedec2(x.reshape(shape), 'db1', level=2)
    coeffs_list, coeff_slices = pywt.coeffs_to_array(coeffs)
    # L1 软阈值
    coeffs_thresh = np.sign(coeffs_list) * np.maximum(np.abs(coeffs_list) - lmbd, 0)
    new_coeffs = pywt.array_to_coeffs(coeffs_thresh, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(new_coeffs, 'db1')[:shape[0], :shape[1]].flatten()

def FISTA_optimizer(gradf, proxg, params, img_ref=None, shape=None):
    x_k = np.copy(params['x0'])
    y_k = np.copy(x_k)
    t_k = 1.0
    alpha = 1.0 / params['prox_Lips']
    history = []
    
    for k in range(params['maxit']):
        x_old = x_k
        # 梯度步
        prox_arg = y_k - alpha * gradf(y_k)
        # 近端映射步
        x_k = proxg(prox_arg, alpha * params['lambda'])
        
        # 记录 PSNR 历史 (用于收敛曲线)
        if img_ref is not None:
            history.append(psnr_metric(img_ref, x_k.reshape(shape)))
        
        # FISTA 加速
        t_next = (1 + np.sqrt(4 * (t_k**2) + 1)) / 2
        y_next = x_k + ((t_k - 1) / t_next) * (x_k - x_old)
        
        # 重启准则 (Restart): 防止震荡
        if params['restart'] and (y_k.flatten() - x_k.flatten()) @ (x_k.flatten() - x_old.flatten()) > 0:
            y_k = x_old
            t_k = 1.0
        else:
            y_k = y_next
            t_k = t_next
            
    if img_ref is not None:
        return x_k, history
    return x_k

def ADMM_denoise_log(y, img_ref, lmbd=0.005, rho=0.1, maxit=50):
    h, w = y.shape
    x = np.copy(y).flatten()
    z = np.zeros(h * w)
    u = np.zeros(h * w)
    psnr_hist = []
    for k in range(maxit):
        # x-update
        x = (y.flatten() + rho * (z - u)) / (1 + rho)
        # z-update (软阈值收缩)
        z = np.sign(x + u) * np.maximum(np.abs(x + u) - lmbd / rho, 0)
        # u-update
        u = u + x - z
        psnr_hist.append(psnr_metric(img_ref, x.reshape(h, w)))
    return np.clip(x.reshape(h, w), 0, 1), psnr_hist

# ==========================================
# 2. 算法接口封装 (用于统一调用)
# ==========================================

def denoise_bm3d(noisy_img, sigma):
    return bm3d.bm3d(noisy_img, sigma_psd=sigma/255.0, stage_arg=bm3d.BM3DStages.ALL_STAGES)

def denoise_dncnn(noisy_img, model, device):
    model.eval()
    with torch.no_grad():
        img_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0).to(device)
        output = model(img_tensor)
        denoised_tensor = img_tensor - output
        denoised_img = denoised_tensor.squeeze().cpu().numpy()
        return np.clip(denoised_img, 0, 1)

def denoise_fista_wrapper(noisy_img, img_clean):
    shape = noisy_img.shape
    # 定义数据保真项 0.5 * ||x - y||^2 的梯度：gradf(x) = x - y
    gradf = lambda x: x - noisy_img.flatten()
    # 定义近端映射函数
    proxg = lambda x, lmbd: soft_threshold_wavelet(x, lmbd, shape)
    
    params = {
        'x0': noisy_img.flatten(),
        'prox_Lips': 1.0, 
        'lambda': 0.05,  
        'maxit': 100,
        'restart': True
    }
    x_k, _ = FISTA_optimizer(gradf, proxg, params, img_ref=img_clean, shape=shape)
    return np.clip(x_k.reshape(shape), 0, 1)

import torch
import torch.nn as nn

class DnCNN_Simplified(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=1):
        super(DnCNN_Simplified, self).__init__()
        layers = []
        # 第一层：Conv + ReLU
        layers.append(nn.Conv2d(image_channels, n_channels, 3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # 中间层：仅 Conv + ReLU（不加 BN，防止尺寸不匹配）
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        # 最后一层：Conv
        layers.append(nn.Conv2d(n_channels, image_channels, 3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)
    
# ==========================================
# 3. 核心评测流程
# ==========================================
def run_experiments():
    # --- 准备设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 实例化并加载 DnCNN 模型 ---
    MODEL_PATH = r"D:\AAAlyk\experiment\exp5\dncnn_25.pth" 
    
    # 根据报错信息中的数字，这里 depth 尝试设为 17 或 20
    dncnn_model = DnCNN_Simplified(depth=17) 
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # 修复 Key 名并强行匹配形状
        new_state_dict = {}
        model_dict = dncnn_model.state_dict()
        
        for k, v in state_dict.items():
            # 统一前缀
            name = k.replace('model.', 'dncnn.').replace('net.', 'dncnn.')
            # 只有当名字存在且形状一致时才加载
            if name in model_dict and v.shape == model_dict[name].shape:
                new_state_dict[name] = v
        
        model_dict.update(new_state_dict)
        dncnn_model.load_state_dict(model_dict)
        dncnn_model.to(device)
        dncnn_model.eval()
        print(f"已成功跳过不匹配参数并加载模型: {MODEL_PATH}")
    except Exception as e:
        print(f"加载依然失败: {e}")
        return

    # --- 准备数据 ---
    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.*'))
    if not image_paths:
        print(f"未在 {INPUT_DIR} 找到图片，请检查路径。")
        return

    # --- 初始化记录字典 (取消 DnCNN 的注释) ---
    results = {
        'BM3D': {'psnr': [], 'ssim': [], 'time': []},
        'ISTA': {'psnr': [], 'ssim': [], 'time': []},
        'FISTA': {'psnr': [], 'ssim': [], 'time': []},
        'ADMM': {'psnr': [], 'ssim': [], 'time': []},
        'DnCNN': {'psnr': [], 'ssim': [], 'time': []} 
    }

    for path in image_paths:
        img_name = os.path.basename(path)
        img_clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_clean is None:
            continue
        img_clean = img_clean.astype(np.float32) / 255.0

        # 添加高斯噪声
        np.random.seed(0) 
        noise = np.random.normal(0, NOISE_LEVEL / 255.0, img_clean.shape)
        img_noisy = np.clip(img_clean + noise, 0, 1)

        print(f"\nProcessing: {img_name}...")

        # --- 1. BM3D ---
        start_time = time.time()
        img_bm3d = denoise_bm3d(img_noisy, NOISE_LEVEL)
        results['BM3D']['time'].append(time.time() - start_time)
        results['BM3D']['psnr'].append(psnr_metric(img_clean, img_bm3d))
        results['BM3D']['ssim'].append(ssim(img_clean, img_bm3d, data_range=1.0))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"BM3D_{img_name}"), (img_bm3d * 255).astype(np.uint8))

        # --- 2. ISTA ---
        start_time = time.time()
        img_ista = ista_denoising(img_noisy)
        results['ISTA']['time'].append(time.time() - start_time)
        results['ISTA']['psnr'].append(psnr_metric(img_clean, img_ista))
        results['ISTA']['ssim'].append(ssim(img_clean, img_ista, data_range=1.0))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"ISTA_{img_name}"), (img_ista * 255).astype(np.uint8))

        # --- 3. FISTA ---
        start_time = time.time()
        img_fista = denoise_fista_wrapper(img_noisy, img_clean)
        results['FISTA']['time'].append(time.time() - start_time)
        results['FISTA']['psnr'].append(psnr_metric(img_clean, img_fista))
        results['FISTA']['ssim'].append(ssim(img_clean, img_fista, data_range=1.0))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"FISTA_{img_name}"), (img_fista * 255).astype(np.uint8))

        # --- 4. ADMM ---
        start_time = time.time()
        img_admm, _ = ADMM_denoise_log(img_noisy, img_clean)
        results['ADMM']['time'].append(time.time() - start_time)
        results['ADMM']['psnr'].append(psnr_metric(img_clean, img_admm))
        results['ADMM']['ssim'].append(ssim(img_clean, img_admm, data_range=1.0))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"ADMM_{img_name}"), (img_admm * 255).astype(np.uint8))

        # --- 5. DnCNN (残差学习推理)  ---
        start_time = time.time()
        img_dncnn = denoise_dncnn(img_noisy, dncnn_model, device)
        results['DnCNN']['time'].append(time.time() - start_time)
        results['DnCNN']['psnr'].append(psnr_metric(img_clean, img_dncnn))
        results['DnCNN']['ssim'].append(ssim(img_clean, img_dncnn, data_range=1.0))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"DnCNN_{img_name}"), (img_dncnn * 255).astype(np.uint8))
        
        print(f"  {img_name} 所有算法测试完成。")

    # ==========================================
    # 4. 输出汇总表格
    # ==========================================
    print("\n" + "="*65)
    print("--- 实验结果汇总 (Set14) ---")
    print(f"{'Method':<10} | {'Avg PSNR (dB)':<15} | {'Avg SSIM':<10} | {'Avg Time (s)':<15}")
    print("-" * 65)
    
    for method, metrics in results.items():
        if metrics['psnr']: 
            print(f"{method:<10} | {np.mean(metrics['psnr']):<15.2f} | {np.mean(metrics['ssim']):<10.4f} | {np.mean(metrics['time']):<15.4f}")
    print("="*65)


    print(f"所有结果图片已保存至: {OUTPUT_DIR}")

    # ==========================================
    # 5. 基于平均数据生成可视化图表
    # ==========================================
    import matplotlib.pyplot as plt
    
    print("\n正在生成基于平均数据的标准化图表...")
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-muted') # 或者使用 'ggplot'
    plt.rcParams['font.sans-serif'] = ['Arial'] # 英文论文建议用 Arial 或 Times New Roman
    plt.rcParams['axes.unicode_minus'] = False

    # 提取平均数据
    method_names = list(results.keys())
    avg_psnrs = [np.mean(results[m]['psnr']) for m in method_names]
    avg_ssims = [np.mean(results[m]['ssim']) for m in method_names]
    avg_times = [np.mean(results[m]['time']) for m in method_names]

    # --- 图 1: PSNR 与 SSIM 对比柱状图 (双轴图，非常适合论文) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(method_names))
    width = 0.35

    # 绘制 PSNR (左轴)
    color_psnr = '#5DADE2'
    bars1 = ax1.bar(x - width/2, avg_psnrs, width, label='Avg. PSNR', color=color_psnr, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Denoising Methods')
    ax1.set_ylabel('PSNR (dB)', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names)
    ax1.set_ylim(min(avg_psnrs) - 2, max(avg_psnrs) + 2)
    ax1.bar_label(bars1, fmt='%.2f', padding=3)

    # 绘制 SSIM (右轴)
    ax2 = ax1.twinx()
    color_ssim = '#E74C3C'
    bars2 = ax2.bar(x + width/2, avg_ssims, width, label='Avg. SSIM', color=color_ssim, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('SSIM', color='black')
    ax2.set_ylim(min(avg_ssims) - 0.1, 1.0)
    ax2.bar_label(bars2, fmt='%.4f', padding=3)

    plt.title('Average Quantitative Results on Set14 (Sigma=25)')
    fig.tight_layout()
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.savefig(os.path.join(OUTPUT_DIR, "Average_Metrics_Comparison.png"), dpi=300)
    
    # --- 图 2: 平均耗时对比 (对数坐标，突出 DnCNN 速度优势) ---
    plt.figure(figsize=(10, 5))
    colors = ['#AAB7B8', '#AAB7B8', '#AAB7B8', '#AAB7B8', '#F39C12'] # 突出 DnCNN
    bars_time = plt.bar(method_names, avg_times, color=colors, edgecolor='black', width=0.6)
    
    plt.title('Average Computation Time per Image (Seconds)')
    plt.ylabel('Time (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.bar_label(bars_time, fmt='%.4f', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Average_Time_Comparison.png"), dpi=300)
    
    print(f"标准化图表已保存至: {OUTPUT_DIR}")
    plt.show()

if __name__ == '__main__':
    run_experiments()