
---

```markdown
# Image Denoising Benchmark: Classical Optimization vs. Deep Residual Learning

本项目是一个关于图像去噪算法的综合评测实验（Experiment 5）。通过对比传统的基于优化迭代的方法（ISTA, FISTA, ADMM）、经典滤波算法（BM3D）以及深度残差学习模型（DnCNN），分析各算法在恢复质量（PSNR/SSIM）与计算效率（Time）之间的权衡。

## 🚀 算法概览

本项目实现了以下五种主流去噪技术：
- **BM3D (Block-matching and 3D filtering)**: 传统去噪领域的 Benchmark 算法。
- **ISTA (Iterative Soft-Thresholding Algorithm)**: 基于小波域 L1 正则化的迭代阈值算法。
- **FISTA (Fast ISTA)**: 带有 Nesterov 加速项及重启策略（Restart Strategy）的快速迭代收敛算法。
- **ADMM (Alternating Direction Method of Multipliers)**: 经典的交替方向乘子法，用于求解全变分（TV）正则化模型。
- **DnCNN (Denoising CNN)**: 基于 PyTorch 实现的深度卷积神经网络，利用残差学习（Residual Learning）直接预测噪声。

## 📊 实验结果 (Dataset: Set14, $\sigma=25$)

在 RTX 4090 服务器上对标准数据集 Set14 进行测试，平均性能指标如下：

| Method | Avg. PSNR (dB) | Avg. SSIM | Avg. Time (s) |
| :--- | :---: | :---: | :---: |
| BM3D | 29.13 | 0.8211 | 0.5422 |
| ISTA | 23.11 | 0.6210 | 0.4011 |
| FISTA | 23.40 | 0.6402 | 0.4205 |
| ADMM | 20.37 | 0.4560 | 0.3120 |
| **DnCNN** | **29.39** | **0.8302** | **0.2779** |

### 关键结论
1. **性能巅峰**: DnCNN 在 PSNR 和 SSIM 上均取得最高分，相比 ADMM 有近 9dB 的提升，相比 BM3D 提升了 0.26dB。
2. **速度优势**: 得益于 GPU 并行计算与单次前向传播机制，DnCNN 的推理速度最快（0.27s），约为 BM3D 的 2 倍。
3. **视觉质量**: DnCNN 有效抑制了 TV 正则化常见的“阶梯效应（Staircase Effect）”，恢复的边缘更为自然。

## 🛠️ 环境配置

确保你的环境中已安装以下依赖库：

```bash
pip install numpy opencv-python torch torchvision matplotlib pywavelets scikit-image bm3d
```

## 📂 项目结构

```text
.
├── main.py                # 核心评测脚本（包含算法实现与绘图逻辑）
├── dncnn_25.pth           # 预训练 DnCNN 模型权重 (Sigma=25)
├── images/                # 测试数据集文件夹 (Set14)
└── result/                # 实验输出（去噪图片与对比图表）
    ├── Average_Metrics_Comparison.png
    └── Average_Time_Comparison.png
```

## 运行指南

1. 将测试图片放入 `images` 文件夹。
2. 确保 `dncnn_25.pth` 路径正确。
3. 运行主程序：
   ```bash
   python main.py
   ```
4. 运行结束后，可在 `result` 文件夹查看所有算法生成的图片及自动生成的分析图表。

## 📈 自动生成图表展示

项目运行后会自动生成标准化图表用于学术报告：
- **Metrics Comparison**: 双轴柱状图对比 PSNR 与 SSIM。
- **Time Comparison**: 对比各算法处理单张图片的平均耗时。

---
**Author:** Youke Lai  
**Institution:** Southwest Petroleum University (SWPU), Dept. of Computer Science  
**Date:** March 2026
```

---
