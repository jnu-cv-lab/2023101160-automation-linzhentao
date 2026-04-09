import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 自动切换到代码所在目录，避免路径问题
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取灰度图像：兼容常见后缀，优先读取1.jpg，自动适配
img = None
for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
    path = f"1{ext}"
    if os.path.exists(path):
        img = cv2.imread(path, 0)
        break
if img is None:
    raise FileNotFoundError("请将测试图片命名为 1.jpg（或1.png等），放在代码同一目录下")

H, W = img.shape
print(f"图像尺寸：{H} × {W}")

# 取图像第一行作为一维测试信号（符合作业要求：灰度图一行像素）
x = np.float32(img[0])
N = len(x)

# ====================== 1. 延拓方式对比（作业要求） ======================
# DFT隐含：周期延拓
x_periodic = np.concatenate([x, x])
# DCT隐含：偶对称延拓
x_symmetric = np.concatenate([x, x[::-1]])

# ====================== 2. 一维DFT与DCT-II计算 ======================
# DFT
X_dft = np.fft.fft(x)
# DCT-II（OpenCV实现，对应标准DCT-II）
X_dct = cv2.dct(x.reshape(-1, 1)).flatten()

# ====================== 3. 能量集中性计算（作业要求） ======================
def energy_ratio(coeffs, k_ratio=0.25):
    """计算前k_ratio比例系数的能量占比"""
    total = np.sum(np.abs(coeffs) ** 2)
    k = int(len(coeffs) * k_ratio)
    low_energy = np.sum(np.abs(coeffs[:k]) ** 2)
    return low_energy / total if total != 0 else 0

# 计算前25%低频能量占比（作业要求指标）
er_dft = energy_ratio(X_dft)
er_dct = energy_ratio(X_dct)
print("\n===== 低频能量占比（前25%系数）=====")
print(f"DFT: {er_dft:.4f}")
print(f"DCT: {er_dct:.4f}")

# ====================== 4. 可视化（完全匹配作业要求） ======================
plt.figure(figsize=(16, 12))

# 1. 原始图像
plt.subplot(3, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# 2. 原始行信号
plt.subplot(3, 4, 2)
plt.plot(x, color='blue')
plt.title("Original Signal (1st Row)")
plt.grid(True)

# 3. DFT周期延拓
plt.subplot(3, 4, 3)
plt.plot(x_periodic, color='red')
plt.title("DFT: Periodic Extension")
plt.axvline(x=N, color='k', linestyle='--', label="Boundary")
plt.legend()
plt.grid(True)

# 4. DCT偶对称延拓
plt.subplot(3, 4, 4)
plt.plot(x_symmetric, color='green')
plt.title("DCT: Even Symmetric Extension")
plt.axvline(x=N, color='k', linestyle='--', label="Boundary")
plt.legend()
plt.grid(True)

# 5. DFT系数幅度
plt.subplot(3, 4, 5)
plt.plot(np.abs(X_dft), color='blue')
plt.title("DFT Coefficients (Magnitude)")
plt.grid(True)

# 6. DCT系数（实数）
plt.subplot(3, 4, 6)
plt.plot(X_dct, color='green')
plt.title("DCT Coefficients (Real)")
plt.grid(True)

# 7. 能量占比对比
plt.subplot(3, 4, 7)
plt.bar(['DFT', 'DCT'], [er_dft, er_dct], color=['blue', 'green'], alpha=0.7)
plt.ylabel("Low-Frequency Energy Ratio (25%)")
plt.title("Energy Concentration Comparison")
plt.grid(True)

# 8. 延拓边界细节对比
plt.subplot(3, 4, 8)
plt.plot(np.arange(N-2, N+2), x_periodic[N-2:N+2], 'ro-', label='Periodic')
plt.plot(np.arange(N-2, N+2), x_symmetric[N-2:N+2], 'go-', label='Even Symmetric')
plt.axvline(x=N, color='k', linestyle='--', label="Original Boundary")
plt.title("Boundary Extension Detail")
plt.legend()
plt.grid(True)

# 9. 实验结论
plt.subplot(3, 4, 9)
plt.axis('off')
text = f"""
Experiment Results:
1. DFT energy ratio (25%): {er_dft:.4f}
2. DCT energy ratio (25%): {er_dct:.4f}

Conclusion:
1. DCT energy is more concentrated in low frequencies
2. Even symmetric extension eliminates boundary discontinuity
3. DCT is more suitable for image compression
"""
plt.text(0.05, 0.5, text, fontsize=11, va='center')
plt.title("Experiment Conclusion")

# 隐藏多余子图
for i in [10, 11, 12]:
    plt.subplot(3, 4, i)
    plt.axis('off')

plt.tight_layout()
plt.savefig("dft_dct_result.png", dpi=300, bbox_inches='tight')
plt.show()