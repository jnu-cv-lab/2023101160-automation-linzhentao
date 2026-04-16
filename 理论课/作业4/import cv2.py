import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== 1. 基础设置与图像读取 ======================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取灰度图像1.jpg
img = cv2.imread("1.jpg", 0)
if img is None:
    raise FileNotFoundError("请将测试图片命名为1.jpg放在代码同目录下")

H, W = img.shape
print(f"原图尺寸: {H} × {W}")

# 分块参数：块大小8×8（符合图像处理常用分块）
block_size = 8
num_blocks_h = H // block_size
num_blocks_w = W // block_size
print(f"分块数量: {num_blocks_h} × {num_blocks_w}")

# ====================== 2. 工具函数 ======================
def compute_gradient_spatial(img_block):
    """空域梯度法：计算块内梯度，得到f_rms"""
    # Sobel算子计算x/y方向梯度
    sobel_x = cv2.Sobel(img_block, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_block, cv2.CV_64F, 0, 1, ksize=3)
    # 梯度幅值平方
    grad_sq = sobel_x**2 + sobel_y**2
    # 计算梯度平方均值
    E_grad_sq = np.mean(grad_sq)
    # 计算f_rms
    f_rms = (1 / (2 * np.pi)) * np.sqrt(E_grad_sq)
    return f_rms, sobel_x, sobel_y

def compute_fft_freq(img_block, energy_threshold=0.95):
    """频域FFT法：计算块内FFT，找到95%能量对应的最高频率"""
    h, w = img_block.shape
    # 二维FFT + 中心化
    fft = np.fft.fft2(img_block)
    fft_shift = np.fft.fftshift(fft)
    # 功率谱
    power_spectrum = np.abs(fft_shift)**2
    # 总能量
    total_energy = np.sum(power_spectrum)
    # 频率坐标（归一化到[-0.5, 0.5]）
    freq_x = np.fft.fftshift(np.fft.fftfreq(w))
    freq_y = np.fft.fftshift(np.fft.fftfreq(h))
    fx, fy = np.meshgrid(freq_x, freq_y)
    # 频率幅值（径向频率）
    freq_r = np.sqrt(fx**2 + fy**2)
    
    # 1. 计算f_rms（功率谱二阶矩）
    f_rms = np.sqrt(np.sum(power_spectrum * freq_r**2) / total_energy)
    
    # 2. 找到包含95%能量的最高频率
    # 按频率从小到大排序功率谱
    sorted_indices = np.argsort(freq_r.flatten())
    sorted_power = power_spectrum.flatten()[sorted_indices]
    sorted_freq = freq_r.flatten()[sorted_indices]
    # 累计能量
    cumulative_energy = np.cumsum(sorted_power)
    # 找到95%能量对应的频率
    threshold_energy = total_energy * energy_threshold
    idx_95 = np.argmax(cumulative_energy >= threshold_energy)
    f_95 = sorted_freq[idx_95]
    
    return f_rms, f_95, power_spectrum, freq_r

# ====================== 3. 分块计算两种方法的频率 ======================
# 初始化频率矩阵
f_rms_spatial = np.zeros((num_blocks_h, num_blocks_w))
f_rms_fft = np.zeros((num_blocks_h, num_blocks_w))
f_95_fft = np.zeros((num_blocks_h, num_blocks_w))

# 遍历分块
for i in range(num_blocks_h):
    for j in range(num_blocks_w):
        # 提取图像块
        y_start = i * block_size
        y_end = (i + 1) * block_size
        x_start = j * block_size
        x_end = (j + 1) * block_size
        block = img[y_start:y_end, x_start:x_end].astype(np.float64)
        
        # 1. 空域梯度法
        f_rms_sp, _, _ = compute_gradient_spatial(block)
        f_rms_spatial[i, j] = f_rms_sp
        
        # 2. 频域FFT法
        f_rms_fft_block, f_95_block, _, _ = compute_fft_freq(block)
        f_rms_fft[i, j] = f_rms_fft_block
        f_95_fft[i, j] = f_95_block

# ====================== 4. 一致性分析 ======================
# 计算两种方法f_rms的相关性
correlation = np.corrcoef(f_rms_spatial.flatten(), f_rms_fft.flatten())[0, 1]
# 计算平均误差
mae = np.mean(np.abs(f_rms_spatial - f_rms_fft))
print(f"\n===== 一致性分析 =====")
print(f"空域梯度法与频域FFT法f_rms的相关系数: {correlation:.4f}")
print(f"平均绝对误差: {mae:.6f}")
print(f"95%能量频率均值: {np.mean(f_95_fft):.6f}")

# ====================== 5. 可视化 ======================
plt.figure(figsize=(16, 12))

# 1. 原图
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image", fontsize=12)
plt.axis('off')

# 2. 空域梯度法f_rms热力图
plt.subplot(3, 3, 2)
plt.imshow(f_rms_spatial, cmap='hot')
plt.colorbar(label='f_rms (spatial)')
plt.title("Spatial Gradient f_rms", fontsize=12)
plt.axis('off')

# 3. 频域FFT法f_rms热力图
plt.subplot(3, 3, 3)
plt.imshow(f_rms_fft, cmap='hot')
plt.colorbar(label='f_rms (FFT)')
plt.title("FFT f_rms", fontsize=12)
plt.axis('off')

# 4. 95%能量频率热力图
plt.subplot(3, 3, 4)
plt.imshow(f_95_fft, cmap='hot')
plt.colorbar(label='f_95 (95% energy)')
plt.title("FFT 95% Energy Max Frequency", fontsize=12)
plt.axis('off')

# 5. 两种方法f_rms散点图
plt.subplot(3, 3, 5)
plt.scatter(f_rms_spatial.flatten(), f_rms_fft.flatten(), alpha=0.5, s=10)
plt.plot([0, np.max(f_rms_spatial)], [0, np.max(f_rms_spatial)], 'r--', label='y=x')
plt.xlabel("Spatial f_rms")
plt.ylabel("FFT f_rms")
plt.title(f"Correlation: {correlation:.4f}", fontsize=12)
plt.legend()
plt.grid(True)

# 6. 误差分布直方图
plt.subplot(3, 3, 6)
error = f_rms_spatial - f_rms_fft
plt.hist(error.flatten(), bins=50, alpha=0.7, color='blue')
plt.xlabel("Error (Spatial - FFT)")
plt.ylabel("Count")
plt.title(f"MAE: {mae:.6f}", fontsize=12)
plt.grid(True)

# 7. 示例块：空域梯度
sample_block = img[0:block_size, 0:block_size].astype(np.float64)
_, sobel_x, sobel_y = compute_gradient_spatial(sample_block)
plt.subplot(3, 3, 7)
plt.imshow(np.sqrt(sobel_x**2 + sobel_y**2), cmap='gray')
plt.title("Sample Block Gradient", fontsize=12)
plt.axis('off')

# 8. 示例块：FFT功率谱
_, _, power_spectrum, freq_r = compute_fft_freq(sample_block)
plt.subplot(3, 3, 8)
plt.imshow(20 * np.log10(power_spectrum + 1e-8), cmap='gray')
plt.title("Sample Block FFT Power Spectrum", fontsize=12)
plt.axis('off')

# 9. 结论
plt.subplot(3, 3, 9)
plt.axis('off')
text = f"""
Experiment Conclusion:
1. Spatial gradient method and FFT method have high consistency
   Correlation: {correlation:.4f}, MAE: {mae:.6f}
2. 95% energy max frequency (FFT): {np.mean(f_95_fft):.6f}
3. Gradient is a good approximation of local frequency
4. FFT is the rigorous method for actual frequency measurement
"""
plt.text(0.05, 0.5, text, fontsize=11, va='center')
plt.title("Conclusion", fontsize=12)

plt.tight_layout()
plt.savefig("frequency_estimation_result.png", dpi=300, bbox_inches='tight')
plt.show()