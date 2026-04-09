import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== 自动切换到代码所在目录 ======================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====================== 1. 图像读入与预处理（灰度图） ======================
img = cv2.imread("1.jpg", 0)
if img is None:
    raise Exception("请将图片命名为 1.jpg 并放在代码同一文件夹")

H, W = img.shape
print(f"原图尺寸: {W} × {H}")

# ====================== 工具函数 ======================
def downsample_quarter(img, use_gaussian=False):
    """ 下采样到 1/4 尺寸 """
    if use_gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
    new_w = W // 4
    new_h = H // 4
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def upsample_restore(img, method):
    """ 恢复到原始尺寸 """
    return cv2.resize(img, (W, H), interpolation=method)

def calc_mse_psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return 0, float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return mse, psnr

def fft_center_log(img):
    """ FFT + 中心化 + 对数幅度谱 """
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    return 20 * np.log(np.abs(fft_shift) + 1e-8)

def dct2(img):
    """ 二维DCT """
    return cv2.dct(np.float32(img))

def low_freq_energy_ratio(dct_mat, ratio=0.25):
    """ 左上角低频能量占比 """
    h, w = dct_mat.shape
    rh, rw = int(h*ratio), int(w*ratio)
    total = np.sum(dct_mat ** 2)
    low = np.sum(dct_mat[:rh, :rw] ** 2)
    return low / total if total != 0 else 0

# ====================== 2. 下采样（1/4） ======================
# ① 无预滤波直接缩小
img_direct_quarter = downsample_quarter(img, use_gaussian=False)
# ② 高斯滤波后再缩小
img_gaussian_quarter = downsample_quarter(img, use_gaussian=True)

# ====================== 3. 图像恢复（三种插值） ======================
methods = {
    "最近邻": cv2.INTER_NEAREST,
    "双线性": cv2.INTER_LINEAR,
    "双三次": cv2.INTER_CUBIC
}

restored = {}
for name, m in methods.items():
    restored[name] = upsample_restore(img_direct_quarter, m)

# ====================== 4. 空间域评价：MSE / PSNR ======================
print("===== 空间域质量评价 =====")
for name, res in restored.items():
    mse, psnr = calc_mse_psnr(img, res)
    print(f"{name:6s} | MSE = {mse:6.2f} | PSNR = {psnr:5.2f} dB")

# ====================== 5. 傅里叶变换分析 ======================
fft_ori = fft_center_log(img)
fft_small = fft_center_log(img_direct_quarter)
fft_small = cv2.resize(fft_small, (W, H))
fft_bilinear = fft_center_log(restored["双线性"])

# ====================== 6. DCT分析 ======================
dct_ori = dct2(img)
dct_nn = dct2(restored["最近邻"])
dct_bl = dct2(restored["双线性"])
dct_cu = dct2(restored["双三次"])

r_ori = low_freq_energy_ratio(dct_ori)
r_nn = low_freq_energy_ratio(dct_nn)
r_bl = low_freq_energy_ratio(dct_bl)
r_cu = low_freq_energy_ratio(dct_cu)

print("\n===== DCT 低频能量占比（左上角25%）=====")
print(f"原图      : {r_ori:.4f}")
print(f"最近邻    : {r_nn:.4f}")
print(f"双线性    : {r_bl:.4f}")
print(f"双三次    : {r_cu:.4f}")

# ====================== 绘图：完全匹配你的实验报告布局 ======================
plt.figure(figsize=(16, 14))

# -------------------- 第1行：原图、直接1/4缩小、高斯1/4缩小 --------------------
plt.subplot(4, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(4, 4, 2)
plt.imshow(img_direct_quarter, cmap='gray')
plt.title("Direct 1/4 Downsample")
plt.axis('off')

plt.subplot(4, 4, 3)
plt.imshow(img_gaussian_quarter, cmap='gray')
plt.title("Gaussian + 1/4 Downsample")
plt.axis('off')

# -------------------- 第1行：三种恢复图（匹配你报告） --------------------
names = list(restored.keys())
for i, name in enumerate(names):
    plt.subplot(4, 4, 4 + i)
    plt.imshow(restored[name], cmap='gray')
    plt.title(f"Restore {name}")
    plt.axis('off')

# -------------------- 第2行：FFT频谱（原图、缩小图、双线性恢复） --------------------
plt.subplot(4, 4, 5)
plt.imshow(fft_ori, cmap='gray')
plt.title("FFT Original")
plt.axis('off')

plt.subplot(4, 4, 6)
plt.imshow(fft_small, cmap='gray')
plt.title("FFT Small (1/4)")
plt.axis('off')

plt.subplot(4, 4, 7)
plt.imshow(fft_bilinear, cmap='gray')
plt.title("FFT Bilinear Restore")
plt.axis('off')

# -------------------- 第3行：DCT系数图 --------------------
plt.subplot(4, 4, 9)
plt.imshow(np.log(np.abs(dct_ori) + 1), cmap='gray')
plt.title("DCT Original")
plt.axis('off')

plt.subplot(4, 4, 10)
plt.imshow(np.log(np.abs(dct_cu) + 1), cmap='gray')
plt.title("DCT Cubic")
plt.axis('off')

plt.subplot(4, 4, 11)
plt.imshow(np.log(np.abs(dct_bl) + 1), cmap='gray')
plt.title("DCT Bilinear")
plt.axis('off')

plt.subplot(4, 4, 12)
plt.imshow(np.log(np.abs(dct_nn) + 1), cmap='gray')
plt.title("DCT Nearest")
plt.axis('off')

# -------------------- 第4行：指标展示（完全匹配你报告文字） --------------------
plt.subplot(4, 4, 13)
plt.axis('off')
text = f"""
MSE & PSNR（直接1/4下采样后恢复）
最近邻    MSE={calc_mse_psnr(img,restored['最近邻'])[0]:.2f}  PSNR={calc_mse_psnr(img,restored['最近邻'])[1]:.2f}dB
双线性    MSE={calc_mse_psnr(img,restored['双线性'])[0]:.2f}  PSNR={calc_mse_psnr(img,restored['双线性'])[1]:.2f}dB
双三次    MSE={calc_mse_psnr(img,restored['双三次'])[0]:.2f}  PSNR={calc_mse_psnr(img,restored['双三次'])[1]:.2f}dB

DCT低频能量占比（左上角25%）
原图      {r_ori:.4f}
双三次    {r_cu:.4f}
双线性    {r_bl:.4f}
最近邻    {r_nn:.4f}
"""
plt.text(0.05, 0.5, text, fontsize=11, va='center')
plt.title("Experiment Result")

# 多余位置关闭
for i in [8, 14, 15, 16]:
    plt.subplot(4, 4, i)
    plt.axis('off')

plt.tight_layout()
plt.savefig("experiment_result.png", dpi=300, bbox_inches='tight')
plt.show()