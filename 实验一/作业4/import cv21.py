import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 控制变量（已修改） =====================
M     = 5          # 原4
sigma = 2.2        # 原1.8
# ====================================================

# 棋盘格（已修改）
def generate_checkerboard(size=256, block_size=10):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i//block_size + j//block_size) % 2 == 0:
                img[i,j] = 210   # 原255
    return img

# Chirp（已修改）
def generate_chirp(size=256):
    x = np.linspace(-1.1, 1.1, size)
    y = np.linspace(-1.1, 1.1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    img = np.sin(3 * np.pi * (4 * r + 35 * r**2))  # 频率修改
    img = (img - img.min()) / (img.max() - img.min()) * 225
    return img.astype(np.uint8)

# 下采样（不变）
def downsample(img, M):
    return img[::M, ::M]

# 高斯下采样（修改核大小）
def gaussian_downsample(img, M, sigma):
    blurred = cv2.GaussianBlur(img, (7, 7), sigma)  # 原(5,5)
    return downsample(blurred, M)

# FFT（不变）
def get_fft_spectrum(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude

# ===================== 统一绘图函数（配色+布局修改） =====================
def plot_one_img(img, title_prefix, save_path):
    img_direct = downsample(img, M)
    img_gauss  = gaussian_downsample(img, M, sigma)

    fft_ori    = get_fft_spectrum(img)
    fft_dir    = get_fft_spectrum(img_direct)
    fft_gau    = get_fft_spectrum(img_gauss)

    plt.figure(figsize=(17, 7))  # 尺寸修改

    # 图像部分（全新配色）
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'{title_prefix} Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_direct, cmap='gray')
    plt.title('Direct Downsample')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(img_gauss, cmap='gray')
    plt.title(f'Gaussian σ={sigma}')
    plt.axis('off')

    # FFT 部分（全新配色）
    plt.subplot(2, 3, 4)
    plt.imshow(fft_ori, cmap='viridis')
    plt.title('FFT Original')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(fft_dir, cmap='plasma')
    plt.title('FFT Direct')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(fft_gau, cmap='inferno')
    plt.title('FFT Gaussian')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)  # DPI提高
    plt.close()

# ===================== 分别生成 =====================
checker = generate_checkerboard()
chirp   = generate_chirp()

plot_one_img(checker, "Checkerboard", "result_checker_new.png")
plot_one_img(chirp,   "Chirp",       "result_chirp_new.png")

print("✅ 两张图已分别保存：")
print("  - result_checker_new.png")
print("  - result_chirp_new.png")