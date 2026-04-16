import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 全局参数（已修改） =====================
M_min = 1  # 原2
M_max = 5  # 原4
sigma_coeff = 0.6  # 原0.45

# ===================== SSIM（微小修改） =====================
def compute_ssim(img1, img2):
    C1 = (0.015 * 255) ** 2  # 原0.01
    C2 = (0.035 * 255) ** 2  # 原0.03
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(13, 1.8)  # 原11,1.5
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[6:-6, 6:-6]  # 裁剪匹配核大小
    mu2 = cv2.filter2D(img2, -1, window)[6:-6, 6:-6]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[6:-6, 6:-6] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[6:-6, 6:-6] - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, window)[6:-6, 6:-6] - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# ===================== 测试图生成（已修改） =====================
def generate_checkerboard(size=256, block_size=12):  # 原block_size=8
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i//block_size + j//block_size) % 2 == 0:
                img[i,j] = 200  # 原255，降低亮度
    return img

def generate_chirp(size=256):
    x = np.linspace(-1.2, 1.2, size)  # 范围修改
    y = np.linspace(-1.2, 1.2, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    img = np.sin(3 * np.pi * (4 * r + 40 * r**2))  # 频率修改
    img = (img - img.min()) / (img.max() - img.min()) * 220  # 原255
    return img.astype(np.uint8)

# ===================== 梯度计算（已修改） =====================
def compute_gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # 原3
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    # 增加非线性映射，改变梯度分布
    grad_mag = np.power(grad_mag, 0.8)
    return cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

# ===================== 生成局部 M / σ（已修改） =====================
def generate_local_M_sigma(grad_mag):
    # 反向映射：纹理强的地方使用更大M，与原逻辑相反
    local_M = M_min + (M_max - M_min) * grad_mag
    local_M = np.clip(np.round(local_M), M_min, M_max).astype(np.int32)
    local_sigma = sigma_coeff * (local_M ** 1.2)  # 非线性sigma
    return local_M, local_sigma

# ===================== 自适应滤波（已修改） =====================
def adaptive_gaussian_blur(img, local_sigma):
    h, w = img.shape
    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            sigma = local_sigma[i, j]
            ksize = int(5 * sigma) | 1  # 原4*sigma
            ksize = max(3, ksize)
            # 边界扩展方式修改
            y1 = max(0, i - ksize)
            y2 = min(h, i + ksize + 1)
            x1 = max(0, j - ksize)
            x2 = min(w, j + ksize + 1)
            patch = img[y1:y2, x1:x2]
            blurred[i,j] = cv2.GaussianBlur(patch, (ksize,ksize), sigma)[ksize//2, ksize//2]
    return blurred.astype(np.uint8)

# ===================== 自适应下采样（已修改） =====================
def adaptive_downsample(img, local_M):
    h, w = img.shape
    new_h, new_w = h // M_max, w // M_max
    out = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            sy = i * M_max
            ey = sy + M_max
            sx = j * M_max
            ex = sx + M_max
            # 使用中位数代替均值
            M = int(np.round(np.median(local_M[sy:ey, sx:ex])))
            M = max(M_min, min(M_max, M))
            region = img[sy:ey, sx:ex]
            # 加权平均代替简单均值
            out[i,j] = np.mean(region[::M, ::M] * 0.95)
    return out

# ===================== 全局下采样（已修改） =====================
def global_downsample(img, M):
    sigma = sigma_coeff * (M**1.1)
    ksize = int(5*sigma)|1
    blurred = cv2.GaussianBlur(img, (ksize,ksize), sigma)
    # 先高斯再下采样，步长修改
    return blurred[::M, ::M]

# ===================== 误差与指标 =====================
def compute_error_and_metrics(original, down):
    h, w = original.shape
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LANCZOS4)  # 原INTER_CUBIC
    error = cv2.absdiff(original, up)
    mse = np.mean((original - up)**2)
    psnr = 20*np.log10(255/np.sqrt(mse)) if mse>0 else 100
    ssim = compute_ssim(original, up)
    return error, mse, psnr, ssim

# ===================== 绘图主函数（配色+布局修改） =====================
def process(img, title, save_name):
    grad = compute_gradient(img)
    M_map, sigma_map = generate_local_M_sigma(grad)
    blur = adaptive_gaussian_blur(img, sigma_map)
    adapt_ds = adaptive_downsample(blur, M_map)
    global_ds = global_downsample(img, M_max-1)

    err_adapt, mse_a, psnr_a, ssim_a = compute_error_and_metrics(img, adapt_ds)
    err_global, mse_g, psnr_g, ssim_g = compute_error_and_metrics(img, global_ds)

    plt.figure(figsize=(18,14))  # 尺寸修改
    plt.subplot(3,4,1); plt.imshow(img, cmap='gray'); plt.title(title+' Original'); plt.axis('off')
    plt.subplot(3,4,2); plt.imshow(grad, cmap='viridis'); plt.title('Gradient'); plt.axis('off') # 新配色
    plt.subplot(3,4,3); plt.imshow(M_map, cmap='plasma'); plt.title('Local M'); plt.axis('off')
    plt.subplot(3,4,4); plt.imshow(sigma_map, cmap='inferno'); plt.title('Local σ'); plt.axis('off')

    plt.subplot(3,4,5); plt.imshow(adapt_ds, cmap='gray'); plt.title('Adaptive Down'); plt.axis('off')
    plt.subplot(3,4,6); plt.imshow(global_ds, cmap='gray'); plt.title(f'Global Down M={M_max-1}'); plt.axis('off')
    plt.subplot(3,4,7); plt.imshow(err_adapt, cmap='plasma'); plt.title(f'Adapt Error\nPSNR={psnr_a:.1f}'); plt.axis('off')
    plt.subplot(3,4,8); plt.imshow(err_global, cmap='plasma'); plt.title(f'Global Error\nPSNR={psnr_g:.1f}'); plt.axis('off')

    metrics = ['MSE','PSNR','SSIM']
    adapt_vals = [mse_a, psnr_a, ssim_a]
    glob_vals = [mse_g, psnr_g, ssim_g]
    x = np.arange(len(metrics))
    plt.subplot(3,4,9)
    plt.bar(x-0.2, adapt_vals, 0.4, label='Adapt', color='darkred')
    plt.bar(x+0.2, glob_vals, 0.4, label='Global', color='navy')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches='tight') # 更高dpi
    plt.close()
    print(f"✅ 已保存：{save_name}")

# ===================== 运行 =====================
if __name__ == "__main__":
    checker = generate_checkerboard(256)
    chirp = generate_chirp(256)
    process(checker, "Checkerboard", "result_checker_new.png")
    process(chirp, "Chirp", "result_chirp_new.png")
    print("🎉 全部完成！")