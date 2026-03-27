import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -------------------------- 手动实现模块 --------------------------
def manual_hist_equalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    return cdf_normalized[img]

def manual_mean_filter(img, kernel_size=3):
    h, w = img.shape
    pad = kernel_size // 2
    img_pad = np.pad(img, pad, mode='edge')
    img_filtered = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            img_filtered[i, j] = np.mean(img_pad[i:i+kernel_size, j:j+kernel_size])
    return img_filtered.astype(np.uint8)

# -------------------------- 处理与可视化 --------------------------
def process_image(img, img_name, clip_limit=2.0, kernel_size=3):
    results = {}
    results["global_eq_manual"] = manual_hist_equalize(img)
    results["global_eq_cv2"] = cv2.equalizeHist(img)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    results["clahe"] = clahe.apply(img)

    results["mean_filter"] = manual_mean_filter(img, kernel_size)
    results["gaussian_filter"] = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    results["median_filter"] = cv2.medianBlur(img, kernel_size)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    results["sharpen"] = cv2.filter2D(img, -1, kernel)

    results["filter_then_eq"] = manual_hist_equalize(cv2.GaussianBlur(img, (kernel_size, kernel_size), 0))
    results["eq_then_filter"] = cv2.GaussianBlur(manual_hist_equalize(img), (kernel_size, kernel_size), 0)

    metrics = {}
    for key, res in results.items():
        psnr = peak_signal_noise_ratio(img, res)
        ssim = structural_similarity(img, res)
        metrics[key] = (psnr, ssim)

    plot_results(img, results, img_name)
    return results, metrics

def plot_results(original, results, img_name):
    plt.figure(figsize=(16, 12))
    plt.subplot(3, 4, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.subplot(3, 4, 2)
    plt.hist(original.flatten(), 256, [0,256])
    plt.title("Original Hist")

    idx = 3
    for name, res in results.items():
        plt.subplot(3, 4, idx)
        plt.imshow(res, cmap="gray")
        plt.title(name)
        plt.subplot(3, 4, idx+1)
        plt.hist(res.flatten(), 256, [0,256])
        plt.title(f"{name} Hist")
        idx += 2
        if idx > 12:
            break
    plt.tight_layout()
    plt.savefig(f"{img_name}_results.png")
    plt.show()

# -------------------------- 【这里换成你自己的3张图片】 --------------------------
def load_my_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(f"图片加载失败：{path}")
    return cv2.resize(img, (256, 256))

# 把这里改成你的图片文件名（和代码放同一个文件夹）
img_low_contrast = load_my_image("1.jpg")   # 低对比度图
img_noisy        = load_my_image("2.jpg")   # 噪声图
img_normal       = load_my_image("3.jpg")   # 普通图

# -------------------------- 执行处理 --------------------------
print("=== 处理低对比度图像 ===")
results_low, metrics_low = process_image(img_low_contrast, "low_contrast")

print("\n=== 处理噪声图像 ===")
results_noisy, metrics_noisy = process_image(img_noisy, "noisy")

print("\n=== 处理普通图像 ===")
results_normal, metrics_normal = process_image(img_normal, "normal")

def print_metrics(metrics, name):
    print(f"\n{name} 定量指标:")
    for k, v in metrics.items():
        print(f"{k:20s} | PSNR: {v[0]:.2f} | SSIM: {v[1]:.4f}")

print_metrics(metrics_low, "低对比度图像")
print_metrics(metrics_noisy, "噪声图像")
print_metrics(metrics_normal, "普通图像")