import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# ========== 解决Linux下tkinter缺失问题（强制保存对比图到本地） ==========
plt.switch_backend('Agg')  # 无GUI后端，仅保存图片不弹出窗口

# ---------------------- 1. 读入图像并转换为YCbCr ----------------------
img = cv2.imread("test.jpg")
if img is None:
    print("❌ 图片读取失败，请检查文件路径")
    exit()

h, w = img.shape[:2]
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)  # OpenCV中顺序是Y, Cr, Cb

# ---------------------- 2. 对Cb、Cr通道下采样（示例：2×2下采样） ----------------------
scale = 2  # 下采样倍数（可改为4等）
Cb_down = Cb[::scale, ::scale]  # 每隔scale取一个像素
Cr_down = Cr[::scale, ::scale]

# ---------------------- 3. 插值恢复到原尺寸（常用插值方法） ----------------------
# 可选插值方法：cv2.INTER_NEAREST（最近邻）、INTER_LINEAR（双线性）、INTER_CUBIC（双三次）
interp_method = cv2.INTER_LINEAR
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=interp_method)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=interp_method)

# ---------------------- 4. 重建YCbCr并转回RGB ----------------------
img_ycrcb_recon = cv2.merge((Y, Cr_up, Cb_up))  # 保持OpenCV通道顺序
img_rgb_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)

# ---------------------- 5. 计算PSNR（峰值信噪比） ----------------------
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100  # 完全相同
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(img, img_rgb_recon)
print(f"📊 下采样倍数: {scale}×{scale}")
print(f"🔍 插值方法: 双线性插值 (INTER_LINEAR)")
print(f"📈 PSNR 值: {psnr_value:.2f} dB")

# ---------------------- 6. 生成并保存「原图+重建图」对比图 ----------------------
# 转换为RGB格式（适配Matplotlib显示）
img_original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_recon_rgb = cv2.cvtColor(img_rgb_recon, cv2.COLOR_BGR2RGB)

# 创建对比画布（1行2列）
plt.figure(figsize=(12, 6), dpi=100)

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(img_original_rgb)
plt.title(f"Original Image", fontsize=12)
plt.axis("off")  # 隐藏坐标轴

# 显示重建图
plt.subplot(1, 2, 2)
plt.imshow(img_recon_rgb)
plt.title(f"Reconstructed Image (PSNR: {psnr_value:.2f} dB)", fontsize=12)
plt.axis("off")

# 调整布局并保存对比图
plt.tight_layout()
plt.savefig("comparison_image.jpg", bbox_inches='tight', pad_inches=0.1)
plt.close()

# ---------------------- 7. 保存重建图 + 提示信息 ----------------------
cv2.imwrite("reconstructed_image.jpg", img_rgb_recon)
print("\n✅ 结果文件已全部生成：")
print("  - comparison_image.jpg（原图+重建图对比图）")
print("  - reconstructed_image.jpg（单独重建图）")
print(f"💡 PSNR 越高，图像质量越接近原图（通常>30dB人眼难以区分）")