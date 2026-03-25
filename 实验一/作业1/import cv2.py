import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 任务1: 读取测试图片
# --------------------------
img = cv2.imread("test.jpg")  # 替换为你的图片路径
if img is None:
    raise ValueError("图片读取失败，请检查路径是否正确！")

# --------------------------
# 任务2: 输出图像基本信息
# --------------------------
height, width, channels = img.shape
dtype = img.dtype
print(f"图像尺寸：{width} × {height}")
print(f"通道数：{channels}")
print(f"像素数据类型：{dtype}")

# --------------------------
# 任务3: 显示原图（Matplotlib 版本）
# --------------------------
# OpenCV 读取为 BGR 格式，Matplotlib 显示需转为 RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# --------------------------
# 任务4: 转换为灰度图并显示
# --------------------------
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# --------------------------
# 任务5: 保存灰度图
# --------------------------
cv2.imwrite("gray_test.jpg", gray_img)
print("灰度图已保存为 gray_test.jpg")

# --------------------------
# 任务6: NumPy 简单操作（裁剪左上角区域 + 输出像素值）
# --------------------------
# 输出指定像素值（示例：坐标(100, 100)的BGR值）
pixel_bgr = img[100, 100]
print(f"坐标(100,100)的BGR像素值：{pixel_bgr}")

# 裁剪左上角 100×100 区域
crop_img = img[0:100, 0:100]
cv2.imwrite("crop_test.jpg", crop_img)
print("左上角裁剪区域已保存为 crop_test.jpg")

# 显示裁剪图
crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(3, 3))
plt.imshow(crop_rgb)
plt.title("Cropped Image (Top-left 100×100)")
plt.axis("off")
plt.show()