import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像（必须叫 1.jpg）
img = cv2.imread("1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# ====================== ① 相似变换 ======================
center = (w//2, h//2)
M_similar = cv2.getRotationMatrix2D(center, angle=30, scale=0.8)
img_similar = cv2.warpAffine(img, M_similar, (w, h))

# ====================== ② 仿射变换 ======================
pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
pts2 = np.float32([[30,50], [w-50,80], [50,h-50]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (w, h))

# ====================== ③ 透视变换 ======================
pts1_p = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
pts2_p = np.float32([[80,60],[w-80,100],[80,h-60],[w-80,h-100]])
M_perspective = cv2.getPerspectiveTransform(pts1_p, pts2_p)
img_perspective = cv2.warpPerspective(img, M_perspective, (w, h))

# ====================== 显示结果 ======================
plt.figure(figsize=(16,6))

plt.subplot(141), plt.imshow(img), plt.title("Original"), plt.axis('off')
plt.subplot(142), plt.imshow(img_similar), plt.title("Similar Transform"), plt.axis('off')
plt.subplot(143), plt.imshow(img_affine), plt.title("Affine Transform"), plt.axis('off')
plt.subplot(144), plt.imshow(img_perspective), plt.title("Perspective Transform"), plt.axis('off')

plt.show()