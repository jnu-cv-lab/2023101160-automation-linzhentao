import cv2
import numpy as np
import os
import time

# ========= 自动找图 + 转PNG =========
def find_and_convert_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    # 处理 box 图
    for name in ["box", "box.png", "box.jpg", "box.jpeg", "box.bmp"]:
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("box.png", img)
                break
    # 处理 box_in_scene 图
    for name in ["box_in_scene", "box_in_scene.png", "box_in_scene.jpg", "box_in_scene.jpeg", "box_in_scene.bmp"]:
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("box_in_scene.png", img)
                break

find_and_convert_images()

img1 = cv2.imread("box.png")
img2 = cv2.imread("box_in_scene.png")

# ========= ORB 实验（对比用） =========
orb = cv2.ORB_create(nfeatures=1000)
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)

start_time = time.time()
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
orb_time = time.time() - start_time

# ORB RANSAC
pts1_orb = np.float32([kp1_orb[m.queryIdx].pt for m in matches_orb]).reshape(-1,1,2)
pts2_orb = np.float32([kp2_orb[m.trainIdx].pt for m in matches_orb]).reshape(-1,1,2)
H_orb, mask_orb = cv2.findHomography(pts1_orb, pts2_orb, cv2.RANSAC, 5.0)
inliers_orb = sum(mask_orb.ravel())
inlier_ratio_orb = inliers_orb / len(matches_orb)
success_orb = "是" if inlier_ratio_orb > 0.2 else "否"

# ========= SIFT 实验 =========
# 1. 创建 SIFT 检测器
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2, None)

start_time = time.time()
# 2. 使用 NORM_L2 + KNN 匹配
bf_sift = cv2.BFMatcher(cv2.NORM_L2)
matches = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

# 3. Lowe's ratio test 筛选匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
sift_time = time.time() - start_time

# 4. RANSAC + Homography 目标定位
pts1_sift = np.float32([kp1_sift[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
pts2_sift = np.float32([kp2_sift[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
H_sift, mask_sift = cv2.findHomography(pts1_sift, pts2_sift, cv2.RANSAC, 5.0)
inliers_sift = sum(mask_sift.ravel())
inlier_ratio_sift = inliers_sift / len(good_matches)
success_sift = "是" if inlier_ratio_sift > 0.2 else "否"

# 保存 SIFT 定位结果
h, w = img1.shape[:2]
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1,1,2)
transformed = cv2.perspectiveTransform(corners, H_sift)
img_with_box = img2.copy()
cv2.polylines(img_with_box, [np.int32(transformed)], True, (0,0,255), 3)
cv2.imwrite("sift_target_loc.png", img_with_box)

# ========= 对比表格输出 =========
print("=== ORB vs SIFT 对比实验结果 ===\n")
print("| 方法 | 匹配数量 | RANSAC内点数 | 内点比例 | 是否成功定位 | 运行速度(秒) |")
print("|------|----------|--------------|----------|--------------|--------------|")
print(f"| ORB  | {len(matches_orb):>8} | {inliers_orb:>12} | {inlier_ratio_orb:.3f} | {success_orb:>12} | {orb_time:.3f} |")
print(f"| SIFT | {len(good_matches):>8} | {inliers_sift:>12} | {inlier_ratio_sift:.3f} | {success_sift:>12} | {sift_time:.3f} |")

print("\n✅ SIFT 目标定位结果已保存为：sift_target_loc.png")