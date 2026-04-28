import cv2
import numpy as np
import os

# ========= 自动找图 + 转PNG（兼容任意格式） =========
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

# 执行图片预处理
find_and_convert_images()

# ========= 任务1：ORB关键点检测 =========
img1 = cv2.imread("box.png")
img2 = cv2.imread("box_in_scene.png")

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 保存关键点可视化
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
cv2.imwrite("box_kp.png", img1_kp)
cv2.imwrite("box_in_scene_kp.png", img2_kp)

print("=== 任务1 结果 ===")
print("box.png 关键点数量：", len(kp1))
print("box_in_scene.png 关键点数量：", len(kp2))
print("描述子维度：", des1.shape[1])

# ========= 任务2：ORB特征匹配 =========
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

print("\n=== 任务2 结果 ===")
print("总匹配数量：", len(matches))

# 保存初始匹配图
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imwrite("orb_initial_matches.png", img_matches)

# ========= 任务3：RANSAC剔除错误匹配 =========
# 提取匹配点坐标
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

# 计算单应矩阵 + RANSAC剔除
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# 筛选内点
inlier_matches = [matches[i] for i in range(len(matches)) if matchesMask[i]]
inlier_count = len(inlier_matches)
total_count = len(matches)
inlier_ratio = inlier_count / total_count

print("\n=== 任务3 结果 ===")
print("Homography矩阵：\n", H)
print("总匹配数量：", total_count)
print("RANSAC内点数量：", inlier_count)
print("内点比例：", inlier_ratio)

# 绘制RANSAC后的匹配图
draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
cv2.imwrite("orb_ransac_matches.png", img_ransac)

# ========= 任务4：目标定位 =========
# 1. 获取box.png的四个角点
h, w = img1.shape[:2]
pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 2. 使用cv2.perspectiveTransform()进行角点投影
pts_transformed = cv2.perspectiveTransform(pts, H)

# 3. 使用cv2.polylines()在场景图中画出四边形边框
img_scene_with_box = img2.copy()
cv2.polylines(img_scene_with_box, [np.int32(pts_transformed)], True, (0, 0, 255), 3, cv2.LINE_AA)

# 4. 保存定位结果图
cv2.imwrite("target_detection_result.png", img_scene_with_box)

print("\n=== 任务4 结果 ===")
print("目标定位已完成，边框已绘制并保存为 target_detection_result.png")
print("定位结果说明：基于估计的Homography矩阵，成功将box.png的四个角点投影到场景图中，并绘制了红色边框，目标定位成功。")

print("\n✅ 所有任务完成，结果已保存：")
print("box_kp.png / box_in_scene_kp.png / orb_initial_matches.png / orb_ransac_matches.png / target_detection_result.png")