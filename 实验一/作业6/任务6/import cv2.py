import cv2
import numpy as np
import os

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

# ========= 任务6：ORB参数对比实验 =========
nfeatures_list = [500, 1000, 2000]
results = []

print("=== ORB 参数对比实验结果 ===\n")

for nfeat in nfeatures_list:
    # 1. 初始化 ORB 检测器
    orb = cv2.ORB_create(nfeatures=nfeat)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2. BFMatcher 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. RANSAC 剔除外点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # 4. 计算指标
    template_kp_count = len(kp1)
    scene_kp_count = len(kp2)
    match_count = len(matches)
    inlier_count = sum(matchesMask)
    inlier_ratio = inlier_count / match_count if match_count > 0 else 0
    # 定位是否成功：内点比例大于 0.2 认为成功
    success = "是" if inlier_ratio > 0.2 else "否"

    # 5. 目标定位 + 保存结果图
    h, w = img1.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    img_with_box = img2.copy()
    cv2.polylines(img_with_box, [np.int32(transformed_corners)], True, (0, 0, 255), 3)
    cv2.imwrite(f"loc_result_nfeat_{nfeat}.png", img_with_box)

    results.append({
        "nfeatures": nfeat,
        "template_kp": template_kp_count,
        "scene_kp": scene_kp_count,
        "matches": match_count,
        "inliers": inlier_count,
        "ratio": inlier_ratio,
        "success": success
    })

# ========= 打印成表格（和你要求的格式一致） =========
print("| nfeatures | 模板图关键点 | 场景图关键点 | 匹配数量 | RANSAC内点 | 内点比例 | 是否成功定位 |")
print("|-----------|--------------|--------------|----------|------------|----------|--------------|")
for r in results:
    print(f"| {r['nfeatures']:>9} | {r['template_kp']:>12} | {r['scene_kp']:>12} | {r['matches']:>8} | {r['inliers']:>10} | {r['ratio']:.3f} | {r['success']:>12} |")

print("\n✅ 实验完成，已生成以下结果图：")
for nfeat in nfeatures_list:
    print(f"  loc_result_nfeat_{nfeat}.png")