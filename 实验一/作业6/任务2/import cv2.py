import cv2
import numpy as np
import os

# ========= 自动找图 + 自动转PNG（和任务1一样的鲁棒性） =========
def find_and_convert_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 找 box 图
    for name in ["box", "box.png", "box.jpg", "box.jpeg", "box.bmp"]:
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("box.png", img)
                break

    # 找 box_in_scene 图
    for name in ["box_in_scene", "box_in_scene.png", "box_in_scene.jpg", "box_in_scene.jpeg", "box_in_scene.bmp"]:
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("box_in_scene.png", img)
                break

# 先自动处理图片
find_and_convert_images()

# 1. 读取图片
img1 = cv2.imread("box.png")
img2 = cv2.imread("box_in_scene.png")

# 2. 创建 ORB 检测器
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 3. 创建暴力匹配器（按题目要求）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 4. 进行匹配
matches = bf.match(des1, des2)

# 5. 按匹配距离从小到大排序
matches = sorted(matches, key = lambda x:x.distance)

# 6. 输出总匹配数量
print("总匹配数量：", len(matches))

# 7. 可视化前50个匹配结果（也可以改成前30个）
draw_params = dict(matchColor = (0,255,0),  # 匹配点连线为绿色
                   singlePointColor = None,
                   flags = 2)
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, **draw_params)

# 保存结果（提交用）
cv2.imwrite("orb_initial_matches.png", img_matches)
print("✅ 已保存 ORB 匹配结果图：orb_initial_matches.png")