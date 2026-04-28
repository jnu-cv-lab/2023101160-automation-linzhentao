import cv2
import numpy as np
import os

# ========= 自动找图 + 自动转PNG（核心修复） =========
def find_and_convert_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    print("当前目录：", base_dir)
    print("文件列表：", os.listdir())

    # 支持的图片格式
    exts = ["", ".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    found = {}

    # 自动找 box
    for name in ["box", "box.png", "box.jpg", "box.jpeg", "box.bmp"]:
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("box.png", img)
                found["box"] = True
                break

    # 自动找 box_in_scene
    for name in ["box_in_scene", "box_in_scene.png", "box_in_scene.jpg", "box_in_scene.jpeg", "box_in_scene.bmp"]:
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("box_in_scene.png", img)
                found["scene"] = True
                break

    # 兜底检查
    if "box" not in found or "scene" not in found:
        print("❌ 没找到图片，请把两张图片放到代码同目录")
        exit()

# 先自动找图转格式
find_and_convert_images()

# 1. 读取两张图片
img1 = cv2.imread("box.png", cv2.IMREAD_COLOR)
img2 = cv2.imread("box_in_scene.png", cv2.IMREAD_COLOR)

# 2. 创建 ORB 检测器，设置 nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点和描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

# 5. 输出关键点数量和描述子维度
print("box.png 关键点数量：", len(kp1))
print("box_in_scene.png 关键点数量：", len(kp2))
print("描述子维度：", des1.shape[1] if des1 is not None else 0)

# 6. 保存结果图（提交用）
cv2.imwrite("box_kp.png", img1_kp)
cv2.imwrite("box_in_scene_kp.png", img2_kp)
print("✅ 已保存特征点可视化图：box_kp.png 和 box_in_scene_kp.png")