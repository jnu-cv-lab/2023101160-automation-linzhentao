import cv2
import numpy as np
import os

# ========= 关键修复：自动识别 test_fin 任意格式，转成标准 test_fin.jpg =========
def auto_convert_fin():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    exts = ["", ".jpg", ".png", ".jpeg", ".bmp", ".webp"]
    for e in exts:
        name = "test_fin" + e
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                cv2.imwrite("test_fin.jpg", img)
                return
auto_convert_fin()

# ===================== 路径配置 =====================
test_first_path = "test_first.jpg"
test_final_path = "test_fin.jpg"

# ===================== 1. 生成【鲜艳彩色】测试图：矩形、圆、平行线、垂直线 =====================
def make_test_img(size=600):
    # 白底
    img = np.ones((size, size, 3), np.uint8) * 255

    # 黑色外大框
    cv2.rectangle(img, (50, 50), (550, 550), (0, 0, 0), 3)

    # 红色圆形
    cv2.circle(img, (220, 300), 130, (0, 0, 255), 3)

    # 蓝色正方形
    cv2.rectangle(img, (380, 170), (480, 430), (255, 0, 0), 3)

    # 绿色水平平行线
    for y in [120, 220, 320, 420]:
        cv2.line(img, (80, y), (520, y), (0, 180, 0), 2)
    # 紫色垂直垂直线
    for x in [120, 220, 320, 420]:
        cv2.line(img, (x, 80), (x, 520), (150, 0, 180), 2)

    return img

# 生成作业要求的标准测试图
ori = make_test_img()
cv2.imwrite(test_first_path, ori)
print("✅ test_first.jpg 彩色测试图已保存")

h, w = ori.shape[:2]

# ===================== 2. 三种变换【任务一完整要求】 =====================
# ① 相似变换
M_sim = cv2.getRotationMatrix2D((w/2, h/2), 15, 0.85)
img_sim = cv2.warpAffine(ori, M_sim, (w, h), borderValue=(255,255,255))

# ② 仿射变换
pts1 = np.float32([[50, 50], [550, 50], [50, 550]])
pts2 = np.float32([[90, 110], [510, 70], [70, 490]])
M_aff = cv2.getAffineTransform(pts1, pts2)
img_aff = cv2.warpAffine(ori, M_aff, (w, h), borderValue=(255,255,255))

# ③ 透视变换
pts1_p = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2_p = np.float32([[50, 40], [w-40, 30], [30, h-40], [w-30, h-50]])
M_per = cv2.getPerspectiveTransform(pts1_p, pts2_p)
img_per = cv2.warpPerspective(ori, M_per, (w, h), borderValue=(255,255,255))

# 四宫格对比图
row1 = np.hstack((ori, img_sim))
row2 = np.hstack((img_aff, img_per))
img_all = np.vstack((row1, row2))

# 保存所有变换结果
cv2.imwrite("transform_相似变换.jpg", img_sim)
cv2.imwrite("transform_仿射变换.jpg", img_aff)
cv2.imwrite("transform_透视变换.jpg", img_per)
cv2.imwrite("transform_四宫格对比.jpg", img_all)
print("✅ 三种几何变换图片已全部保存")

# ===================== 3. A4 透视畸变校正【读取你自己的 test_fin】 =====================
img = cv2.imread(test_final_path)
if img is None:
    print("❌ 读取 test_fin 失败，请检查文件")
    exit()

img_show = img.copy()
H, W = img.shape[:2]

# 预处理（这里修复了 BGR2GRAY 的写法）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
edge = cv2.Canny(blur, 30, 120)
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

# 找纸张轮廓
cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
paper_box = None

for c in cnts:
    arc = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * arc, True)
    if len(approx) == 4 and cv2.contourArea(approx) > W*H*0.2:
        paper_box = approx.reshape(4, 2).astype(np.float32)
        break

# 四点排序
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

if paper_box is not None:
    paper_box = order_points(paper_box)
    cv2.polylines(img_show, [paper_box.astype(np.int32)], True, (0,0,255), 4)
else:
    paper_box = np.float32([[W*0.05,H*0.05],[W*0.95,H*0.05],[W*0.95,H*0.95],[W*0.05,H*0.95]])

cv2.imwrite("detect_纸张轮廓.jpg", img_show)

# 透视校正
A4_w, A4_h = 700, int(700 * 1.4142)
dst_pts = np.float32([[0,0],[A4_w,0],[A4_w,A4_h],[0,A4_h]])
M = cv2.getPerspectiveTransform(paper_box, dst_pts)
result = cv2.warpPerspective(img, M, (A4_w, A4_h))

cv2.imwrite("correct_final_校正完成图.jpg", result)
print("✅ A4纸张透视校正完成，全部运行完毕")