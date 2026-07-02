import cv2
import numpy as np
import os

# ===================== 1. 标定参数配置 =====================
# 你的棋盘横向9内角点，纵向6内角点，参数不变
CHESSBOARD_SIZE = (9, 6)
# 棋盘方格边长 mm
SQUARE_LENGTH = 25
# 固定图片绝对路径
IMG_FOLDER = "/home/lin/cv-course/"
OUTPUT_CORNER = "/home/lin/cv-course/corner_draw"
OUTPUT_UNDIST = "/home/lin/cv-course/undist_result"
# 亚像素优化迭代条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ===================== 2. 棋盘格三维世界坐标 =====================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_LENGTH

obj_points = []
img_points = []

os.makedirs(OUTPUT_CORNER, exist_ok=True)
os.makedirs(OUTPUT_UNDIST, exist_ok=True)

# ===================== 3. 读取 img1.jpg ~ img15.jpg =====================
img_list = []
for file_name in os.listdir(IMG_FOLDER):
    if file_name.startswith("img") and file_name.endswith(".jpg"):
        full_path = os.path.join(IMG_FOLDER, file_name)
        img_list.append(full_path)

# 按数字1~15升序排序
def sort_num(file_path):
    name = os.path.basename(file_path)
    num = int(name.replace("img", "").replace(".jpg", ""))
    return num
img_list.sort(key=sort_num)

print(f"===== 读取图片列表，共 {len(img_list)} 张 =====")
for p in img_list:
    print(os.path.basename(p))

if len(img_list) == 0:
    print("❌ 未找到 img1~img15.jpg")
    exit()

img_size = None

# 逐张预处理降噪，解决屏幕反光识别失败
for img_path in img_list:
    fname = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ {fname} 读取失败，跳过")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 新增预处理：高斯模糊降噪 + 直方图均衡，消除反光光斑
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    if img_size is None:
        img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_sub)
        draw_img = cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_sub, ret)
        cv2.imwrite(os.path.join(OUTPUT_CORNER, f"corner_{fname}"), draw_img)
        print(f"✅ {fname} 角点识别成功")
    else:
        print(f"❌ {fname} 角点识别失败（屏幕反光遮挡格子）")

# ===================== 4. 统计有效图片 =====================
valid_num = len(obj_points)
print(f"\n===== 识别完成统计 =====")
print(f"有效可标定图片数量：{valid_num}")
if valid_num < 10:
    print("警告：有效图片不足10张，标定精度极差！")
    print("解决办法：打印纸质棋盘，避免屏幕反光拍摄")
    if valid_num == 0:
        exit()

# ===================== 5. 相机标定 =====================
print("\n开始标定相机……")
reproj_err, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

print("=" * 60)
print(f"整体平均重投影误差：{reproj_err:.4f} 像素")
print("\n相机内参矩阵 K：")
print(K)
print("\n畸变系数 [k1,k2,p1,p2,k3]：")
print(dist[0])
print("=" * 60)

# ===================== 6. 去畸变矫正 =====================
demo_img = cv2.imread(img_list[0])
h, w = demo_img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), 5)
undist_full = cv2.remap(demo_img, map_x, map_y, cv2.INTER_LINEAR)
x, y, w_roi, h_roi = roi
undist_crop = undist_full[y:y+h_roi, x:x+w_roi]

cv2.imwrite(os.path.join(IMG_FOLDER, "demo_original.jpg"), demo_img)
cv2.imwrite(os.path.join(OUTPUT_UNDIST, "undist_full.jpg"), undist_full)
cv2.imwrite(os.path.join(OUTPUT_UNDIST, "undist_crop.jpg"), undist_crop)
print("\n去畸变对比图已保存")

# ===================== 7. 单张重投影误差 =====================
sum_err = 0
for i in range(valid_num):
    proj_p, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(img_points[i], proj_p, cv2.NORM_L2) / len(proj_p)
    sum_err += err
print(f"单张图片平均重投影误差均值：{sum_err / valid_num:.4f}")

# 弹窗展示
cv2.imshow("Original Image", demo_img)
cv2.imshow("Undistorted Crop", undist_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()