2023101160-automation-linzhentao-homework1
自动化专业 Python/OpenCV 图像处理作业 1  
**作者**：linzhentao  
**学号**：2023101160  
**专业**：自动化  

## 项目概述
本作业围绕 OpenCV 库展开数字图像处理基础实践，核心实现了图像的读取、基础信息解析、色彩空间转换、灰度化处理、区域裁剪、结果保存与可视化展示等功能。通过本次作业，掌握 Python 环境下 OpenCV 库的基本使用方法，理解数字图像的存储格式（BGR/RGB）、像素组成及基础处理逻辑，为后续复杂图像处理（如滤波、边缘检测、特征提取）奠定基础。

## 技术栈
- **编程语言**：Python 3.8+（兼容 3.7 及以上版本）
- **核心库**：
  - OpenCV-Python（cv2）：图像读取、处理、保存
  - NumPy：图像数组操作、像素值处理
  - Matplotlib：图像可视化展示

## 文件清单
| 文件名          | 类型       | 功能说明                                                                 |
|-----------------|------------|--------------------------------------------------------------------------|
| `import cv2.py` | 源代码文件 | 主程序入口，包含所有图像处理逻辑：图片读取校验、信息打印、色彩转换、灰度化、裁剪、显示、保存 |
| `test.jpg`      | 输入文件   | 原始测试图像，作为程序处理的输入源（建议使用 jpg/png 格式，避免特殊编码格式）|
| `gray_test.jpg` | 输出文件   | 程序运行后自动生成的灰度化处理结果图，保留原始图像尺寸，仅保留亮度信息       |
| `crop_test.jpg` | 输出文件   | 程序运行后自动生成的裁剪结果图，截取原始图像左上角 100×100 像素区域         |
| `README.md`     | 文档文件   | 项目完整说明，包含环境配置、运行步骤、功能解析、常见问题等                 |

## 环境配置
### 1. 环境要求
- 操作系统：Windows 10/11、Linux（Ubuntu 18.04+）、macOS 12+ 均可
- Python 版本：3.7 及以上（推荐 3.8-3.10，兼容性最佳）

### 2. 依赖安装
#### 方式 1：pip 直接安装（推荐）
打开终端/命令提示符，执行以下命令安装所有依赖：
```bash
# 安装 OpenCV-Python
pip install opencv-python
# 安装 NumPy（OpenCV 依赖，通常会自动安装）
pip install numpy
# 安装 Matplotlib（用于图像显示）
pip install matplotlib

# 批量安装（一行命令）
pip install opencv-python numpy matplotlib
```

#### 方式 2：国内镜像源安装（解决下载慢/失败问题）
若直接安装超时或失败，使用清华镜像源加速：
```bash
pip install opencv-python numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 方式 3：固定版本安装（可选，保证环境一致性）
新建 `requirements.txt` 文件，写入以下内容：
```txt
opencv-python==4.8.1.78
numpy==1.24.3
matplotlib==3.7.2
```
执行安装命令：
```bash
pip install -r requirements.txt
```

## 运行步骤
### 1. 前置准备
- 将 `import cv2.py` 和 `test.jpg` 放在同一目录下，避免路径嵌套（如 `src/import cv2.py` 需同步修改图片读取路径）；
- 确认 `test.jpg` 文件未损坏，可通过系统图片查看器正常打开。

### 2. 运行程序
#### 方式 1：终端/命令行运行（推荐）
1. 打开终端/命令提示符，切换到文件所在目录（示例：Windows 系统）：
   ```bash
   cd D:\homework\2023101160-automation-linzhentao-homework1
   ```
2. 执行运行命令：
   ```bash
   # Windows/Linux/macOS 通用
   python "import cv2.py"
   # 若系统存在多个 Python 版本，指定 Python3
   python3 "import cv2.py"
   ```

#### 方式 2：IDE 运行（VS Code/PyCharm）
1. 打开 IDE 并导入项目目录；
2. 右键点击 `import cv2.py` 文件，选择「运行」/「Run」；
3. 等待程序执行，自动弹出图像显示窗口。

### 3. 运行结果
程序执行后会完成以下操作：
1. **控制台输出**：打印图像尺寸、通道数、像素类型、指定坐标像素值等基础信息；
2. **图像可视化**：弹出包含「原始图像、灰度图、裁剪图」的展示窗口（无坐标轴干扰）；
3. **文件生成**：在项目目录下自动生成 `gray_test.jpg`（灰度图）和 `crop_test.jpg`（裁剪图）。

## 核心功能解析
### 1. 图像读取与有效性校验
```python
import cv2
# 读取图像（OpenCV 默认 BGR 格式）
img = cv2.imread("test.jpg")
# 校验读取结果，避免文件不存在/损坏导致程序崩溃
if img is None:
    print("图片读取失败！请检查文件路径或格式")
    exit()
```
- 核心函数：`cv2.imread()` 返回图像的 NumPy 数组，`None` 表示读取失败；
- 关键作用：提升程序健壮性，避免因文件问题导致崩溃。

### 2. 图像基础信息解析
```python
# 获取图像尺寸（高, 宽, 通道数）
height, width, channels = img.shape
# 获取像素数据类型（通常为 uint8）
dtype = img.dtype
# 获取指定坐标(100,100)的 BGR 像素值
pixel_value = img[100, 100]
print(f"图像尺寸：{height}×{width}，通道数：{channels}，像素类型：{dtype}")
print(f"坐标(100,100)像素值：{pixel_value}")
```
- 坐标规则：OpenCV 像素坐标为 `(行, 列)`，对应图像的 `(y, x)` 轴，而非常规 `(x, y)`。

### 3. 色彩空间转换与图像显示
```python
import matplotlib.pyplot as plt
# BGR 转 RGB（适配 Matplotlib 显示规则）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图像（隐藏坐标轴）
plt.imshow(img_rgb)
plt.title("Original Image (RGB)")
plt.axis("off")
plt.show()
```
- 核心问题：OpenCV 读取为 BGR 格式，Matplotlib 按 RGB 显示，直接显示会色彩失真；
- 解决方法：`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 实现通道反转。

### 4. 灰度化与裁剪处理
```python
# 彩色图转灰度图（通道数从 3 降为 1）
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 裁剪左上角 100×100 区域（切片规则：行起始:行结束, 列起始:列结束）
crop_img = img[0:100, 0:100]

# 保存处理结果
cv2.imwrite("gray_test.jpg", gray_img)
cv2.imwrite("crop_test.jpg", crop_img)
```
- 灰度化原理：加权平均 RGB 通道值（Y=0.299R+0.587G+0.114B），保留亮度信息；
- 裁剪规则：NumPy 数组切片 `img[y1:y2, x1:x2]`，左闭右开，0:100 对应 100×100 像素区域。

## 常见问题排查
| 问题现象                | 原因与解决方法                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| 程序提示「图片读取失败」 | 文件路径错误/名称大小写问题/格式不支持 → 确认文件同目录、名称一致、使用 jpg/png 格式 |
| 图像显示色彩失真        | 未转换色彩空间 → 显示前执行 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`              |
| 安装库时报错            | 网络问题 → 使用国内镜像源；Python 版本过低 → 升级到 3.7+                          |
| 图像窗口无响应          | 系统图形界面异常 → 关闭所有窗口后重启程序                                      |

## 作业总结
本次作业完成了 OpenCV 图像处理基础实践，掌握了图像读取、色彩转换、灰度化、裁剪等核心操作，理解了数字图像的存储格式与可视化规则。后续可扩展方向包括：
1. 实现图像旋转、缩放、翻转等几何变换；
2. 增加均值滤波、高斯滤波等去噪操作；
3. 基于 Canny 算法实现边缘检测。
