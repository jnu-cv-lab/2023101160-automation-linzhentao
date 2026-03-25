
# YCbCr 通道下采样与插值重建实验
基于 OpenCV 实现图像 YCbCr 色彩空间的色度通道（Cb/Cr）下采样、插值恢复，并通过**峰值信噪比（PSNR）** 量化重建图像与原图的质量差异，生成可视化对比结果。

## 实验原理
图像在 YCbCr 色彩空间中，**Y 通道**表示亮度信息，**Cb/Cr 通道**表示色度（色彩）信息；人眼对色度信息的敏感度远低于亮度，因此对 Cb/Cr 通道进行下采样不会显著降低视觉效果，这也是 JPEG 等图像压缩算法的核心思想之一。
1. 将 RGB 图像转换为 YCbCr 色彩空间，分离 Y、Cb、Cr 三个通道
2. 对 Cb、Cr 通道进行指定倍数的下采样，保留 Y 通道不变
3. 采用插值算法将下采样后的 Cb、Cr 通道恢复至原图像尺寸
4. 合并通道并转回 RGB 空间，得到重建图像
5. 计算 PSNR 评估重建图像质量，PSNR 越高（通常＞30dB），与原图差异越小，人眼越难区分

## 环境依赖
确保安装以下 Python 库，建议使用虚拟环境：
```bash
pip install opencv-python numpy matplotlib
```

## 快速使用
### 1. 项目结构
```
.
├── cv3.py          # 主实验代码
├── test.jpg        # 测试输入图像（需自行放入）
├── comparison_image.jpg  # 生成的原图+重建图对比图
└── reconstructed_image.jpg # 生成的单独重建图
```

### 2. 运行代码
1. 将测试图像命名为 `test.jpg`，放入代码同级目录
2. 直接运行 Python 脚本：
   ```bash
   python cv3.py
   ```

### 3. 参数自定义
可在代码中修改以下核心参数，适配不同实验需求：
```python
scale = 2  # 下采样倍数，可改为4/8等
interp_method = cv2.INTER_LINEAR  # 插值方法，可选：
# cv2.INTER_NEAREST （最近邻插值，速度快，精度低）
# cv2.INTER_CUBIC   （双三次插值，精度高，速度稍慢）
```

## 实验结果
运行代码后，控制台会输出关键实验数据，示例：
```
📊 下采样倍数: 2×2
🔍 插值方法: 双线性插值 (INTER_LINEAR)
📈 PSNR 值: 38.56 dB

✅ 结果文件已全部生成：
  - comparison_image.jpg（原图+重建图对比图）
  - reconstructed_image.jpg（单独重建图）
💡 PSNR 越高，图像质量越接近原图（通常>30dB人眼难以区分）
```
同时生成两张结果图：
- `comparison_image.jpg`：左右分栏展示**原图**和**重建图**，标题标注 PSNR 数值
- `reconstructed_image.jpg`：单独的重建后图像，可直接用于对比分析

## 核心函数说明
### `calculate_psnr(img1, img2)`
计算两幅图像的峰值信噪比，评估重建质量：
1. 计算两幅图像的均方误差（MSE）
2. 若 MSE=0，说明图像完全相同，返回 PSNR=100dB
3. 按公式 $PSNR=20\times\log_{10}(\frac{255}{\sqrt{MSE}})$ 计算 PSNR，单位为 dB

## 跨平台兼容
代码中已加入**Linux 环境适配**，解决 tkinter 缺失导致的 matplotlib 绘图报错问题：
```python
plt.switch_backend('Agg')  # 无GUI后端，仅保存图片不弹出窗口，兼容Linux服务器/无桌面环境
```
Windows/macOS 环境可注释此代码，支持弹窗显示对比图。

## 实验拓展方向
1. 测试不同**下采样倍数**（2/4/8）对图像质量和文件体积的影响
2. 对比不同**插值算法**（最近邻/双线性/双三次）的重建效果和运行速度
3. 统计不同类型图像（风景/人物/纹理）的色度下采样视觉差异
4. 结合文件压缩工具，测试色度下采样后的图像压缩比提升效果

## 注意事项
1. 测试图像需为 `jpg` 格式，若使用其他格式（png/bmp），需修改代码中 `cv2.imread` 的输入文件名
2. 确保测试图像路径正确，若读取失败，控制台会输出 `❌ 图片读取失败，请检查文件路径` 提示
3. 运行后会覆盖同级目录下同名的结果图，建议及时重命名保存实验结果

---
**Author**：实验项目自用  
**Language**：Python 3.x  
**Dependencies**：OpenCV, NumPy, Matplotlib
