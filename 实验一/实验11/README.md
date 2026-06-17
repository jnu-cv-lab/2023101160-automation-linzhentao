# Badminton Skeleton Transformer
基于MediaPipe Pose + Transformer Encoder 的羽毛球6类击球动作时序分类项目

## 项目简介
原始视频提取人体33关键点骨架序列，消除像素冗余，使用轻量Transformer建模动作时序，完成击球动作识别。
完整流程：视频读取→骨架提取→归一化重采样→数据集划分→模型训练→测试评估→单视频推理/可视化

## 数据集
Kaggle badminton_storke_video
6类动作：
forehand_drive / forehand_lift / forehand_net_shot / forehand_clear / backhand_drive / backhand_net_shot

## 环境依赖
```bash
pip install torch opencv-python mediapipe scikit-learn numpy matplotlib scipy
```

## 文件说明
- main.py：一体化完整代码（预处理+模型+训练+推理+可视化）
- dataset_npy/：缓存骨架训练集、测试集与模型权重
- output_images/：训练曲线、混淆矩阵、注意力热力图、骨架视频

## 使用步骤
1. 将数据集放入 `/home/lin/cv-course/archive`，内部按6类分文件夹存放mp4视频
2. 运行代码：`python main.py`
3. 首次自动解析视频生成npy数据集，二次运行直接加载缓存加速

## 功能清单
1. 预处理：OpenCV读视频、MediaPipe提取132维骨架、髋中心+肩宽归一、插值固定30帧、分层划分训练/测试集并保存npy
2. 模型：2层多头Transformer Encoder，带正弦位置编码，可输出时序注意力权重
3. 训练评估：Adam+交叉熵损失，输出损失/精度曲线、混淆矩阵、分类报告
4. 推理：单视频输入输出动作类别与置信度
5. 可视化（拓展）：人体骨架标注视频、帧注意力热力图

## 实验效果
测试集总体准确率48.19%；正反手相似动作易混淆，模型自动聚焦击球后半段关键帧。

## 优缺点
### 优势
轻量化，无需处理大量图像像素，低设备即可训练；自注意力可定位动作关键时序片段，可解释性强
### 局限
人物遮挡、快速运动易丢失关键点；仅使用全身骨架，缺少球拍、场地视觉信息，同类动作区分度不足
