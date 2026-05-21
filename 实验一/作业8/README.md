# README
PyTorch CNN 图像分类实验（MNIST / CIFAR-10）

## 项目简介
使用卷积神经网络完成图像分类，包含基础任务 + 进阶实验：
- 基础：MNIST 手写数字分类
- 进阶1：改进网络结构（加深卷积、增加通道、Dropout）
- 进阶2：优化器对比（SGD vs Adam）
- 进阶3：数据集对比（MNIST vs CIFAR-10）

## 实验环境
- Python 3
- PyTorch
- torchvision
- matplotlib

## 运行方法
```
python main.py
```

## 主要结果
- MNIST 基础模型准确率：98.72%
- 改进模型准确率：99.21%
- Adam 效果优于 SGD
- CIFAR-10 更难，准确率约 64.26%

## 生成文件
- 样本图、训练曲线、测试预测图
