### README.md
```markdown
# CNN-MNIST-Experiments
基于PyTorch实现CNN手写数字分类进阶实验

## 功能
1. 基础模型重训练
2. 优化器对比：SGD / SGD+Momentum / Adam
3. 学习率对比：0.1 / 0.01 / 0.001
4. 卷积核、特征图可视化
5. 错误样本分析与混淆矩阵绘制

## 环境
```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

## 运行
```bash
python main.py
```

## 输出图片
- task4_kernels.png 卷积核
- task5_featuremap.png 特征图
- task6_errors.png 错误样本
- task7_cm.png 混淆矩阵

## 实验小结
- 模型测试准确率：98.83%
- 最优组合：Adam + 学习率 0.001
- 数字3与8、4与9最易混淆
```
