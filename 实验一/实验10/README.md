# Sinusoidal PE vs RoPE 位置编码对比实验
基于 PyTorch 实现正弦位置编码与旋转位置编码（RoPE），并完成可视化与对比验证。

## 功能
- 实现正弦位置编码（Sinusoidal Position Encoding）
- 实现二维向量旋转
- 实现高维 RoPE
- 对比 E+pos 与 RoPE 输入方式
- 验证 RoPE 相对位置性质
- 生成热力图、旋转轨迹、特征图、对比图

## 环境
```bash
pip install torch matplotlib numpy
```

## 运行
```bash
python main.py
```

## 输出图片
- sinusoidal_pe_heatmap.png  
- rope_2d_rotation.png  
- rope_vs_addition.png  
- rope_feature_map.png  
- rope_relative_verification.png

## 核心结论
- RoPE 用旋转表示位置，不混合内容与位置
- RoPE 天然建模相对位置，注意力更合理
- RoPE 长序列泛化更强，是大模型标配位置编码
