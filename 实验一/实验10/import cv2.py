import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 正弦位置编码 =====================
def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

# ===================== 2. 二维向量旋转 =====================
def rotate_2d(vec, theta):
    c, s = math.cos(theta), math.sin(theta)
    x, y = vec
    return (x * c - y * s, x * s + y * c)

# ===================== 3. 高维 RoPE =====================
def rope(x, pos):
    d = x.size(-1)
    half_d = d // 2
    theta = torch.pow(10000.0, -2 * torch.arange(0, half_d) / d).to(x.device)
    pos_theta = pos.unsqueeze(1) * theta
    cos = torch.cos(pos_theta)
    sin = torch.sin(pos_theta)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rx = x1 * cos - x2 * sin
    ry = x1 * sin + x2 * cos

    out = torch.zeros_like(x)
    out[..., 0::2] = rx
    out[..., 1::2] = ry
    return out

# ===================== 4. 生成你要的4张图 =====================
def generate_all_plots():
    max_len = 50
    d_model = 64
    device = torch.device("cpu")

    # --- 图1：正弦位置编码热力图 ---
    pe = sinusoidal_pe(max_len, d_model)
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Sinusoidal Position Encoding (Heatmap)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("sinusoidal_pe_heatmap.png", dpi=150)
    plt.close()

    # --- 图2：E+pos vs RoPE 对比热力图 ---
    # 随机词嵌入
    emb = torch.randn(max_len, d_model)
    # E+pos
    pe = sinusoidal_pe(max_len, d_model)
    e_add_pos = emb + pe
    # RoPE
    emb_rope = rope(emb, torch.arange(max_len))

    plt.figure(figsize=(16, 6))
    plt.suptitle("Input Method Comparison (MAX Visual Effect)", fontsize=16)
    plt.subplot(1, 2, 1)
    plt.imshow(e_add_pos.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("E + Pos (Addition)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")

    plt.subplot(1, 2, 2)
    plt.imshow(emb_rope.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("RoPE (Rotation)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("rope_vs_addition.png", dpi=150)
    plt.close()

    # --- 图3：高维 RoPE 特征图 ---
    plt.figure(figsize=(10, 6))
    plt.imshow(emb_rope.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("High-Dimensional RoPE Feature Map")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("rope_feature_map.png", dpi=150)
    plt.close()

    # --- 图4：RoPE 相对位置性质验证柱状图 ---
    q = torch.randn(1, d_model)
    k = torch.randn(1, d_model)

    # 相对距离 Δ=2
    dot1 = (rope(q, torch.tensor([1])) @ rope(k, torch.tensor([3])).T).item()
    dot2 = (rope(q, torch.tensor([5])) @ rope(k, torch.tensor([7])).T).item()
    # 相对距离 Δ=5
    dot3 = (rope(q, torch.tensor([0])) @ rope(k, torch.tensor([5])).T).item()

    labels = ["(1,3) Δ=2", "(5,7) Δ=2", "(0,5) Δ=5"]
    values = [dot1, dot2, dot3]
    colors = ["blue", "blue", "orange"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.title("RoPE Relative Position Property Verification")
    plt.ylabel("Q-K Dot Product")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("rope_relative_verification.png", dpi=150)
    plt.close()

    print("✅ 所有图片已生成：")
    print("1. sinusoidal_pe_heatmap.png")
    print("2. rope_vs_addition.png")
    print("3. rope_feature_map.png")
    print("4. rope_relative_verification.png")

# ===================== 主程序 =====================
if __name__ == "__main__":
    generate_all_plots()