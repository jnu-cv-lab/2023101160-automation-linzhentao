# 屏蔽MediaPipe冗余GL/EGL日志（适配WSL，消除刷屏）
import os
# WSL无GUI兼容，禁止matplotlib弹窗
os.environ["MPL_BACKEND"] = "Agg"
os.environ["GLOG_minloglevel"] = "2"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

# MediaPipe 人体姿态
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 可视化 & 评估
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import math
import json

# ==========================================
# 全局超参数配置（已改为你的WSL绝对路径）
# ==========================================
CONFIG = {
    'input_dim': 132,
    'target_frames': 30,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'num_classes': 6,
    'dropout': 0.2,
    'batch_size': 16,
    'epochs': 40,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'test_size': 0.2,
    'random_seed': 42,
    'data_dir': '/home/lin/cv-course/archive',  # 绝对路径，不会找不到
    'output_dir': './output_images',
    'npy_save_dir': './dataset_npy',
    'class_names': [
        'forehand_drive',
        'forehand_lift',
        'forehand_net_shot',
        'forehand_clear',
        'backhand_drive',
        'backhand_net_shot'
    ],
    'detect_conf': 0.3  # 降低阈值，减少识别失败
}

# 创建输出目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['npy_save_dir'], exist_ok=True)

# 固定随机种子（实验可复现）
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['random_seed'])

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 训练设备: {device}")

# 全局初始化Pose（仅创建一次，节省资源）
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=CONFIG['detect_conf'],
    min_tracking_confidence=0.3,
    model_complexity=1
)

# 保存标签映射json
label_map = {idx: name for idx, name in enumerate(CONFIG['class_names'])}
with open(os.path.join(CONFIG['npy_save_dir'], "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

# ==========================================
# 1. 骨架归一化：髋中心平移 + 肩宽尺度归一
# ==========================================
def normalize_skeleton(skel_data):
    T, feat_dim = skel_data.shape
    data = skel_data.copy()
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    for t in range(T):
        # 髋部中心点
        lh_x, lh_y = data[t, LEFT_HIP*4], data[t, LEFT_HIP*4+1]
        rh_x, rh_y = data[t, RIGHT_HIP*4], data[t, RIGHT_HIP*4+1]
        hip_cx = (lh_x + rh_x) / 2.0
        hip_cy = (lh_y + rh_y) / 2.0

        # 肩宽作为缩放因子
        ls_x = data[t, LEFT_SHOULDER * 4]
        rs_x = data[t, RIGHT_SHOULDER * 4]
        shoulder_w = abs(ls_x - rs_x)
        scale = shoulder_w if shoulder_w > 1e-6 else 1.0

        # 平移缩放所有关键点xy
        for k in range(33):
            x_idx = k * 4
            y_idx = k * 4
            data[t, x_idx] = (data[t, x_idx] - hip_cx) / scale
            data[t, y_idx] = (data[t, y_idx] - hip_cy)
    return data

# ==========================================
# 2. 单视频提取固定长度骨架序列
# ==========================================
def extract_skeleton_from_video(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return None

    frames_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            frame_feat = []
            for lm in results.pose_landmarks.landmark:
                frame_feat.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames_data.append(frame_feat)
    cap.release()

    if len(frames_data) == 0:
        return None
    frames_data = np.array(frames_data, dtype=np.float32)
    T_orig = frames_data.shape[0]
    indices = np.linspace(0, T_orig - 1, target_frames)
    resampled = np.zeros((target_frames, CONFIG['input_dim']), dtype=np.float32)
    for dim in range(CONFIG['input_dim']):
        resampled[:, dim] = np.interp(indices, np.arange(T_orig), frames_data[:, dim])
    resampled = normalize_skeleton(resampled)
    return resampled

# ==========================================
# 3. 数据集构建，缓存npy加速重复运行
# ==========================================
def prepare_dataset(data_dir):
    npy_root = CONFIG['npy_save_dir']
    # 优先读取缓存
    try:
        X_train = np.load(os.path.join(npy_root, "X_train.npy"))
        X_test = np.load(os.path.join(npy_root, "X_test.npy"))
        y_train = np.load(os.path.join(npy_root, "y_train.npy"))
        y_test = np.load(os.path.join(npy_root, "y_test.npy"))
        print("[INFO] 加载缓存NPY数据集，跳过视频解析")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("[INFO] 无缓存数据，开始解析全部视频...")

    if not os.path.exists(data_dir):
        print(f"[ERROR] 数据集文件夹不存在：{data_dir}")
        return None, None, None, None

    X, y = [], []
    class_to_idx = {name: idx for idx, name in enumerate(CONFIG['class_names'])}
    total_video = 0
    valid_video = 0

    for cls_name in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path) or cls_name not in class_to_idx:
            continue
        label = class_to_idx[cls_name]
        print(f"\n==== 处理类别 {cls_name} 标签={label} ====")

        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                total_video += 1
                vid_path = os.path.join(cls_path, fname)
                skel = extract_skeleton_from_video(vid_path, CONFIG['target_frames'])
                if skel is not None:
                    X.append(skel)
                    y.append(label)
                    valid_video += 1
                    print(f"成功读取: {fname}")
                else:
                    print(f"丢弃无人体视频: {fname}")
    print(f"\n[统计] 总视频数: {total_video} | 有效骨架样本: {valid_video}")

    if len(X) == 0:
        print("[ERROR] 未提取到任何有效人体骨架样本！")
        return None, None, None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"[INFO] 全部样本shape: {X.shape}")

    # 分层划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_seed'], stratify=y
    )
    # 保存缓存
    np.save(os.path.join(npy_root, "X_train.npy"), X_train)
    np.save(os.path.join(npy_root, "X_test.npy"), X_test)
    np.save(os.path.join(npy_root, "y_train.npy"), y_train)
    np.save(os.path.join(npy_root, "y_test.npy"), y_test)
    print(f"[INFO] 数据集已保存至 {npy_root}")
    return X_train, X_test, y_train, y_test

# ==========================================
# 4. 正弦位置编码
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==========================================
# 5. 自定义Encoder层，输出注意力权重
# ==========================================
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, src):
        src2, attn = self.self_attn(src, src, src, need_weights=True)
        src = self.norm1(src + self.drop1(src2))
        src2 = self.linear2(self.drop2(self.act(self.linear1(src))))
        src = self.norm2(src + self.drop2(src2))
        return src, attn

# ==========================================
# 6. 多层Encoder堆叠
# ==========================================
class CustomTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, src):
        attn_list = []
        out = src
        for l in self.layers:
            out, attn = l(out)
            attn_list.append(attn)
        return out, attn_list

# ==========================================
# 7. 完整骨架Transformer模型
# ==========================================
class SkeletonTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(cfg['input_dim'], cfg['d_model'])
        self.pos_enc = PositionalEncoding(cfg['d_model'], cfg['dropout'], cfg['target_frames'])
        enc_layer = CustomTransformerEncoderLayer(
            cfg['d_model'], cfg['nhead'], cfg['dim_feedforward'], cfg['dropout']
        )
        self.encoder = CustomTransformerEncoder(enc_layer, cfg['num_layers'])
        self.head = nn.Sequential(
            nn.Linear(cfg['d_model'], 64),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(64, cfg['num_classes'])
        )
        self.last_attn = None

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        feat, attn_weights = self.encoder(x)
        self.last_attn = attn_weights[-1].detach().cpu()
        feat_pool = torch.mean(feat, dim=1)
        return self.head(feat_pool)

# ==========================================
# 8. 训练、测试、绘图全套流程
# ==========================================
def train_and_evaluate():
    X_train, X_test, y_train, y_test = prepare_dataset(CONFIG['data_dir'])
    if X_train is None:
        return None

    train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = SkeletonTransformer(CONFIG).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = StepLR(opt, step_size=10, gamma=0.8)

    train_loss_hist = []
    test_acc_hist = []

    print("\n===== 开始训练 =====")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
        avg_loss = total_loss / len(train_set)
        train_loss_hist.append(avg_loss)

        # 测试
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                all_pred.extend(pred.cpu().tolist())
                all_true.extend(y.cpu().tolist())
        acc = accuracy_score(all_true, all_pred) * 100
        test_acc_hist.append(acc)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d} | Loss:{avg_loss:.4f} Test Acc:{acc:.2f}%")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_hist, c='royalblue', lw=2)
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(test_acc_hist, c='forestgreen', lw=2)
    plt.title("Test Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "training_curves.png"), dpi=150)
    plt.close()

    # 混淆矩阵
    print("\n===== 分类报告 =====")
    print(classification_report(all_true, all_pred, target_names=CONFIG['class_names'], digits=4))
    cm = confusion_matrix(all_true, all_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=range(CONFIG['num_classes']))
    plt.figure(figsize=(8,8))
    disp.plot(ax=plt.gca(), cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "confusion_matrix.png"), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(CONFIG['npy_save_dir'], "badminton_transformer.pth"))
    print("[INFO] 模型权重保存完成")
    return model

# ==========================================
# 9. 单视频推理输出类别+置信度
# ==========================================
def inference_single_video(model, video_path):
    print("\n===== 单视频推理 =====")
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        return None
    feat = extract_skeleton_from_video(video_path, CONFIG['target_frames'])
    if feat is None:
        print("[ERROR] 骨架提取失败，无法推理")
        return None
    input_tensor = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = np.argmax(prob)
    pred_name = CONFIG['class_names'][pred_idx]
    conf = prob[pred_idx]
    print(f"预测动作: {pred_name}")
    print(f"置信度: {conf:.2f}")
    return input_tensor

# ==========================================
# 10. 绘制骨架可视化视频
# ==========================================
def draw_skeleton_video(src_video, dst_name="skeleton_vis.mp4"):
    if not os.path.exists(src_video):
        print(f"[WARN] 跳过不存在视频 {src_video}")
        return
    cap = cv2.VideoCapture(src_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = os.path.join(CONFIG['output_dir'], dst_name)
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec((0,255,0), 2, 3),
                mp_drawing.DrawingSpec((255,0,0), 2)
            )
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"[INFO] 骨架视频已保存: {save_path}")

# ==========================================
# 11. 注意力热力图绘制
# ==========================================
def plot_attention(model, input_tensor):
    if model.last_attn is None:
        print("[WARN] 无注意力权重，跳过绘图")
        return
    attn = model.last_attn[0].numpy()
    plt.figure(figsize=(10,8))
    plt.imshow(attn, cmap="hot", aspect="auto")
    plt.colorbar()
    plt.title("Transformer Frame Attention Heatmap")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.tight_layout()
    save_path = os.path.join(CONFIG['output_dir'], "attention_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] 注意力热力图保存: {save_path}")

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    trained_model = train_and_evaluate()
    if trained_model is not None:
        # 推理视频路径，按需修改文件名
        vid1 = os.path.join(CONFIG['data_dir'], "forehand_clear", "002.mp4")
        vid2 = os.path.join(CONFIG['data_dir'], "backhand_net_shot", "003.mp4")

        # 生成骨架可视化视频
        draw_skeleton_video(vid1, "skeleton_video_1.mp4")
        draw_skeleton_video(vid2, "skeleton_video_2.mp4")

        # 推理+注意力图
        input_t = inference_single_video(trained_model, vid1)
        if input_t is not None:
            plot_attention(trained_model, input_t)

    print("\n==== 全部流程执行完成 ====")