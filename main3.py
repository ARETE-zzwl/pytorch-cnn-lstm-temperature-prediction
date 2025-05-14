# main_multivar.py ─────────────────────────────────────────────────────────
# 预测目标：meantemp / humidity / wind_speed / meanpressure (多变量单步预测)
# ──────────────────────────────────────────────────────────────────────────

import os, random, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ————— 0. 基本参数 —————
CSV_PATH    = "data/DailyDelhiClimateTrain.csv"  # ← 修改为数据文件路径
TIME_STEP   = 12
TARGET_COLS = ["meantemp", "humidity", "wind_speed", "meanpressure"]
TRAIN_PCT   = 0.8
NUM_EPOCHS  = 500
BATCH_SIZE  = 64
LR          = 1e-4
HIDDEN_SIZE = 64
LSTM_LAYERS = 2
SEED        = 42
SAVE_DIR    = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ————— 1. 固定随机种子 —————
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ————— 2. 读取数据 —————
df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date")
FEATURE_COLS = TARGET_COLS[:]                 # 这次输入特征 = 预测目标
features = df[FEATURE_COLS].values            # (N, 4)
targets  = df[TARGET_COLS].values             # (N, 4)

# ————— 3. 归一化 —————
scaler_X = MinMaxScaler(); X_scaled = scaler_X.fit_transform(features)
scaler_y = MinMaxScaler(); y_scaled = scaler_y.fit_transform(targets)

# ————— 4. 构造滑动窗序列 —————
def build_sequences(X, y, step):
    Xs, ys = [], []
    for i in range(len(X) - step):
        Xs.append(X[i : i + step])   # (step, F)
        ys.append(y[i + step])       # (F,)
    return np.array(Xs), np.array(ys)

dataX, dataY = build_sequences(X_scaled, y_scaled, TIME_STEP)
print("dataX:", dataX.shape, "dataY:", dataY.shape)  # (N-T, 12, 4) (N-T,4)

# ————— 5. 划分训练 / 测试 —————
split = int(len(dataX) * TRAIN_PCT)
train_X, train_y = dataX[:split], dataY[:split]
test_X , test_y  = dataX[split:], dataY[split:]
print("train:", train_X.shape, "test:", test_X.shape)

# ————— 6. CNN-LSTM 模型 —————
class CNN_LSTM(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, out_dim=4):
        super().__init__()
        self.conv = nn.Conv1d(n_features, n_features, kernel_size=1)
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden, out_dim)

    def forward(self, x):            # x:(B,T,F)
        x = self.conv(x.permute(0, 2, 1))      # (B,F,T)
        x = x.permute(0, 2, 1)                 # (B,T,F)
        out, _ = self.lstm(x)                  # 自动 h0/c0
        return self.fc(out[:, -1, :])          # 取最后步

# ————— 7. 评价指标（逐列） —————
def metrics_per_column(pred, true, col_names):
    if torch.is_tensor(pred): pred = pred.cpu().numpy()
    if torch.is_tensor(true): true = true.cpu().numpy()
    res = {}
    for i, name in enumerate(col_names):
        p, t = pred[:, i], true[:, i]
        mse  = np.mean((p - t) ** 2)
        res[name] = dict(
            MAE  = mean_absolute_error(t, p),
            RMSE = np.sqrt(mse),
            R2   = r2_score(t, p),
            MAPE = np.mean(np.abs((t - p) / (t + 1e-8))) * 100
        )
    return res

# ————— 8. 训练准备 —————
device  = "cuda" if torch.cuda.is_available() else "cpu"
n_feat  = train_X.shape[2]
out_dim = train_y.shape[1]

model = CNN_LSTM(n_feat, HIDDEN_SIZE, LSTM_LAYERS, out_dim).to(device)
crit  = nn.MSELoss()
opt   = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))

X_train = torch.tensor(train_X, dtype=torch.float32).to(device)
y_train = torch.tensor(train_y, dtype=torch.float32).to(device)
X_test  = torch.tensor(test_X , dtype=torch.float32).to(device)
y_test  = torch.tensor(test_y , dtype=torch.float32).to(device)

best_rmse = np.inf  # 基于平均 RMSE 保存最佳模型

# ————— 9. 训练循环 —————
for epoch in range(1, NUM_EPOCHS + 1):
    # —— 训练阶段 ——
    model.train()
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(idx), BATCH_SIZE):
        bx = X_train[idx[i:i+BATCH_SIZE]]
        by = y_train[idx[i:i+BATCH_SIZE]]
        opt.zero_grad()
        loss = crit(model(bx), by)
        loss.backward()
        opt.step()

    # —— 评估阶段 ——
    if epoch % 50 == 0 or epoch == NUM_EPOCHS:
        model.eval()
        with torch.no_grad():
            pred_t = model(X_test)
        m_dict = metrics_per_column(pred_t, y_test, TARGET_COLS)

        print(f"[{epoch:>3}/{NUM_EPOCHS}]")
        for k, v in m_dict.items():
            print(f"  {k:<13} MAE={v['MAE']:.4f}  RMSE={v['RMSE']:.4f}  "
                  f"R²={v['R2']:.4f}  MAPE={v['MAPE']:.2f}%")

        # 按平均 RMSE 选最佳
        avg_rmse = np.mean([v["RMSE"] for v in m_dict.values()])
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "cnn_lstm_best.pth"))

# ————— 10. 保存最终权重和 scaler —————
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cnn_lstm_last.pth"))
with open(os.path.join(SAVE_DIR, "scaler_X.pkl"), "wb") as f: pickle.dump(scaler_X, f)
with open(os.path.join(SAVE_DIR, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)
print("\n模型与 scaler 已保存至", SAVE_DIR)

# ————— 11. 反归一化可视化 & 全样本指标 —————
model.eval()
with torch.no_grad():
    full_pred = model(torch.tensor(dataX, dtype=torch.float32).to(device)).cpu().numpy()

true_inv = scaler_y.inverse_transform(dataY)
pred_inv = scaler_y.inverse_transform(full_pred)

plt.figure(figsize=(11, 6))
for i, col in enumerate(TARGET_COLS):
    plt.subplot(len(TARGET_COLS), 1, i+1)
    plt.plot(true_inv[:, i], label=f"True {col}", linewidth=1)
    plt.plot(pred_inv[:, i],  label="Pred", linewidth=1)
    plt.ylabel(col)
    if i == 0:
        plt.title("CNN-LSTM prediction vs reality")
        plt.legend()
plt.tight_layout(); plt.show()

final_dict = metrics_per_column(pred_inv, true_inv, TARGET_COLS)
print("\n—— 最终全样本指标 ——")
for k, v in final_dict.items():
    print(f"{k:<13} MAE={v['MAE']:.4f}  RMSE={v['RMSE']:.4f}  "
          f"R²={v['R2']:.4f}  MAPE={v['MAPE']:.2f}%")
# ──────────────────────────────────────────────────────────────────────────
