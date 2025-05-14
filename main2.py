# -------------------------------
#  main2.py  ―  CNN‑LSTM 温度预测
# -------------------------------
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

# ========== 1. 固定随机种子 ==========
SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ========== 2. 读数据 & 归一化 ==========
csv_path = "data/DailyDelhiClimateTrain.csv"
train_df = pd.read_csv(csv_path)
meantemp = train_df["meantemp"].values

plt.plot(meantemp); plt.title("Original mean temperature"); plt.show()

scaler = MinMaxScaler()
meantemp_scaled = scaler.fit_transform(meantemp.reshape(-1, 1))

# ========== 3. 构造时序样本 ==========
def build_sequences(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step])
        y.append(data[i + time_step])
    return np.array(X).reshape(-1, time_step, 1), np.array(y)

TIME_STEP = 12
dataX, dataY = build_sequences(meantemp_scaled, TIME_STEP)
print("dataX:", dataX.shape, "dataY:", dataY.shape)

# ========== 4. 划分训练 / 测试 ==========
def split_train_test(X, y, pct=0.8, shuffle=False):
    if shuffle:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
    s = int(len(X) * pct)
    return X[:s], y[:s], X[s:], y[s:]

train_X, train_y, test_X, test_y = split_train_test(dataX, dataY)
print("train:", train_X.shape, "test:", test_X.shape)

# ========== 5. 模型定义 ==========
class CNN_LSTM(nn.Module):
    def __init__(self, in_channels=1, hidden=64, layers=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.lstm = nn.LSTM(in_channels, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, 1) → (B, 1, T)
        x = self.conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)          # (B, T, C)
        out, _ = self.lstm(x)           # 由 PyTorch 自动创建 h0/c0，设备一致
        return self.fc(out[:, -1, :])   # 取最后一个时间步

# ========== 6. 训练准备 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))

X_train = torch.tensor(train_X, dtype=torch.float32).to(device)
y_train = torch.tensor(train_y, dtype=torch.float32).to(device)
X_test  = torch.tensor(test_X , dtype=torch.float32).to(device)
y_test  = torch.tensor(test_y , dtype=torch.float32).to(device)

NUM_EPOCHS, BATCH_SIZE = 500, 64
best_rmse = np.inf
save_dir = "checkpoints"; os.makedirs(save_dir, exist_ok=True)

def calc_metrics(pred, true):
    pred, true = pred.cpu().numpy(), true.cpu().numpy()
    mse  = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
    r2   = r2_score(true, pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}

# ========== 7. 训练 ==========
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(idx), BATCH_SIZE):
        batch_X = X_train[idx[i:i+BATCH_SIZE]]
        batch_y = y_train[idx[i:i+BATCH_SIZE]]
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()

    # 每 50 epoch 验证一次
    if epoch % 50 == 0 or epoch == NUM_EPOCHS:
        model.eval()
        with torch.no_grad():
            pred_t = model(X_test)
            m = calc_metrics(pred_t, y_test)
            print(f"[{epoch:>3}/{NUM_EPOCHS}]  MAE={m['MAE']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")
            # 保存最佳
            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "cnn_lstm_best.pth"))

# ========== 8. 保存最终模型+Scaler ==========
torch.save(model.state_dict(), os.path.join(save_dir, "cnn_lstm_last.pth"))
with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("\n模型与归一化器已保存至 checkpoints/")

# ========== 9. 反归一化可视化 ==========
model.eval()
with torch.no_grad():
    full_pred = model(torch.tensor(dataX, dtype=torch.float32).to(device)).cpu().numpy()
true_inv = scaler.inverse_transform(dataY).ravel()
pred_inv = scaler.inverse_transform(full_pred).ravel()

plt.figure(figsize=(10, 4))
plt.plot(true_inv, label="True", linewidth=1)
plt.plot(pred_inv, label="Pred", linewidth=1)
plt.title("CNN‑LSTM prediction vs reality (mean temp)")
plt.legend(); plt.tight_layout(); plt.show()

final_m = calc_metrics(torch.tensor(pred_inv), torch.tensor(true_inv))
print("全样本指标：", {k: f"{v:.4f}" for k, v in final_m.items()})
