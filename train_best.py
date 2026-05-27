# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

DATA_PATH = "lotto_clean.csv"
SAVE_DIR = "checkpoints_best"
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pdparams")

SEQ_LEN = 10
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
paddle.seed(42)


def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                nums = [int(x) for x in parts[1:]]
                rows.append(nums)

    # 旧 -> 新
    rows = rows[::-1]
    return np.array(rows, dtype="float32")


def build_samples(data, seq_len=10):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len].reshape(-1))
        ys.append(data[i + seq_len])
    return np.array(xs, dtype="float32"), np.array(ys, dtype="float32")


class LottoMLP(nn.Layer):
    def __init__(self, input_dim=70, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def postprocess(pred):
    pred = pred.tolist()

    front = [round(x) for x in pred[:5]]
    back = [round(x) for x in pred[5:]]

    front = [min(max(x, 1), 35) for x in front]
    back = [min(max(x, 1), 12) for x in back]

    front = sorted(list(set(front)))
    back = sorted(list(set(back)))

    while len(front) < 5:
        x = random.randint(1, 35)
        if x not in front:
            front.append(x)
    while len(back) < 2:
        x = random.randint(1, 12)
        if x not in back:
            back.append(x)

    return sorted(front[:5]) + sorted(back[:2])


print("Loading data...")
data = load_data(DATA_PATH)
x, y = build_samples(data, SEQ_LEN)

split = int(len(x) * 0.9)

# 时间序列切分：旧数据训练，最新 10% 验证
x_train, y_train = x[:split], y[:split]
x_val, y_val = x[split:], y[split:]

train_ds = paddle.io.TensorDataset([
    paddle.to_tensor(x_train),
    paddle.to_tensor(y_train)
])
val_ds = paddle.io.TensorDataset([
    paddle.to_tensor(x_val),
    paddle.to_tensor(y_val)
])

train_loader = paddle.io.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = paddle.io.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = LottoMLP(input_dim=SEQ_LEN * 7, output_dim=7)

optimizer = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=model.parameters(),
    weight_decay=WEIGHT_DECAY
)

best_val_loss = float("inf")
wait = 0

print("Start training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = F.mse_loss(pred, batch_y)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        train_losses.append(float(loss.numpy()))

    model.eval()
    val_losses = []

    with paddle.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            val_losses.append(float(loss.numpy()))

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        paddle.save(model.state_dict(), BEST_PATH)
        print(f"  ✅ Saved best model: {BEST_PATH}")
    else:
        wait += 1
        print(f"  ⚠️ No improvement. patience={wait}/{PATIENCE}")

    if wait >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training finished.")
print("Best val loss:", best_val_loss)
