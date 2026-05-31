# -*- coding: utf-8 -*-

import random
import numpy as np
import paddle
import paddle.nn as nn
from datetime import datetime

paddle.set_device("cpu")

random.seed(42)
np.random.seed(42)
paddle.seed(42)

DATA_PATH = "lotto_clean.csv"
MODEL_PATH = "checkpoints_best/best_model.pdparams"
SEQ_LEN = 10

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


def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                rows.append([int(x) for x in parts[1:]])
    return np.array(rows[::-1], dtype="float32")


def fix_numbers(pred):
    pred = pred.tolist()
    front = [min(max(round(x), 1), 35) for x in pred[:5]]
    back = [min(max(round(x), 1), 12) for x in pred[5:]]
    front = sorted(set(front))
    back = sorted(set(back))
    while len(front) < 5:
        x = random.randint(1, 35)
        if x not in front:
            front.append(x)
    while len(back) < 2:
        x = random.randint(1, 12)
        if x not in back:
            back.append(x)
    return sorted(front[:5]) + sorted(back[:2])


# ── 加载数据和模型 ──────────────────────────────────────────
data = load_data(DATA_PATH)
latest_seq = data[-SEQ_LEN:].reshape(1, -1).astype("float32")

model = LottoMLP()
model.set_state_dict(paddle.load(MODEL_PATH))
model.eval()

x = paddle.to_tensor(latest_seq)
with paddle.no_grad():
    raw_pred = model(x).numpy()[0]

main_pred = fix_numbers(raw_pred)

# ── 构建输出内容 ────────────────────────────────────────────
output_lines = []
output_lines.append("Raw prediction:\n")
output_lines.append(str(raw_pred) + "\n\n")

output_lines.append("Most likely prediction:\n")
output_lines.append(f"{main_pred[:5]} + {main_pred[5:]}\n\n")

output_lines.append("Randomized candidate sets:\n")
for i in range(10):
    noise = np.random.normal(0, 2.0, size=raw_pred.shape)
    candidate = fix_numbers(raw_pred + noise)
    output_lines.append(f"Set {i+1}: {candidate[:5]} + {candidate[5:]}\n")

result_text = "".join(output_lines)

# ── 打印到终端（Actions 日志可见）──────────────────────────
print(result_text)

# ── 写入文件（artifact 可下载）─────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"prediction_{timestamp}.txt"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"Results saved to: {output_filename}")
