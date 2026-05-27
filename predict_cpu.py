# -*- coding: utf-8 -*-
import random
import numpy as np
import paddle
import paddle.nn as nn

paddle.set_device("cpu")

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

data = load_data(DATA_PATH)
latest_seq = data[-SEQ_LEN:].reshape(1, -1).astype("float32")

model = LottoMLP()
model.set_state_dict(paddle.load(MODEL_PATH))
model.eval()

x = paddle.to_tensor(latest_seq)

with paddle.no_grad():
    pred = model(x).numpy()[0]

main_pred = fix_numbers(pred)

print("Most likely prediction:")
print(main_pred[:5], "+", main_pred[5:])

print("\nRandomized candidate sets:")
for i in range(10):
    noise = np.random.normal(0, 2.0, size=pred.shape)
    candidate = fix_numbers(pred + noise)
    print(f"Set {i+1}: {candidate[:5]} + {candidate[5:]}")
