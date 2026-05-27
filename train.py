# 文件：train.py
import os
import paddle
import paddle.nn as nn
import pandas as pd
import numpy as np
from models import model
from dataset import LottoDataset
import settings

# --- 自定义损失函数 ---
class TotalLoss(nn.Layer):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, *args):
        num_outputs = settings.FRONT_SIZE + settings.BACK_SIZE
        preds, labels = args[:num_outputs], args[num_outputs:]
        total_loss = sum(self.cross_entropy_loss(p, l) for p, l in zip(preds, labels))
        return total_loss

# --- 主训练流程 ---
if __name__ == "__main__":
    print("🚀 Starting Training")
    print("📦 Loading dataset...")
    train_dataset = LottoDataset(train=True)
    test_dataset = LottoDataset(train=False)
    print("✅ Dataset loaded.")

    # 构建 Paddle 模型
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
    pad_model = paddle.Model(model)
    pad_model.prepare(optimizer=optimizer, loss=TotalLoss())

    # 检查模型保存路径
    if not os.path.exists(settings.CHECKPOINTS_PATH):
        os.makedirs(settings.CHECKPOINTS_PATH)

    for epoch in range(1, settings.EPOCHS + 1):
        print(f"\n📚 Epoch {epoch}/{settings.EPOCHS}")
        pad_model.fit(train_dataset, batch_size=settings.BATCH_SIZE, epochs=1, verbose=1)

        # ✅ 保存动态图模型参数（用于 predict.py 的加载方式）
        save_path = os.path.join(settings.CHECKPOINTS_PATH, f'model_epoch_{epoch}.pdiparams')
        paddle.save(model.state_dict(), save_path)
        print(f"✅ Model parameters saved to: {save_path}")

        # 测试集评估
        print(f"🧪 Evaluating on test set (epoch {epoch})...")
        eval_result = pad_model.evaluate(test_dataset, batch_size=settings.BATCH_SIZE, verbose=1)
        print(f"📊 Evaluation result: {eval_result}")

    print("\n🏁 --- Training Finished ---")
    print("👉 Use `predict.py` to make predictions.")
