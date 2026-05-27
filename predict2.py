# 文件: predict.py
import os
import paddle
import pandas as pd
import numpy as np
import settings
from models import LottoModel
import utils

def predict_next_lotto():
    print("🔍 Loading model...")

    model = LottoModel()
    param_path = os.path.join(settings.CHECKPOINTS_PATH, f'model_epoch_{settings.EPOCHS}.pdparams')
    if not os.path.exists(param_path):
        print(f"❌ Model parameter file not found: {param_path}")
        return

    try:
        state_dict = paddle.load(param_path)
        model.set_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded from: {param_path}")
    except Exception as e:
        print(f"❌ Failed to load model parameters: {e}")
        return

    print("\n📄 Preparing input data...")
    try:
        df = pd.read_csv(settings.CLEAN_DATA_PATH, sep='\s+', header=None)
    except Exception as e:
        print(f"❌ Error reading input CSV: {e}")
        return

    expected_cols = settings.FRONT_SIZE + settings.BACK_SIZE + 1
    if df.shape[1] < expected_cols:
        print(f"❌ Error: Data contains insufficient columns, expected {expected_cols} columns.")
        return

    latest_data = df.iloc[-settings.SEQUENCE_LENGTH:, 1:].values
    input_vector = latest_data.flatten().astype('int64')
    input_tensor = paddle.to_tensor(input_vector).unsqueeze(0)
    print(f"✅ Input shape: {input_tensor.shape}")

    print("\n🎯 Predicting...")
    predictions = model(input_tensor)
    probabilities = [paddle.nn.functional.softmax(p, axis=1)[0].numpy() for p in predictions]
    most_likely = [int(p.argmax()) for p in predictions]

    print(f"\n✅ Most likely prediction: {most_likely}")

    # --------- Intelligent Sampling ----------
    print(f"\n🎲 Generating {settings.PREDICT_NUM} smart predictions:")

    for i in range(settings.PREDICT_NUM):
        pick = []
        used = set()
        for idx, prob in enumerate(probabilities):
            # 选择不重复的号码
            top_indices = np.argsort(prob)[::-1]  # 从大到小排序
            for num in top_indices:
                if num not in used and num != 0:
                    pick.append(num)
                    used.add(num)
                    break
        print(f"Set {i+1}: {pick}")

if __name__ == '__main__':
    required_settings = ['EPOCHS', 'CLEAN_DATA_PATH', 'SEQUENCE_LENGTH', 'FRONT_SIZE', 'BACK_SIZE', 'CHECKPOINTS_PATH', 'PREDICT_NUM']
    missing = [s for s in required_settings if not hasattr(settings, s)]
    if missing:
        print(f"❌ Missing settings: {missing}")
    else:
        predict_next_lotto()
