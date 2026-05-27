import os
import paddle
import numpy as np
import pandas as pd
import settings
from models import LottoModel
import utils

def predict_next_lotto():
    print("🔍 Loading model...")

    # 初始化模型结构
    model = LottoModel()

    # 构建参数文件路径
    param_path = os.path.join(settings.CHECKPOINTS_PATH, f'model_epoch_{settings.EPOCHS}.pdiparams')
    if not os.path.exists(param_path):
        print(f"❌ Model parameter file not found: {param_path}")
        return

    # 加载参数
    try:
        state_dict = paddle.load(param_path)
        model.set_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded from: {param_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 准备输入数据
    print("\n📄 Preparing input data...")
    try:
        df = pd.read_csv(settings.CLEAN_DATA_PATH, sep='\s+', header=None)
    except Exception as e:
        print(f"❌ Error reading input CSV: {e}")
        return

    # 检查列数是否符合预期
    expected_cols = settings.FRONT_SIZE + settings.BACK_SIZE + 1  # +1 是期号或序号列
    if df.shape[1] < expected_cols:
        print(f"❌ Error: Data contains insufficient columns, expected {expected_cols} columns.")
        return

    # 构建输入张量
    latest_data = df.iloc[-settings.SEQUENCE_LENGTH:, 1:].values
    input_vector = latest_data.flatten().astype('int64')
    input_tensor = paddle.to_tensor(input_vector).unsqueeze(0)
    print(f"✅ Input shape: {input_tensor.shape}")

    # 模型预测
    print("\n🎯 Predicting...")
    outputs = model(input_tensor)  # List of logits

    # 转为概率 & 输出最可能号码
    probabilities = [paddle.nn.functional.softmax(p, axis=1)[0] for p in outputs]
    numpy_probs = [p.numpy() for p in probabilities]
    most_likely = [int(p.argmax()) for p in probabilities]

    print(f"\n✅ Most likely prediction: {most_likely}")

    # 基于概率随机生成预测
    if hasattr(utils, 'select_seqs'):
        print(f"\n🎲 Generating {settings.PREDICT_NUM} random predictions:")
        for i in range(settings.PREDICT_NUM):
            random_pick = utils.select_seqs(numpy_probs)
            print(f"Set {i + 1}: {random_pick}")
    else:
        print("⚠️ 'select_seqs' not found in utils.py. Skipping random sampling.")

if __name__ == '__main__':
    required_settings = ['EPOCHS', 'CLEAN_DATA_PATH', 'SEQUENCE_LENGTH', 'FRONT_SIZE', 'BACK_SIZE', 'CHECKPOINTS_PATH']
    missing = [s for s in required_settings if not hasattr(settings, s)]
    if missing:
        print(f"❌ Missing settings: {missing}")
    else:
        predict_next_lotto()
