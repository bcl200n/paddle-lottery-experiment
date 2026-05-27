# 文件: /home/aistudio/lotto_paddle1/dataset.py (适配单输入)
import paddle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import settings
import os

class LottoDataset(paddle.io.Dataset):
    def __init__(self, train=True, train_data_rate=0.9):
        super(LottoDataset, self).__init__()
        df = pd.read_csv(ABSOLUTE_DATA_PATH, sep='\s+', header=None)
        all_x, all_y = [], []
        for i in range(len(df) - settings.SEQUENCE_LENGTH):
            x = df.iloc[i:i + settings.SEQUENCE_LENGTH, 1:].values.flatten()
            y = df.iloc[i + settings.SEQUENCE_LENGTH, 1:].values.flatten()
            all_x.append(x)
            all_y.append(y)
        all_x_np = np.array(all_x, dtype='int64')
        all_y_np = np.array(all_y, dtype='int64')
        train_x, test_x, train_y, test_y = train_test_split(all_x_np, all_y_np, test_size=1-train_data_rate, random_state=42)
        self.x = train_x if train else test_x
        self.y = train_y if train else test_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 返回一个 Tensor 和一个 label 元组
        return self.x[idx], tuple(self.y[idx, i] for i in range(settings.FRONT_SIZE + settings.BACK_SIZE))

# 静态地获取路径，避免在__init__中重复
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABSOLUTE_DATA_PATH = os.path.join(BASE_DIR, settings.CLEAN_DATA_PATH)