# 文件: /home/aistudio/lotto_paddle1/models.py (最终模型)
import paddle
import paddle.nn as nn
import settings

class LottoModel(nn.Layer):
    def __init__(self):
        super(LottoModel, self).__init__()
        # Keras: Input -> Embedding -> LSTM -> Dense
        # 我们模拟这个流程
        
        # 输入是一个长向量，里面是球的ID。我们先对这些ID做嵌入
        self.embedding = nn.Embedding(num_embeddings=settings.MAX_NUM + 1, embedding_dim=settings.EMBEDDING_DIM)
        
        # LSTM的输入是 (N, L, H_in), L是序列长度, H_in是输入特征维度
        # 我们的输入是 (batch_size, SEQUENCE_LENGTH * 7), 经过 embedding 后是 (batch_size, SEQUENCE_LENGTH * 7, EMBEDDING_DIM)
        # 这就是LSTM的输入
        self.lstm = nn.LSTM(input_size=settings.EMBEDDING_DIM, hidden_size=settings.HIDDEN_SIZE)
        
        # 输出层
        self.outputs = nn.LayerList()
        # 我们有 7 个输出
        for _ in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            self.outputs.append(nn.Linear(in_features=settings.HIDDEN_SIZE, out_features=settings.MAX_NUM + 1))
            
    def forward(self, x): # <--- 输入从字典变成了一个 Tensor
        # x 的 shape: (batch_size, SEQUENCE_LENGTH * 7)
        
        # 1. 嵌入
        # embedded_x shape: (batch_size, SEQUENCE_LENGTH * 7, EMBEDDING_DIM)
        embedded_x = self.embedding(x)
        
        # 2. LSTM
        # lstm_output shape: (batch_size, SEQUENCE_LENGTH * 7, HIDDEN_SIZE)
        lstm_output, (last_hidden, last_cell) = self.lstm(embedded_x)
        
        # 3. 取最后一个时间步的输出
        last_step_output = lstm_output[:, -1, :]
        
        # 4. 预测
        final_outputs = []
        for output_layer in self.outputs:
            final_outputs.append(output_layer(last_step_output))
            
        return final_outputs

model = LottoModel()