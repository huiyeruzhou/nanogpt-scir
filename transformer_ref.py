# Defined in Section 4.6.8
# https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/transformer_sent_polarity.py
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import defaultdict
from vocab import Vocab
from utils import load_sentence_polarity, length_to_mask, load_tiny_shakespeare

# tqdm是一个Pyth模块，能以进度条的方式显式迭代的进度
from tqdm.auto import tqdm


class TransformerDataset(Dataset):
    """
    用于加载Transformer的数据集
    数据集内部是字符token对应的id
    每次取样时返回一个block_size长度的序列
    目标是预测下一个字符
    """
    def __init__(self, data, block_size):
        """
        初始化一个数据集

        :param data: 数据集中的内容
        :param block_size: 每个样本的长度
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # 虽然数据长度为len(self.data), 但是每个样本的长度为block_size + 1
        # 因此可取的样本数只有len(self.data) - self.block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # 获取一个长为block_size + 1的序列
        chunk = self.data[idx:idx + self.block_size + 1]
        # 输入序列为前block_size个字符
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # 目标序列, 每一个字符都是输入序列中对应位置的下一个字符
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)


    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

embedding_dim = 128
hidden_dim = 128
num_class = 2
batch_size = 32
num_epoch = 5

# 加载数据
train_data, test_data, vocab = load_tiny_shakespeare()
train_dataset = TransformerDataset(train_data)
test_dataset = TransformerDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device) # 将模型加载到GPU中（如果已经正确安装）

# 训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

# 测试过程
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=1) == targets).sum().item()

# 输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")