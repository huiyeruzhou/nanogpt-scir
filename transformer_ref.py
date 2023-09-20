# Defined in Section 4.6.8
# https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/transformer_sent_polarity.py
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


# tqdm是一个Pyth模块，能以进度条的方式显式迭代的进度


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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_len, num_head, num_layers,
                 dim_feedforward=512, dropout=0.1, activation: str = "gelu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.drop = nn.Dropout(0)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(max_len,1, embedding_dim))
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层 : 将隐藏层的输出映射到词表
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)


    def forward(self, inputs, targets=None):
        # inputs(b,t) -> (t,b)
        inputs = torch.transpose(inputs, 0, 1)
        # hidden(t, b, e)
        hidden_states = self.embeddings(inputs)
        # hidden_states = self.position_embedding(hidden_states)
        # attention_mask = length_to_mask(lengths) == False
        # position(t, 1, e)
        position_embeddings = self.pos_emb[:hidden_states.shape[0], :, :] # each position maps to a (learnable) vector
        
        hidden_states =  self.drop(hidden_states + position_embeddings)
        hidden_states = self.transformer(hidden_states)
        # hidden_states = hidden_states[0, :, :]
        hidden_states = self.ln_f(hidden_states)
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
        return log_probs, loss




