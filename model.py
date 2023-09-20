from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
hi，最近有空帮我写个代码吧（用于教学），
分别用LSTM以及Transformer实现基于字符的语言模型，
包括模型训练和用训练号的模型生成文本（实现最基本的功能就好，不用支持多卡训练，代码增加必要的注释）。
数据直接加载HuggingFace的datasets（https://huggingface.co/datasets/tiny_shakespeare）。
Transformer的代码可以参考：https://github.com/karpathy/nanoGPT ，
其中的模型文件用我发给你的（需要进行进一步修改）。
"""


@dataclass
class Config:
    def __init__(self, vocab_size, block_size,
                 batch_size=2, embedding_dim=2, n_head=2, n_layer=2,
                 **kwargs):
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.n_embd = embedding_dim
        self.batch_size = batch_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.proj = nn.Linear(config.n_embd, config.n_embd * 3)

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, n_embd

        # 获得batch中每个输入的q, k, v，并将q, k, v分解为n_head组
        q, k, v = self.proj(x).chunk(3, dim=-1)
        k = k.view(B, T, self.config.n_head, -1).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, -1).transpose(1, 2)

        # 计算自注意力：
        # (B, n_head, T, hs) x (B, n_head, hs, T) -> (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 词嵌入: 将输入的id映射为词向量
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入: 将输入的位置映射为位置向量
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # dropout: 用于防止过拟合
        self.drop = nn.Dropout(config.embd_pdrop)
        # 层归一化: 对输入进行归一化(块间和块输出已经进行了归一化)
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 编码层: 由多个Transformer块组成
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # 解码层: 将输出的词向量映射为词id
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, y=None):
        # 要求输入序列长度不能大于块大小
        _, seq_len = x.size()
        assert seq_len <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # 获取词嵌入
        # x(batch_size, seq_len) --> token_embeddings: (batch_size, seq_len, n_embd)
        token_embeddings = self.tok_emb(x)

        # 获取位置嵌入
        # pos_emb(1, block_size, n_embd) --> position_embeddings(1, seq_len, n_embd)
        position_embeddings = self.pos_emb[:, :seq_len, :]

        # 二者相加作为输入, 通过广播操作将相同的位置嵌入应用在同一batch的每一个输入上
        # position_embeddings(1, seq_len, n_embd) --broadcast--> (batch_size, seq_len, n_embd)
        x = token_embeddings + position_embeddings

        # 对输入进行dropout和归一化
        x = self.drop(x)
        x = self.ln_f(x)

        # 通过多个Transformer块
        for block in self.blocks:
            x = block(x)

        # 解码
        # x(batch_size, seq_len, n_embd) --> logits(batch_size, seq_len, vocab_size)
        logits = self.head(x)

        # 如果有给定的输出, 则计算损失
        loss = None
        if y is not None:
            # 计算损失
            # x(batch_size, seq_len, vocab_size) --> x(batch_size*seq_len, vocab_size)
            # y(batch_size * seq_len)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss


