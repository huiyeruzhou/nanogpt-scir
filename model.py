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
    def __init__(self, vocab_size, block_size, batch_size=2, n_embd=2, n_head=2, n_layer=2, attn_pdrop=0.1,
                 resid_pdrop=0.1, embd_pdrop=0.1, **kwargs):
        """

        :param vocab_size: 词表大小
        :param block_size: 最大序列长度, 即Transformer块的"大小"
        :param batch_size: 批次大小
        :param n_embd: 词向量维度
        :param n_head: 注意力头数
        :param n_layer: 注意力层数
        :param attn_pdrop: 注意力dropout概率
        :param resid_pdrop: 残差dropout概率
        :param embd_pdrop: 词向量dropout概率
        :param kwargs: 其他参数
        """
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_embd = n_embd
        self.batch_size = batch_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.block_size = block_size

        # 其他参数可以通过kwargs传入
        # 例如希望添加名为perceiver的字段: self.perceiver = True
        # 可以直接通过Config(...., perceiver=True)来设置
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 保存模型配置
        self.config = config

        # 保证n_embd可以被n_head整除
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # 将向量映射到q/k/v
        self.proj = nn.Linear(config.n_embd, config.n_embd * 3)

        # 注意力掩码: 不对当前字符之后的内容施加注意力, 避免模型看到未来的信息
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        # dropout可以有效防止过拟合
        # 抛弃一部分注意力分数
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # 抛弃一部分输出
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, n_embd

        # 获得batch中每个输入的q, k, v
        # x(batch_size, seq_len, n_embd) --proj--> (batch_size, seq_len, n_embd*3)
        # --chunk--> q,k,v(batch_size, seq_len, n_embd)
        q, k, v = self.proj(x).chunk(3, dim=-1)

        # 将q, k, v分解为n_head组, 每个head对应的向量维度为n_embd/n_head, 在第四维
        k = k.view(B, T, self.config.n_head, -1).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, -1).transpose(1, 2)

        # 计算自注意力分数
        # (B, n_head, T, hs) x (B, n_head, hs, T) -> (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

        # 应用掩码
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # 将注意力分数转化为注意力分布
        attn = F.softmax(attn, dim=-1)

        # 对注意力分布施加dropout
        attn = self.attn_drop(attn)
        # 注意力分布与v相乘, 得到注意力输出
        y = attn @ v

        # head组的输出拼接起来
        y = y.transpose(1, 2).reshape(B, T, C)

        # 对输出施加dropout
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    """
    两层全连接网络
    用于为Transformer的每个Block添加非线性表示能力
    """

    def __init__(self, config):
        super().__init__()
        # 隐层, 将向量映射到4倍的维度
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        # 激活
        self.gelu = nn.GELU()
        # 输出层, 将向量映射回原来的维度
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        # dropout
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.resid_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer的基本单元
    在每个子层的入口进行归一化和残差连接
    """

    def __init__(self, config):
        super().__init__()
        # 归一化
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 多头自注意力块
        self.attn = MultiHeadSelfAttention(config)
        # 归一化
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 前馈网络
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (batch_size, seq_len, n_embd)

        # self.attn(x) 对 x 应用多头自注意力
        # x + self.attn(x)的过程为残差连接
        # self.ln_1对残差连接的结果进行归一化
        x = self.ln_1(x + self.attn(x))

        # 应用前馈网络, 并进行残差连接和归一化
        x = self.ln_2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    """
    Transformer模型
    输入部分: 词嵌入 + 位置嵌入 + dropout
    编码部分: 由多个Block组成
    输出部分: 归一化 + 线性映射

    Transformer模型的结构如下:

       Logits of next token
               /\
            [Linear]
               ||
           [LayerNorm]
               ||
    |----------------------|
    |      Transformer     |
    |        Block         |
    |-----------------------
    |      Transformer     |
    |        Block         |
    |----------------------|
               /\
               ||
            [Dropout]
               ||
      Tok_emb   +   Pos_emb
         /\           /\
         ||           ||
     [Embedding]    [Linear]
         ||           ||
       tok_id       position
    """

    def __init__(self, config):
        super().__init__()
        # 配置信息
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

        # 获取位置嵌入, 并截取到与输入序列相同的长度
        # pos_emb(1, block_size, n_embd) --> position_embeddings(1, seq_len, n_embd)
        position_embeddings = self.pos_emb[:, :seq_len, :]

        # 二者相加作为输入, 通过广播操作将相同的同一个嵌入应用在这个batch的每一个sample上
        # position_embeddings(1, seq_len, n_embd) --broadcast--> (batch_size, seq_len, n_embd)
        x = token_embeddings + position_embeddings

        # 对输入进行dropout和归一化
        x = self.drop(x)
        x = self.ln_f(x)

        # 通过多个Transformer块进行编码
        for block in self.blocks:
            x = block(x)

        # 解码为对下一个token的回归预测
        # x(batch_size, seq_len, n_embd) --> logits(batch_size, seq_len, vocab_size)
        logits = self.head(x)

        # 如果有给定的目标输出, 则计算对数似然损失
        loss = None
        if y is not None:
            # 计算损失
            # x(batch_size, seq_len, vocab_size) --> x(batch_size*seq_len, vocab_size)
            # y(batch_size * seq_len)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss
