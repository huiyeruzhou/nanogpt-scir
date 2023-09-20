# Defined in Section 4.6.8
# https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/transformer_sent_polarity.py
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


# tqdm是一个Pyth模块，能以进度条的方式显式迭代的进度

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
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layers, dropout=0.1,
                 activation: str = "gelu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = n_embd
        self.drop = nn.Dropout(0)
        self.embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(block_size, 1, n_embd))
        self.position_embedding = PositionalEncoding(n_embd, dropout, block_size)
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(n_embd, n_head, 4 * n_embd, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        # 输出层 : 将隐藏层的输出映射到词表
        self.ln_f = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size)


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




