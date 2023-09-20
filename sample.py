import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_ref import Transformer
from vocab import read_vocab


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    接收一个输入序列 x （形状为 (b, t)）并预测序列中的下一个标记，每次将预测结果反馈给模型。
    显然，这种抽样具有二次复杂度，不同于仅具有线性复杂度的循环神经网络（RNN），
    并且具有块大小（block_size）的有限上下文窗口，不同于具有无限上下文窗口的循环神经网络。
    """
    block_size = 64
    model.is_training = False
    for k in range(steps):
        # 如果需要，裁剪上下文
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        # 提取最后一步的 logits 并按温度缩放, 温度越高，抽样越随机
        logits, _ = model(x_cond)
        logits = logits[-1, : , :] / temperature

        # 可选地将概率裁剪为前 k 个选项
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # 用 softmax 转换为概率
        probs = F.softmax(logits, dim=-1)

        # 如果启用sample选项, 则根据prob进行抽样, 否则选择最大的概率
        # https://pytorch.org/docs/stable/generated/torch.multinomial.html
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # 将结果添加到序列并继续
        x = torch.cat((x, ix), dim=1)
    return x

if __name__ == "__main__":
    embedding_dim = 128
    hidden_dim = 128
    batch_size = 2 * (256 + 128 + 64)
    block_size = 64
    num_epoch = 2
    n_layer = 3
    n_head = 4

    # 加载词表
    vocab = read_vocab('dataset/vocab.json')
    # 加载模型
    model = Transformer(len(vocab), embedding_dim, hidden_dim, block_size, num_head=n_head, num_layers=n_layer).to('cuda')
    # 从checkpoint的pth文件中加载模型
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('checkpoint/model_checkpoint-0.pth'))

    # 获取起点
    begin = input("请输入开头单词：")
    # 转换为token序列
    context = "O God, O God!How can we live in such a world!"
    x = torch.tensor([vocab.convert_tokens_to_ids(context)]).to('cuda')
    # 生成序列
    y = sample(model, x, 64, temperature=1.0, sample=True, top_k=10)[0]
    print("".join(vocab.convert_ids_to_tokens(y)))