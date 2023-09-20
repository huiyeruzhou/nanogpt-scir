import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Config
from model import Transformer
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
    接收一个输入序列 x （形状为 (b, t)）并预测序列中的下一个标记，每次将预测结果反馈给模型。
    可选是否进行随机抽样, 并指定待抽样样本的个数
    用temperature配合随机抽样可以增加/减少随机性

    :param model: 训练好的模型
    :param x: 输入序列
    :param steps: 预测的序列长度
    :param temperature: 温度, 温度越高，抽样越随机
    :param sample: 是否进行随机抽样
    :param top_k: 抽样样本个数
    """
    block_size = model.config.block_size

    # 设置为评估模式, 停用dropout
    model.is_training = False

    # 生成符合目标长度的序列
    for k in range(steps):
        # 如果上文过长, 则截断到block_size
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        # 用模型进行预测
        logits, _ = model(x_cond)
        # 提取最后一步的回归结果并按温度缩放, 温度越高，抽样越随机'
        logits = logits[:, -1, :] / temperature

        # 可选地将样本裁剪为前 k 个概率最高的
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # 用 softmax 将回归值转换为概率
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
    # 判断是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载词表, 注意, 这里的词表必须是训练时使用的词表
    vocab = read_vocab('dataset/vocab.json')
    # 设置参数, 注意, 这里的参数必须和训练时的参数一致
    block_size = 96
    batch_size = 384
    num_epoch = 2
    train_config = Config(vocab_size=len(vocab), block_size=block_size, batch_size=batch_size, n_embd=384, n_head=6,
                          n_layer=4, hidden_dim=384, num_epoch=num_epoch)
    # 创建模型对象
    model = Transformer(train_config).to(device)
    # 从checkpoint的pth文件中加载模型参数
    model.load_state_dict(torch.load('checkpoint/model_checkpoint-0.pth'))

    # 将输入内容转换为token序列
    context = "Hamlet:\nTo be or not to be, this is a question.\n"
    x = torch.tensor([vocab.convert_tokens_to_ids(context)]).to(device)

    # 生成结果并转换回字符
    y = sample(model, x, 1000, temperature=1.0, sample=True, top_k=10)[0]
    print("".join(vocab.convert_ids_to_tokens(y)))
