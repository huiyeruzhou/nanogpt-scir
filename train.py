import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, Config
from utils import load_tiny_shakespeare


class CharDataset(Dataset):
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

if __name__ == '__main__':
    # 加载数据
    train_data, test_data, vocab = load_tiny_shakespeare()

    # 设置参数
    block_size = 64
    batch_size = 768
    num_epoch = 2
    train_config = Config(vocab_size=len(vocab), block_size=block_size, batch_size=batch_size, n_embd=128, n_head=4,
                          n_layer=3, hidden_dim=128, num_epoch=num_epoch)

    # 创建数据集, 注意, 由于Transformer限制输入序列长度, 数据集每次输出的序列长度由block_size决定(与之相等)
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)
    # 创建DataLoader, 指定放置于GPU上(如有), 用4个进程加载数据
    # 训练集样本每次随机打乱, 测试集样本不打乱
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)


    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(train_config)
    model = nn.DataParallel(model)
    model.to(device)

    # 将模型设置为训练模式, 这会启用Dropout等正则化操作
    model.train()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=6e-4)
    # 在训练集上训练指定世代
    for epoch in range(num_epoch):
        total_loss = 0
        best_loss = 1e9
        # 使用tqdm显示进度条
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for it, (inputs, targets) in pbar:
            # 将输入和目标放置于GPU上(如有)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            log_probs, loss = model(inputs, targets)

            # 优化器梯度清零, 反向传播并更新参数
            loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()

            # 更新进度条, 显示当前训练进度和损失
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")

        # 保存检查点
        if (loss.item() < best_loss):
            best_loss = loss.item()
            os.makedirs("checkpoint", exist_ok=True)
            checkpoint_path = "checkpoint/model_checkpoint-{}.pth".format(epoch)
            torch.save(model.state_dict(), checkpoint_path)
        # 总结本世代的训练
        print(f"Loss: {total_loss:.5f}, Best loss: {best_loss:.5f}")

    # 测试过程
    acc = 0
    losses = []
    pbar  = tqdm(enumerate(test_data_loader), total=len(test_data_loader))
    for batch in pbar:
        it, (inputs, targets) = batch
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        with torch.no_grad():
            log_probs, loss = model(inputs, targets)
            losses.append(loss.item())
        pbar.set_description(f"iter {it}: test loss: {loss.item():.5f}")

    # 输出在测试集上的准确率
    print(f"Avarage loss: {np.mean(losses):.2f}")