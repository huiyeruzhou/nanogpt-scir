import os

import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from mingpt_model import GPT
from model import Config, Transformer
from train import CharDataset
from utils import load_tiny_shakespeare
from vocab import read_vocab

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils import load_tiny_shakespeare

    train_data, test_data, vocab = load_tiny_shakespeare()
    print("Vocab Size: {}".format(len(vocab)))
    print(vocab.token_to_idx)
    print("Train Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(train_data[:100])) + "...(more tokens, total: {})\n".format(
        len(train_data)))
    print("Test Data Sample:")
    print(
        "".join(vocab.convert_ids_to_tokens(test_data[:100])) + "...(more tokens, total: {})\n".format(len(test_data)))
    from model import Transformer
    from torch.utils.data import DataLoader
    from train import CharDataset

    # 加载数据
    block_size = 64
    batch_size = 2 * (256 + 128 + 64)
    num_epoch = 2
    train_config = Config(vocab_size=len(vocab), block_size=block_size, batch_size=batch_size, n_embd=128, n_head=4,
                          n_layer=3, hidden_dim=128, num_epoch=num_epoch)
    train_data, test_data, vocab = load_tiny_shakespeare()
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    for batch in train_data_loader:
        print("The first sample, input:")
        print("".join(vocab.convert_ids_to_tokens(batch[0][0])))
        print("The first sample, target:")
        print("".join(vocab.convert_ids_to_tokens(batch[1][0])))
        break

    # # 加载模型
    # model = GPT(train_config)
    model = Transformer(train_config)
    model = nn.DataParallel(model)
    model.to(device)
    # model.to(device)  # 将模型加载到GPU中（如果已经正确安装）

    # 训练过程
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=6e-4)  # 使用Adam优化器

    model.train()

    for epoch in range(num_epoch):
        total_loss = 0
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for it, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            log_probs, loss = model(inputs, targets)
            loss.to(device)
            # loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 保存检查点
            os.makedirs("checkpoint", exist_ok=True)
            checkpoint_path = "checkpoint/model_checkpoint-{}.pth".format(epoch)
            torch.save(model.state_dict(), checkpoint_path)
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")
        print(f"Loss: {total_loss:.5f}")
    # 加载词表
    vocab = read_vocab('dataset/vocab.json')
    # 加载数据
    block_size = 64
    batch_size = 4 * 256
    num_epoch = 2
    train_config = Config(vocab_size=len(vocab), block_size=block_size, batch_size=batch_size, n_embd=128, n_head=4,
                          n_layer=3, hidden_dim=128, num_epoch=num_epoch)
    train_data, test_data, vocab = load_tiny_shakespeare()
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    # # 加载模型
    # model = Transformer(train_config).to('cuda')
    # # 从checkpoint的pth文件中加载模型
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('checkpoint/model_checkpoint-0.pth'))
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
        pbar.set_description(f"iter {it}: test loss {loss.item():.5f}")

    # 输出在测试集上的准确率
    print(f"Avarage loss: {np.mean(losses):.2f}")
