import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils import load_tiny_shakespeare
    train_data, test_data, vocab = load_tiny_shakespeare()
    print("Vocab Size: {}".format(len(vocab)))
    print(vocab.token_to_idx)
    print("Train Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(train_data[:100])) + "...(more tokens, total: {})\n".format(len(train_data)))
    print("Test Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(test_data[:100])) + "...(more tokens, total: {})\n".format(len(test_data)))
    from transformer_ref import TransformerDataset, Transformer
    from torch.utils.data import DataLoader
    # 加载数据
    embedding_dim = 128
    hidden_dim = 128
    batch_size = 2 * (256 + 128 + 64)
    block_size = 64
    num_epoch = 2
    n_layer = 3
    n_head = 4
    train_data, test_data, vocab = load_tiny_shakespeare()
    train_dataset = TransformerDataset(train_data, block_size)
    test_dataset = TransformerDataset(test_data, block_size)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    for batch in train_data_loader:
        print("The first sample, input:")
        print("".join(vocab.convert_ids_to_tokens(batch[0][0])))
        print("The first sample, target:")
        print("".join(vocab.convert_ids_to_tokens(batch[1][0])))
        break

    class CharDataset(Dataset):

      def __init__(self, data, block_size):
          chars = sorted(list(set(data)))
          data_size, vocab_size = len(data), len(chars)
          print('data has %d characters, %d unique.' % (data_size, vocab_size))

          self.stoi = {ch: i for i, ch in enumerate(chars)}
          self.itos = {i: ch for i, ch in enumerate(chars)}
          self.block_size = block_size
          self.vocab_size = vocab_size
          self.data = data

      def __len__(self):
          return len(self.data) - self.block_size

      def __getitem__(self, idx):
          # grab a chunk of (block_size + 1) characters from the data
          chunk = self.data[idx:idx + self.block_size + 1]
          # encode every character to an integer
          dix = [self.stoi[s] for s in chunk]
          x = torch.tensor(chunk[:-1], dtype=torch.long)
          y = torch.tensor(chunk[1:], dtype=torch.long)
          return x, y

    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open('dataset/train.txt', 'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, block_size)  # one line of poem is roughly 50 characters


    # 加载模型
    model = Transformer(len(vocab), embedding_dim, hidden_dim, block_size, num_head=n_head, num_layers=n_layer)
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
            checkpoint_path = "checkpoint/model_checkpoint-{}.pth".format(epoch)
            torch.save(model.state_dict(), checkpoint_path)
            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}.")
        print(f"Loss: {total_loss:.2f}")

    # 测试过程
    acc = 0
    losses = []
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        inputs, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            log_probs, loss = model(inputs)
            losses.append(loss)

    # 输出在测试集上的准确率
    print(f"Avarage loss: {np.mean(losses):.2f}")