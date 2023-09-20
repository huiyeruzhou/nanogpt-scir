import torch
from torch.utils.data import Dataset


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