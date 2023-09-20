import os
from vocab import Vocab, save_vocab, read_vocab


def load_tiny_shakespeare(use_cache=True):
    """
    加载tiny_shakespeare数据集

    :param use_cache: 是否使用缓存
    :return: train_data, 训练集的token_id序列
    :return: test_data, 测试集的token_id序列
    :return: vocab 一个Vocab对象, 包含训练集和测试集中出现的所有字符
    """
    # 检查是否已经缓存到文件
    if use_cache and os.path.exists("dataset/train.txt") \
            and os.path.exists("dataset/test.txt") and os.path.exists("dataset/vocab.json"):

        # 从文件中读取
        with open("dataset/train.txt", "r") as f:
            train = f.read()
        with open("dataset/test.txt", "r") as f:
            test = f.read()
        vocab = read_vocab("dataset/vocab.json")
    else:
        # 若没有缓存， 则从datasets库加载并缓存
        from datasets import load_dataset
        dataset = load_dataset("tiny_shakespeare")

        # dataset包含train、test、validation三个子集，每个子集包含text和label两个字段
        train = "".join(dataset["train"]["text"])
        test = "".join(dataset["test"]["text"])

        vocab = Vocab.build(sorted(list(set(train + test))), reserved_tokens=["<pad>"])

        # 输出为文件, 使用系统无关路径
        os.makedirs("dataset", exist_ok=True)
        with open("dataset/train.txt", "w") as f:
            f.write(train)
        with open("dataset/test.txt", "w") as f:
            f.write(test)
        save_vocab(vocab, "dataset/vocab.json")

    # 将文本转换为id序列
    train_data = vocab.convert_tokens_to_ids(train)
    test_data = vocab.convert_tokens_to_ids(test)

    return train_data, test_data, vocab

# 测试集相关信息:
# # length of dataset in characters:  1115394
# # all the unique characters:
# #  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# # vocab size: 67(含<unk>和<pad>)
# # train has 1003854 tokens
# # val has 111540 tokens
