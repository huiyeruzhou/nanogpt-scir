# Defined in Section 4.6.1
# # length of dataset in characters:  1115394
# # all the unique characters:
# #  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# # vocab size: 65
# # train has 1003854 tokens
# # val has 111540 tokens
# https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/vocab.py
from collections import defaultdict


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            # if "<unk>" not in tokens:
            #     tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            # self.unk = self.token_to_idx['<unk>']
            self.unk = None

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        """
        工厂方法, 从文本构建词表

        :param text: 文本
        :param min_freq: 最小词频
        :param reserved_tokens: 保留词
        :return: 一个词表对象
        """

        # 统计词频
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        # 加入unk和保留词, 并根据词频过滤部分单词
        # uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        # uniq_tokens += [token for token, freq in token_freqs.items() \
        #                 if freq >= min_freq and token != "<unk>"]

        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        """
        将词转换为id

        :param tokens: 可迭代, 每次返回一个token
        :return: ids : list[int], 每个token对应的id
        """
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]

import json
def save_vocab(vocab, path):
    """
    保存词表为json格式
    :param vocab: 词表
    :param path: 保存路径
    """
    with open(path, 'w') as writer:
        writer.write(json.dumps(vocab.idx_to_token, indent=4))



def read_vocab(path):
    with open(path, 'r') as f:
        json_str = f.read()
    tokens = json.loads(json_str)
    return Vocab(tokens)