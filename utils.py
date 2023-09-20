# Defined in Section 4.6.4
# https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/utils.py
import os

import torch
from vocab import Vocab, save_vocab ,read_vocab
from torch.nn.utils.rnn import pad_sequence




def load_sentence_polarity():
    from nltk.corpus import sentence_polarity

    vocab = Vocab.build(sentence_polarity.sents())

    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data, test_data, vocab



def load_tiny_shakespeare(use_cache=True):
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

def collate_fn(examples):
    print(examples)
    # lengths = torch.tensor([len(ex[0]) for ex in examples])
    # inputs = [torch.tensor(ex[0]) for ex in examples]
    # targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # # 对batch内的样本进行padding，使其具有相同长度
    # inputs = pad_sequence(inputs, batch_first=True)
    return [x for x, y in examples], [y for x, y in examples]


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask

def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab
