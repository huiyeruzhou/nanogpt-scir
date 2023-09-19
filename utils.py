# Defined in Section 4.6.4
# https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/utils.py

import torch
from vocab import Vocab

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



def load_tiny_shakespeare():
    from datasets import load_dataset
    from collections import Counter
    dataset = load_dataset("tiny_shakespeare")

    # Combine all the text into one long string
    train =  "".join(dataset["train"]["text"])
    test = "".join(dataset["test"]["text"])


    # Build the vocabulary
    vocab = Vocab.build(train + test, reserved_tokens=["<pad>"])

    # Split the data into train and test sets
    train_data = vocab.convert_tokens_to_ids(train)
    test_data = vocab.convert_tokens_to_ids(test)

    return train_data, test_data, vocab


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
