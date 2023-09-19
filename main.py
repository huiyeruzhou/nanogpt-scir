if __name__ == '__main__':
    from utils import load_tiny_shakespeare
    train_data, test_data, vocab = load_tiny_shakespeare()
    print("Train Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(train_data[:100])) + "...(more tokens, total: {})\n".format(len(train_data)))
    print("Test Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(test_data[:100])) + "...(more tokens, total: {})\n".format(len(test_data)))
    print("Vocab Size: {}".format(len(vocab)))
    print(vocab.token_to_idx)
