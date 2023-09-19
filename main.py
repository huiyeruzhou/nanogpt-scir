if __name__ == '__main__':
    from utils import load_tiny_shakespeare
    train_data, test_data, vocab = load_tiny_shakespeare()
    print("Vocab Size: {}".format(len(vocab)))
    print(vocab.token_to_idx)
    print("Train Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(train_data)) + "...(more tokens, total: {})\n".format(len(train_data)))
    print("Test Data Sample:")
    print("".join(vocab.convert_ids_to_tokens(test_data)) + "...(more tokens, total: {})\n".format(len(test_data)))
    from transformer_ref import TransformerDataset
    from torch.utils.data import DataLoader
    from utils import collate_fn
    train_data, test_data, vocab = load_tiny_shakespeare()
    batch_size = 32
    block_size = 32
    train_dataset = TransformerDataset(train_data, block_size=block_size)
    test_dataset = TransformerDataset(test_data, block_size=block_size)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for batch in train_data_loader:
        print("The first sample, input:")
        print("".join(vocab.convert_ids_to_tokens(batch[0][0])))
        print("The first sample, target:")
        print("".join(vocab.convert_ids_to_tokens(batch[1][0])))
        break
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)