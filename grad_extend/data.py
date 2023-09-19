import numpy as np
import torch

from text import text_to_sequence
from grad.utils import fix_len_compatibility


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path):
        self.filelist = parse_filelist(filelist_path)
        # random.seed(random_seed)
        # random.shuffle(self.filelist)
        print(f'----------{len(self.filelist)}----------')

    def get_pair(self, filepath_and_text):
        spec = filepath_and_text[1]
        bert = filepath_and_text[2]
        text = filepath_and_text[3]
        text = self.get_text(text)
        bert = self.get_bert(bert)
        spec = self.get_mel(spec)
        assert text.shape[0] == bert.shape[0], filepath_and_text[0]
        return (text, bert, spec)

    def get_mel(self, filepath):
        mel = torch.load(filepath)
        return mel
    
    def get_bert(self, filepath):
        bert_embed = np.load(filepath)
        bert_embed = bert_embed.astype(np.float32)
        bert_embed = torch.FloatTensor(bert_embed)
        return bert_embed

    def get_text(self, text):
        text_norm = text_to_sequence(text)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, bert, mel = self.get_pair(self.filelist[index])
        item = {'y': mel, 'x': text, 'b': bert}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]
        n_berts = batch[0]['b'].shape[-1]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        b = torch.zeros((B, x_max_length, n_berts), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, b_, x_ = item['y'], item['b'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            b[i, :b_.shape[0], :] = b_
            x[i, :x_.shape[0]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'b': b, 'y': y, 'y_lengths': y_lengths}
