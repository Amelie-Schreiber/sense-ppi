from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


class PairSequenceData(Dataset):
    def __init__(self,
                 actions_file,
                 emb_dir,
                 max_len,
                 pad_inputs=True,
                 labels=True):

        super(PairSequenceData, self).__init__()
        self.max_len = max_len
        self.pad_inputs = pad_inputs
        self.emb_dir = emb_dir
        self.action_path = actions_file
        self.labels = labels

        dtypes = {'seq1': str, 'seq2': str}
        if self.labels:
            dtypes.update({'label': np.float16})
            self.actions = pd.read_csv(self.action_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=dtypes)
        else:
            self.actions = pd.read_csv(self.action_path, delimiter='\t', usecols=[0, 1], names=["seq1", "seq2"], dtype=dtypes)

    def get_emb(self, emb_id):
        f = os.path.join(self.emb_dir, '{}.pt'.format(emb_id))

        try:
            emb = torch.load(f)
        except FileNotFoundError as _:
            raise Exception('Embedding file {} not found. Check your fasta file and make sure it contains '
                            'all the sequences used in training/testing.'.format(f))

        tensor_emb = emb['representations'][36]  # [33]
        tensor_len = tensor_emb.size(0)
        if self.pad_inputs:
            if tensor_emb.shape[0] > self.max_len:
                tensor_emb = tensor_emb[:self.max_len]
                tensor_len = self.max_len
            if tensor_emb.shape[0] < self.max_len:
                tensor_emb = F.pad(tensor_emb, (0, 0, 0, self.max_len - tensor_emb.size(0)), "constant", 0)

        return tensor_emb, tensor_len

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        id1 = self.actions["seq1"][idx]
        id2 = self.actions["seq2"][idx]

        if self.labels:
            label = int(self.actions["label"][idx])
        else:
            label = 0

        emb1, len1 = self.get_emb(id1)
        emb2, len2 = self.get_emb(id2)

        return {"emb1": emb1,
                "len1": len1,
                "emb2": emb2,
                "len2": len2,
                "label": label,
                "prot1": id1,
                "prot2": id2}


if __name__ == '__main__':
    pass
