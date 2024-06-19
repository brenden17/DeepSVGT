import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


BASIC_BASES = 'ATCG'
DEL_BASE = '-'
INS_BASE = '+'
MISMATCHED_BASE = '|'
EXTEND_BASES = BASIC_BASES + DEL_BASE + INS_BASE + MISMATCHED_BASE
BASE_SIZE = len(BASIC_BASES)
EXTEND_BASE_SIZE = len(EXTEND_BASES)

BASE_INDICES = dict((b, i) for i, b in enumerate(EXTEND_BASES))
# BASE_VALUE_INDICES = dict((b, i/5) for i, b in enumerate(BASES))
BASE_VALUE_INDICES = dict((b, i) for i, b in enumerate(EXTEND_BASES))
INDICES_BASE = dict((i, b) for i, b in enumerate(EXTEND_BASES))


def encode(seq):
    #TODO, try exception
    # seq = seq.upper()
    # print((len(seq), BASE_SIZE))
    vec = np.zeros((len(seq), EXTEND_BASE_SIZE)) # len_seq * 6

    for i, b in enumerate(seq):
        vec[i, BASE_INDICES.get(b, EXTEND_BASE_SIZE)] = BASE_VALUE_INDICES.get(b, EXTEND_BASE_SIZE)
        # vec[i, BASE_INDICES.get(b, EXTEND_BASE_SIZE)] = 1

    return vec


def decode(invec):
    #TODO, try exception
    vec = invec.argmax(axis=-1)
    return ''.join(INDICES_BASE.get(v, EXTEND_BASE_SIZE) for v in vec)


def kmer(seq, k_size=50, step=1, padding=True, encoding=True):
    seq_size = len(seq)

    num_kmers = seq_size - k_size + 1

    empty_size = seq_size - k_size

    # print(f'k_size:{k_size}, num_kmers:{num_kmers}, empty_size:{empty_size}')
    # print(f'num_kmers:{num_kmers}')

    l = []

    for i in range(num_kmers):
        s = 'N' * i + seq[i:i+k_size] + 'N' * (empty_size-i) if padding else seq[i:i+k_size]
        s = encode(s) if encoding else s
        l.append(s)
    return l


class SeqDataset2(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    def __init__(self, seqs, kmer_size=512):
        self.seqs = seqs
        self.seq_len = len(self.seqs)

        # one-hot encode sequences, then stack in a torch tensor
        # self.encoded_seqs = torch.stack([torch.tensor(kmer(seq, kmer_size)) for seq in self.seqs])
        # print(f'{self.encoded_seqs.shape[0]} is encoded.l')

        self.encoded_seqs = []
        for seq in seqs:
            self.encoded_seqs.extend(torch.tensor(kmer(seq, kmer_size)))

        print('len(self.encoded_seqs)')
        print(len(self.encoded_seqs))
        self.encoded_seqs = torch.stack(self.encoded_seqs)

        # print(f'{len(self.encoded_seqs)} is encoded.')

        self.labels = torch.tensor([1 for _ in self.seqs]).unsqueeze(1)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self,idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.encoded_seqs[idx]
        label = self.labels[idx]
        # print(seq.shape)
        # return seq, label
        return torch.tensor(seq, dtype=torch.float32) , label
        # return torch.tensor(seq) , label


class SeqDataset(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    def __init__(self, seqs):
        self.seqs = seqs
        
        # one-hot encode sequences, then stack in a torch tensor
        # self.encoded_seqs = torch.stack([torch.tensor(kmer(seq, kmer_size)) for seq in self.seqs])
        # print(f'{self.encoded_seqs.shape[0]} is encoded.l')

        self.encoded_seqs = [encode(seq) for seq in seqs]
        self.labels = [encode(seq) for seq in seqs]

        #for seq in seqs:
        #    if seq:
        #        self.encoded_seqs.extend(torch.tensor(kmer(seq, kmer_size)))

        # for seq in seqs:
        #     #self.encoded_seqs.extend(torch.tensor(encode(seq)))
        #     self.encoded_seqs.append(torch.tensor(encode(seq)))

        # print('len(self.encoded_seqs)')
        # print(len(self.encoded_seqs))
        # self.encoded_seqs = torch.stack(self.encoded_seqs)

        # print(f'{len(self.encoded_seqs)} is encoded.')

        # self.labels = torch.tensor([1 for _ in self.seqs]).unsqueeze(1)
        
    def __len__(self):
        return len(self.encoded_seqs)
    
    def __getitem__(self, idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        x = torch.FloatTensor(self.encoded_seqs[idx])
        y = self.labels[idx]
        return x, y
        # print(seq.shape)
        seq = self.encoded_seqs[idx]
        # return seq, label
        return torch.tensor(seq, dtype=torch.float32) , label
        # return torch.tensor(seq) , label


def build_simdataloader(seq_size=200, group_subs_rate=0.4, subs_rate=0.2, batch_size=100, kmer=0, shuffle=True):
    # seqs = Seq2Vec.get_bulk()
    # seqs = generate_two_groups(seq_size, group_subs_rate, subs_rate, kmer)
    seqs = generate_three_groups(seq_size, group_subs_rate, subs_rate, kmer)

    # create Datasets
    train_ds = SeqDataset(seqs)
    test_ds = SeqDataset(seqs)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl


def build_dataloader(seqs, batch_size=100, shuffle=True):
    # create Datasets
    train_ds = SeqDataset(seqs)
    test_ds = SeqDataset(seqs)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl


def build_dataloader_with_kmer(seqs, kmer_size, batch_size=100, shuffle=True):
    # create Datasets
    train_ds = SeqDataset(seqs, kmer_size)
    test_ds = SeqDataset(seqs, kmer_size)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl


if __name__ == '__main__':
    # generate_two_groups()
    print('abc',encoding=False)
    test_kmer()
