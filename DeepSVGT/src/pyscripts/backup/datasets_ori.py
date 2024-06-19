import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader

random.seed(0)

class Seq2Vec:
    basic_bases = 'ATCG'
    bases = basic_bases + '-N'
    len_bases = len(bases)

    base_indices = dict((b, i) for i, b in enumerate(bases))
    indices_base = dict((i, b) for i, b in enumerate(bases))

    def __init__(self):
        pass

    @staticmethod
    def encode(seq_):
        seq = seq_.upper()
        vec = np.zeros((len(seq), Seq2Vec.len_bases)) # len_seq * over 4

        for i, b in enumerate(seq):
            vec[i, Seq2Vec.base_indices.get(b, Seq2Vec.len_bases-1)] = 1
        return vec

    @staticmethod
    def decode(vec_):
        vec = vec_.argmax(axis=-1)
        return ''.join(Seq2Vec.indices_base.get(v, Seq2Vec.len_bases-1) for v in vec)

    @staticmethod
    def get_sim_vec(len_=200):
        rseq = [random.choice(Seq2Vec.basic_bases) for _ in range(len_)]
        return ''.join(rseq) 
        # seq = ''.join(rseq)
        # return Seq2Vec.encode(seq), seq

    @staticmethod
    def get_sim_error(seq, pi=0.00, pd=0.00, ps=0.33):
        """
        Given an input sequence `seq`, generating another
        sequence with errors. 
        pi: insertion error rate
        pd: deletion error rate
        ps: substitution error rate
        """
        out_seq = []
        for c in seq:
            while 1:
                r = random.uniform(0,1)
                if r < pi:
                    out_seq.append(random.choice(Seq2Vec.basic_bases))
                else:
                    break
            r -= pi
            if r < pd:
                continue
            r -= pd
            if r < ps:
                out_seq.append(random.choice(Seq2Vec.basic_bases))
                continue
            out_seq.append(c)
        
        # return ''.join(out_seq)
        s = ''.join(out_seq)
        return Seq2Vec.encode(s)


    @staticmethod
    def get_bulk(count=250):
        seq1 = Seq2Vec.get_sim_vec()
        seq2 = Seq2Vec.get_sim_vec()

        # return [Seq2Vec.get_sim_error(seq) for _ in range(count)]

        s = []
        for  _ in range(count):
            s.append(Seq2Vec.get_sim_error(seq1))
            s.append(Seq2Vec.get_sim_error(seq2))

        return s

class SeqDatasetOHE(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    def __init__(self, seqs):
        self.seqs = seqs
        self.seq_len = len(self.seqs)
        
        # one-hot encode sequences, then stack in a torch tensor
        self.ohe_seqs = torch.stack([torch.tensor(x) for x in self.seqs])
        # print(self.ohe_seqs.shape)
    
        # +------------------+
        # | Get the Y labels |
        # +------------------+
        self.labels = torch.tensor([1 for _ in self.seqs]).unsqueeze(1)
        # print(self.labels.shape)
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self,idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]
        # print(seq.shape)
        # return seq, label
        return torch.tensor(seq, dtype=torch.float32) , label


def build_dataloaders(batch_size=100,
                      shuffle=True):
    seqs = Seq2Vec.get_bulk()

    # create Datasets    
    train_ds = SeqDatasetOHE(seqs)
    test_ds = SeqDatasetOHE(seqs)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    return train_dl, test_dl


if __name__ == '__main__':
    v, s = Seq2Vec.get_sim_vec(20)

    print(s)
    s1 =Seq2Vec.get_sim_error(s)
    print(s1)
    # print(Seq2Vec.get_sim_error(s))
    # print(Seq2Vec.diff(s, s1))

    alignment, score, start_end_positions = global_pairwise_align_nucleotide(DNA(s), DNA(s1))
    print(alignment)
    print(score)