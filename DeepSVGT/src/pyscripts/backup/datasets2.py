import random
import math

import numpy as np

import torch
# from torch.utils.data import Dataset, DataLoader

random.seed(1)


BASIC_BASES = 'ATCG'
# BASES = BASIC_BASES + '-N'
BASES = BASIC_BASES 
BASE_SIZE = len(BASES)

BASE_INDICES = dict((b, i) for i, b in enumerate(BASES))
INDICES_BASE = dict((i, b) for i, b in enumerate(BASES))


# def generate_dna(len=200, basic_base=True):
#     bases = BASIC_BASES if basic_base else BASES
def generate_seq(seq_size=150):
    seq = [random.choice(BASES) for _ in range(seq_size)]
    return ''.join(seq)

def encode(seq):
    #TODO, try exception
    # seq = seq.upper()
    print(seq)
    print(len(seq))
    vec = np.zeros((len(seq), BASE_SIZE)) # len_seq * 6
    print(vec.shape)
    for i, b in enumerate(seq):
        vec[i, BASE_INDICES.get(b, BASE_SIZE)] = 1

    return vec

def decode(invec):
    #TODO, try exception
    vec = invec.argmax(axis=-1)
    return ''.join(INDICES_BASE.get(v, BASE_SIZE) for v in vec)


def generate_similar_seq(inseq, subs_rate=0.1, del_rate=0, ins_rate=0):
    """
    seq has only one del or ins. Becuase it is a SV.
    """
    #TODO, try exception: 0<subs_rate<1, 0<del_rate<1, 0<ins_rate<1

    if del_rate and ins_rate:
        raise Exception('Please choose a del or ins mode.') 


    info = {
        'inseq_size':0,

        'del_size':0,
        'del_loc':0,
        'del_seq':[],

        'subs_size':0,
        'subs_locs':[],
        'subs_seqs':[],
        
        'ins_size':0,
        'ins_seq':[],
    }

    outseq = list(inseq)
    
    info['inseq_size'] = len(inseq)
    info['del_size'] = math.floor(info['inseq_size'] * del_rate) if del_rate else 0
    info['subs_size'] = math.floor(info['inseq_size'] * subs_rate)
    info['ins_size'] = math.floor(info['inseq_size'] * ins_rate) if ins_rate else 0


    # step 1: delete
    if del_rate:
        del_loc = random.randint(0, info['inseq_size']-info['del_size'])
        del_seq = outseq[del_loc:del_loc+info['del_size']]
        info['del_loc'] = del_loc
        # info['del_seq'] = ''.join(del_seq)
        info['del_seq'] = del_seq
        outseq = outseq[0:del_loc] + outseq[del_loc+info['del_size']:]

    # step 2: substitution
    if subs_rate:
        for i in range(info['subs_size']):
            outseq_len = len(outseq)-1
            subs_loc = 0
            
            while True:
                subs_loc = random.randint(0, outseq_len)
                if not subs_loc in info['subs_locs']:
                    info['subs_locs'].append(subs_loc)
                    break

            while True:
                b = random.choice(BASES)
                if outseq[subs_loc] == b:
                    continue
                outseq[subs_loc] = b
                info['subs_seqs'].append(b)
                break

    # print(''.join(outseq))
    # step 3: insertion
    if ins_rate:
        outseq_len = len(outseq)-1
        ins_loc = 0
            
        while True:
            ins_loc = random.randint(0, outseq_len)
            if not ins_loc in info['subs_locs']:
                break

        info['ins_loc'] = ins_loc


        ins_seq = [random.choice(BASES) for _ in range(info['ins_size'])]

        # info['ins_seq'] = ''.join(ins_seq)
        info['ins_seq'] = ins_seq
        outseq = outseq[0:ins_loc] + ins_seq + outseq[ins_loc:] 

    return ''.join(outseq), info
    # return ''.join(out_seq)
    # o = ''.join(outseq)
    # return Seq2Vec.encode(s)

def kmer(seq, k=50, step=1):
    seq_size = len(seq)

    k_size = k if k > 1 else math.floor(seq_size * k)

    num_kmers = seq_size - k_size + 1

    return [seq[i:i+k_size] for i in range(num_kmers)]


def print_seq(seq, info):
    outseq = list(seq)

    if info['ins_size']:
        # ins_seq = info['ins_seq']

        # ins_seq.insert(0, '\033[95m')
        # ins_seq.append('\033[00m')
        outseq = outseq[0:info['ins_loc']] + outseq[info['ins_loc']+info['ins_size']:]

    
    if info['subs_size']:
        for i, l in enumerate(info['subs_locs']):
            s = info['subs_seqs'][i]
            outseq[l] = f'\033[91m{s}\033[00m'

    if info['del_size']:
        del_seq = info['del_seq']

        del_seq.insert(0, '\033[92m')
        del_seq.append('\033[00m')

        outseq = outseq[0:info['del_loc']] + del_seq + outseq[info['del_loc']:]

    
    if info['ins_size']:
        ins_seq = info['ins_seq']

        ins_seq.insert(0, '\033[95m')
        ins_seq.append('\033[00m')
        outseq = outseq[0:info['ins_loc']] + ins_seq + outseq[info['ins_loc']:]


    # print(outseq)
    print(''.join(outseq))
    print('reds are for subs, green is for dels, purple is for ins ')



def generate_three_groups():

    s1 = generate_seq(200)
    s2, s2_info = generate_similar_seq(s1, subs_rate=0.3)
    
    # s1 = generate_seq(150)

    print(s1)
    print_seq(s2, s2_info)


def generate_two_groups():

    s1 = generate_seq(200)
    s2, s2_info = generate_similar_seq(s1, subs_rate=0.4)
    
    print(s1)
    print_seq(s2, s2_info)


def test_two_seqs():
    s = generate_seq(150)
    print(s)
    s1, info_ = generate_similar_seq(s, subs_rate=0.1, ins_rate=0.03)
    print(s1)
    s2, info_ = generate_similar_seq(s, subs_rate=0.1, ins_rate=0.05)
    print_seq(s2, info_)
    # s1, info_ = generate_similar_seq(s, subs_rate=0.4, del_rate=0.3)
    # s1, info_ = generate_similar_seq(s, subs_rate=0.1, del_rate=0.8)
    # print_seq(s1, info_)
    # print(info_)
    ss = encode(s)

    print(len(ss))  

    l = kmer(s, k=50)
    print(l)


# class SeqDatasetOHE(Dataset):
#     '''
#     Dataset for one-hot-encoded sequences
#     '''
#     def __init__(self, seqs):
#         self.seqs = seqs
#         self.seq_len = len(self.seqs)
        
#         # one-hot encode sequences, then stack in a torch tensor
#         self.ohe_seqs = torch.stack([torch.tensor(x) for x in self.seqs])
#         # print(self.ohe_seqs.shape)
    
#         # +------------------+
#         # | Get the Y labels |
#         # +------------------+
#         self.labels = torch.tensor([1 for _ in self.seqs]).unsqueeze(1)
#         # print(self.labels.shape)
        
#     def __len__(self):
#         return len(self.seqs)
    
#     def __getitem__(self,idx):
#         # Given an index, return a tuple of an X with it's associated Y
#         # This is called inside DataLoader
#         seq = self.ohe_seqs[idx]
#         label = self.labels[idx]
#         # print(seq.shape)
#         # return seq, label
#         return torch.tensor(seq, dtype=torch.float32) , label


# def build_dataloaders(batch_size=100,
#                       shuffle=True):
#     seqs = Seq2Vec.get_bulk()

#     # create Datasets    
#     train_ds = SeqDatasetOHE(seqs)
#     test_ds = SeqDatasetOHE(seqs)

#     # Put DataSets into DataLoaders
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
#     test_dl = DataLoader(test_ds, batch_size=batch_size)
    
#     return train_dl, test_dl


if __name__ == '__main__':
    generate_two_seqs()