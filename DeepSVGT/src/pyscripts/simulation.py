import math
import random

from datasets import BASES

random.seed(0)

def generate_seq(seq_size=150):
    seq = [random.choice(BASES) for _ in range(seq_size)]
    return ''.join(seq)


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


def generate_two_groups(seq_size=200, group_subs_rate=0.4, subs_rate=0.1, kmer_k_size=0):
    group1_seed_seq = generate_seq(seq_size)
    group2_seed_seq, s2_info = generate_similar_seq(group1_seed_seq, subs_rate=group_subs_rate)
    # group2_seed_seq = generate_seq(seq_size)
    
    count = 150

    group = []
    group1 = []
    group2 = []
    for _ in range(count):
        # if kmer_k_size:
        #     gs1 = kmer(generate_similar_seq(group1_seed_seq, subs_rate)[0], kmer_k_size)
        #     gs2 = kmer(generate_similar_seq(group2_seed_seq, subs_rate)[0], kmer_k_size)
        #     group.extend(gs1)
        #     group.extend(gs2)
        #     group1.extend(gs1)
        #     group2.extend(gs2)
        # else:
        #     gs1 = encode(generate_similar_seq(group1_seed_seq, subs_rate)[0])
        #     gs2 = encode(generate_similar_seq(group2_seed_seq, subs_rate)[0])

        #     group.append(gs1)
        #     group.append(gs2)
        #     group1.append(gs1)
        #     group2.append(gs2)


        gs1 = generate_similar_seq(group1_seed_seq, subs_rate)[0]
        gs2 = generate_similar_seq(group2_seed_seq, subs_rate)[0]

        group.append(gs1)
        group.append(gs2)
        group1.append(gs1)
        group2.append(gs2)


    return group, group1, group2


def generate_three_groups(seq_size=200, group_subs_rate=0.4, subs_rate=0.2, kmer_k_size=0):
    group1_seq = generate_seq(seq_size)
    # group2_seed_seq, s2_info = generate_similar_seq(group1_seq, subs_rate=group_subs_rate)
    group2_seq = generate_seq(seq_size)
    # group3_seq, s3_info = generate_similar_seq(group1_seq, subs_rate=group_subs_rate)
    group3_seq = generate_seq(seq_size)
    count = 50

    s = []
    for _ in range(count):
        if kmer_k_size:
            s.extend(kmer(generate_similar_seq(group1_seq, subs_rate)[0], kmer_k_size))
            s.extend(kmer(generate_similar_seq(group2_seq, subs_rate)[0], kmer_k_size))
            s.extend(kmer(generate_similar_seq(group2_seq, subs_rate)[0], kmer_k_size))
        else:
            s.append(encode(generate_similar_seq(group1_seq, subs_rate)[0]))
            s.append(encode(generate_similar_seq(group2_seq, subs_rate)[0]))
            s.append(encode(generate_similar_seq(group2_seq, subs_rate)[0]))

    return s

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
