from collections import defaultdict
import random

import pysam
from pysam import VariantFile
import numpy as np

from datasets import BASIC_BASES, DEL_BASE, INS_BASE, BASE_SIZE, MISMATCHED_BASE
from models import SV, Group, ALTERNATE, REFERENCE, REF_ATTR, ALTER_ATTR, get_sv_len


def loaddata(analysis_data):
    filter_option = analysis_data.filter_option
    sv = analysis_data.sv

    QUALITY_THRESHOLD = filter_option['quality_threshold'] 
    BAM_FILES = filter_option['bam_files']
    FETCH_SV_LEN = filter_option['fetch_sv_len']

    # QUERY_CHROM = filter_option['chrom']
    # QUERY_START = filter_option['query_start']-2
    # QUERY_END = filter_option['query_end']
    # QUERY_SVTYPE = filter_option['query_svtype']
    # SEQ_SIZE = filter_option['seq_size']

    QUERY_CHROM = sv.chrom
    QUERY_START = sv.query_pos
    QUERY_END = sv.query_pos_end
    QUERY_SVTYPE = sv.t
    SEQ_SIZE = sv.query_l

    consesus_reads_from_bams = {}
    consesus_reads_attr_from_bams = {}
    coverage_from_bams = {}
    consesus_from_bams = {}
    read_counters_bams = {}
    read_counter_ratios_bams = {}

    # for file_index, bam_file in enumerate(BAM_FILES):
    for bam_file in BAM_FILES:
        bam_reader = pysam.AlignmentFile(bam_file, 'rb')
        # print(f'bam_file:{bam_file}=============================================================')
        ## consesus
        # result of count_coverage
        # [ACGT]
        # [[2, 2, 1, 2, 2, 1, 1, 3, 5, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 2, 0, 1, 2]]
        # A [2, 2, 1, 2, 2, 1, 1, 3, 5, 7], 
        # C [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        # G [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 
        # T [0, 0, 1, 0, 0, 1, 2, 0, 1, 2]
        # consesus
        # AAAAAAAA

        # create consesus
        consesus_seq = []
        count_coverage = [] # [3,54,8,3,3]

        count_coverage_acgt = bam_reader.count_coverage(QUERY_CHROM, QUERY_START, QUERY_END, quality_threshold=0)

        # consesus
        for i in range(QUERY_END-QUERY_START):
            count_acgt = [count_coverage_acgt[j][i] for j in range(BASE_SIZE)]
            # print(count_acgt)

            # count
            count_coverage.append(sum(count_acgt))

            # consesus base
            if sum(count_acgt) == 0:
                consesus_seq.append(DEL_BASE)
            else:
                m = np.argmax(count_acgt)
                consesus_seq.append(BASIC_BASES[m])

        # print('consesus_seq================')
        # print(len(consesus_seq), ''.join(consesus_seq))

        consesus_from_bams[bam_file] = consesus_seq
        coverage_from_bams[bam_file] = count_coverage

        # https://pysam.readthedocs.io/en/latest/api.html?highlight=aligned_pairs#pysam.AlignedSegment
        consesus_read_seq_from_bam = []
        consesus_read_seq_attr_from_bam = []
        count = 0
        ref_read_counter = 0
        alter_read_counter = 0

        for read in bam_reader.fetch(QUERY_CHROM, QUERY_START, QUERY_END):
            count = count + 1
            # filter 
            if read.is_duplicate == True or \
                read.is_secondary == True or \
                read.mapping_quality == 0 or \
                read.is_unmapped == True:
                continue

            entering_query = False
            leaving_query = False

            entering_pos = QUERY_START
            leaving_pos = QUERY_END

            consesus_read_seq = []
            inserted_base_counter = 0
            deleted_base_counter = 0
            mismatched_base_counter = 0

            # arrange read base
            read_query_sequence = read.query_sequence
            for aligned_pair in read.get_aligned_pairs(matches_only=False, with_seq=True):
                # print(aligned_pair)

                aligned_pair_pos = int(aligned_pair[1]) if aligned_pair[1] != None else aligned_pair[1]

                if aligned_pair_pos != None and  aligned_pair_pos >= QUERY_END:
                    leaving_query = True
                    break

                if (aligned_pair_pos != None and aligned_pair_pos >= QUERY_START) or \
                    (aligned_pair_pos == None and entering_query):

                    if entering_query == False:
                        entering_pos = aligned_pair_pos
                        entering_query = True
                    else:
                        leaving_pos = aligned_pair_pos

                    if aligned_pair[0] == None: # DEL
                        consesus_base = consesus_seq[aligned_pair_pos-QUERY_START]
                        consesus_read_seq.append(DEL_BASE)
                        deleted_base_counter += 1
                    elif aligned_pair_pos == None: # INS
                        consesus_read_seq.append(INS_BASE)
                        inserted_base_counter += 1
                    else:
                        aaa = int(aligned_pair[0])
                        if read_query_sequence[aaa] == aligned_pair[2]:
                            consesus_read_seq.append(aligned_pair[2])
                            # consesus_read_seq.append(MISMATCHED_BASE)
                        else:
                            consesus_read_seq.append(MISMATCHED_BASE)
                            mismatched_base_counter += 1

            # read seq
            if entering_query and entering_pos and leaving_pos:
                # |-----actc|
                if entering_pos-QUERY_START > 0:
                    consesus_read_seq = consesus_seq[0:entering_pos-QUERY_START] + consesus_read_seq
                # |actc-----|
                if QUERY_END-leaving_pos > 0:
                    consesus_read_seq = consesus_read_seq + consesus_seq[QUERY_END-leaving_pos:]

                read_seq = ''.join(consesus_read_seq[:SEQ_SIZE]).upper()
                
                # print(f'FETCH_SV_LEN:{FETCH_SV_LEN}')
                # print(len(read_seq), inserted_base_counter, deleted_base_counter, entering_query, leaving_query, entering_pos-QUERY_START, QUERY_END-leaving_pos, read_seq)

                if len(read_seq) >= SEQ_SIZE :
                    consesus_read_seq_from_bam.append(read_seq)

                    if (QUERY_SVTYPE == 'DEL' and deleted_base_counter>=FETCH_SV_LEN) \
                        or (QUERY_SVTYPE in ['INS', 'DUP'] and inserted_base_counter>=FETCH_SV_LEN)\
                        or (QUERY_SVTYPE == 'INV' and mismatched_base_counter>=FETCH_SV_LEN):
                        alter_read_counter += 1
                        consesus_read_seq_attr_from_bam.append(ALTER_ATTR)
                    else: 
                        consesus_read_seq_attr_from_bam.append(REF_ATTR)
                        ref_read_counter += 1

                # if len(read_seq) >= SEQ_SIZE \
                #     and ((QUERY_SVTYPE == 'DEL' and deleted_base_counter>=FETCH_SV_LEN) \
                #     or (QUERY_SVTYPE in ['INS', 'DUP'] and inserted_base_counter>=FETCH_SV_LEN)):
                #     print(len(read_seq), inserted_base_counter, deleted_base_counter, entering_query, leaving_query, entering_pos-QUERY_START, QUERY_END-leaving_pos,read_seq)
                #     consesus_read_seq_from_bam.append(read_seq)


        ## CHECK consesus_read for debug
        # print('consesus_read_seq_from_bam.....')
        # print(consesus_read_seq_from_bam)

        consesus_reads_from_bams[bam_file] = consesus_read_seq_from_bam
        consesus_reads_attr_from_bams[bam_file] = consesus_read_seq_attr_from_bam
        read_counters_bams[bam_file] = (ref_read_counter, alter_read_counter)
        read_counters_bams[bam_file] = Group(ref_read_counter, REFERENCE, alter_read_counter, ALTERNATE, analysis_data.IGNORE_ERROR)

    analysis_data.consesus_from_bams = consesus_from_bams
    analysis_data.consesus_reads_from_bams = consesus_reads_from_bams 
    analysis_data.consesus_reads_attr_from_bams = consesus_reads_attr_from_bams 
    analysis_data.coverage_from_bams = coverage_from_bams
    analysis_data.read_counters_bams = read_counters_bams
    # analysis_data.read_counter_ratios_bams = read_counter_ratios_bams


def get_svs_from_vcf2(filter_option, simvcf=False):
    vcf_filename = filter_option['vcf_file']
    QUERY_LEN_MIN = filter_option['query_len_min']
    QUERY_LEN_MAX = filter_option['query_len_max']


    svs = []

    if simvcf:
        with open(vcf_filename) as f:
            lines = f.readlines()
            for line in lines:
                if line.find('#') == 0:
                    continue

                (chrom, pos, svlen, svtype, gt, v) = line.split('\t')

                # if svtype in ('INS', 'DEL'):
                # if svtype in ('INV',):
                filter_option['seq_size'] = svlen
                svs.append(SV(filter_option, None, (chrom, pos, svlen, svtype, gt, v)))

        return svs

    vcf_file = VariantFile(vcf_filename)

    for sv in vcf_file.fetch():
        sv_l = abs(sv.info.get('SVLEN', 0))

        if QUERY_LEN_MIN < sv_l < QUERY_LEN_MAX:
            # print('sv_l......')
            # print(QUERY_LEN_MIN)
            # print(QUERY_LEN_MAX)
            # print(sv_l)
            filter_option['seq_size'] = sv_l
            svs.append(SV(filter_option,sv))

    print('len(svs)')
    print(len(svs))
    return svs


def get_svs_from_vcf(filter_option, simvcf=True):
    vcf_filename = filter_option['vcf_file']
    QUERY_CHROM = filter_option['chrom']
    QUERY_START = filter_option['query_start']
    QUERY_END = filter_option['query_end']
    QUERY_SVTYPE = filter_option['query_svtype']
    QUERY_LEN_MIN = filter_option['query_len_min']
    QUERY_LEN_MAX = filter_option['query_len_max']
    print(QUERY_SVTYPE)
    vcf_file = VariantFile(vcf_filename)

    print(QUERY_CHROM, QUERY_START, QUERY_END, QUERY_SVTYPE, QUERY_LEN_MIN)

    svs = []

    if QUERY_CHROM and QUERY_START and QUERY_SVTYPE:
        QUERY_START = QUERY_START - 100  if QUERY_START > 100  else 0
        QUERY_END = QUERY_START + filter_option['seq_size']

        for sv in vcf_file.fetch(QUERY_CHROM, QUERY_START, QUERY_END):
            sv_l = abs(sv.info.get('SVLEN', 0))
            if sv_l < QUERY_LEN_MIN or sv_l > QUERY_LEN_MAX:
                continue
            if sv.info.get("SVTYPE", "None") == QUERY_SVTYPE:
                svs.append(SV(sv))
                break
    else:
        for sv in vcf_file.fetch():
            # sv_l = abs(sv.info.get('SVLEN', 0)[0])
            sv_l = get_sv_len(sv)

            # print(QUERY_LEN_MAX)

            if sv_l < QUERY_LEN_MIN or sv_l > QUERY_LEN_MAX:
                continue

            # print(sv.info.get("SVTYPE", "None"), QUERY_SVTYPE)
            if sv.info.get("SVTYPE", "None") != QUERY_SVTYPE:
                continue

            filter_option['seq_size'] = sv_l
            s = SV(filter_option, sv)
            # print(s)
            svs.append(s)

    return svs

"""
def get_svs_from_bed(filer_option):
    bed_filename = filter_option['bed_file']
    query_chrom = filter_option['chrom']
    QUERY_START = filter_option['QUERY_START']
    QUERY_END = filter_option['query_end']

    svs = []
    with open(bed_filename, 'r') as f:
        for line in freadlines():
            if line.startswith('#') or len(line)==0:
                 continue

            _line = line.strip().split('\t')
            
            chrom, start, end = _line[:3]
            if query == chrom and start >= QUERY_START and start <= QUERY_END:
               svs.append(('None', int(start), int(end)-int(start)) 
      
    return svs
"""

if __name__ == '__main__':
    main()
