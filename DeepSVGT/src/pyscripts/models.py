import math
from datetime import datetime
from decimal import Decimal

from scipy.stats import binom
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# IGNORE_RATIO = 0.2

ALTERNATE = 'Alternate'
REFERENCE = 'Reference'
EST_ALTERNATE = 'Est. Alternate'
EST_REFERENCE = 'Est. Reference'

REF_ATTR = 0
ALTER_ATTR = 1

# PRE_QUERY_LEN = 5
# POST_QUERY_LEN = 2

def get_sv_len(sv):
    sv_l = 0
    if isinstance(sv.info.get('SVLEN', 0), tuple):
        sv_l = sv.info.get('SVLEN', 0)[0]
    else:
        sv_l = sv.info.get('SVLEN', 0)

    return abs(sv_l)

class SV:
    def __init__(self, filter_option, sv=None, attrs=None):
        self.filter_option = filter_option

        if sv:
            self.set_sv(sv)

        if attrs:
            self.set_from_attrs(attrs)

        return None


    def set_sv(self, sv):
        # print('sv.chrom=====')
        # print(sv.chrom)
        # print([sample[1].items() for sample in sv.samples.items()])
        # self.chrom = 'chr' + sv.chrom
        self.chrom = sv.chrom
        self.t = sv.info.get('SVTYPE', 'None')
        # self.l = abs(sv.info.get('SVLEN', 0)[0])
        self.l = get_sv_len(sv)
        self.pos = sv.pos
        self.pos_end = self.pos + self.l

        PRE_QUERY_LEN = self.filter_option['pre_query_len']
        POST_QUERY_LEN = self.filter_option['post_query_len']

        self.query_pos = sv.pos - PRE_QUERY_LEN
        self.query_pos_end = sv.pos + self.l + POST_QUERY_LEN
        self.query_l = self.l + PRE_QUERY_LEN + POST_QUERY_LEN

        if sv.samples.get("SAMPLE", None) and sv.samples["SAMPLE"].get("GT", None):
            self.gt = f'{sv.samples["SAMPLE"]["GT"][0]}/{sv.samples["SAMPLE"]["GT"][1]}'
        elif len(sv.samples.items()) == 1: # for human, HG002
            tempgt = sv.samples.items()[0][1].items()[0][1]
            if tempgt[0] != None and tempgt[1] != None:
                self.gt = '/'.join(map(str, tempgt))
            else:
                self.gt = '-/-'
        else:
            self.gt = ''


        # if self.filter_option['temp_vcf_attr']: # for human 
        #     self.gt = [sample[1].items()[2][1] for sample in sv.samples.items()][0]
        #     # print([sample[1].items() for sample in sv.samples.items()])
        #     # print([sample[1].items() for sample in sv.samples.items()][0])
        #     # print([sample[1].items()[0][1] for sample in sv.samples.items()][0])

        #     print('len(sv.samples.items())')
        #     print(len(sv.samples.items()))
        #     for sample in sv.samples.items():
        #         print(sample[1].items()[0][1])


    def set_from_attrs(self, attrs):
        (chrom, pos, svlen, svtype, gt, v) = attrs 

        self.chrom = chrom
        self.t = svtype
        self.l = int(svlen)
        self.pos = int(pos)
        self.pos_end = self.pos + self.l

        PRE_QUERY_LEN = self.filter_option['pre_query_len']
        POST_QUERY_LEN = self.filter_option['post_query_len']

        self.query_pos = self.pos - PRE_QUERY_LEN
        self.query_pos_end = self.pos + self.l + POST_QUERY_LEN
        self.query_l = self.l + PRE_QUERY_LEN + POST_QUERY_LEN

        self.gt = gt


    def __str__(self):
        return f'{self.chrom}, {self.pos:,}, {self.pos_end:,} {self.l:,}, {self.t}, {self.query_pos}-{self.query_pos_end}-{self.query_l}, {self.gt}'


    def title(self):
        return f'{self.chrom}, {self.pos:,}({self.l:,}), {self.t}'


    def title_for_exception(self, sep='\t'):
        return f'{self.chrom}{sep}{self.pos:,}{sep}{self.pos_end:,}{sep}No result\n'


    def fname(self, ext='png'):
        return f'{self.chrom}:{self.pos}_{self.t}.{ext}'


    def fname2(self, ext='png'):
        # return f'{self.chrom}:{self.pos}_{self.t}_{datetime.now().strftime("%m_%d_%H_%M_%S")}.{ext}'
        # return f'{self.chrom}_{self.pos}_{self.t}_{datetime.now().strftime("%m_%d_%H_%M_%S")}.{ext}'
        return f'{self.chrom}:{self.pos}-{self.t}_{datetime.now().strftime("%m_%d_%H_%M_%S")}.{ext}'


class PassValue:
    def __init__(self, value, is_pass):
        self.value = round(value, 4)
        self.is_pass = is_pass

    def __str__(self):
        return f'value:{self.value}, is_pass:{self.is_pass}'

    def get_is_pass(self):
        # if self.value < 10 and 
        return self.is_pass

    def get_value(self):
        return self.value


class GroupItem:
    def __init__(self, counter, est_type):
        self.counter = counter
        self.est_type = est_type

    def __str__(self):
        return f'{self.counter}[{self.est_type}]'


class Group:
    def __init__(self, cluster1_count, cluster1_est_type, cluster2_count, cluster2_est_type, ignore_pvalue=0.05):
        # ref count is bottom alter is top on graph
        if cluster1_est_type == EST_REFERENCE or cluster1_est_type == REFERENCE:
            self.bottom_group_item = GroupItem(cluster1_count, cluster1_est_type)
            self.top_group_item = GroupItem(cluster2_count, cluster2_est_type)
        else:
            # self.bottom_group_item = GroupItem(cluster2_count, cluster1_est_type)
            # self.top_group_item = GroupItem(cluster1_count, cluster2_est_type)
            self.bottom_group_item = GroupItem(cluster2_count, cluster2_est_type)
            self.top_group_item = GroupItem(cluster1_count, cluster1_est_type)

        if cluster1_count > cluster2_count:
            b_counter = cluster1_count
            t_counter = cluster2_count
        else:
            b_counter = cluster2_count
            t_counter = cluster1_count

        # ratio
        total_count = cluster1_count + cluster2_count
        self.group_ratio = 0 if t_counter == 0 else t_counter/total_count

        # pvalue
        # cluster1_pvlaue = binom.pmf(cluster1_count, n=total_count, p=0.5)
        # # cluster2_pvlaue = binom.pmf(cluster2_count, n=total_count, p=0.5)
        # # is_pass = True if cluster1_pvlaue < ignore_pvalue else False
        # is_pass = True if cluster1_pvlaue > ignore_pvalue else False
        # print(f'total_count:{total_count}, cluster1_count:{cluster1_count}, cluster2_count:{cluster2_count}, pvalue:{round(cluster1_pvlaue,4)}, is_pass:{is_pass}')
        # self.pvalue = PassValue(cluster1_pvlaue, is_pass)

        # print(f'binom.pmf(cluster1_count, n=total_count, p=0.5):{cluster1_pvlaue}')
        # print(f'binom.pmf(cluster2_count, n=total_count, p=0.5):{cluster2_pvlaue}')

        # if binom.pmf(cluster1_count, n=total_count, p=0.5) < ignore_pvalue:
        #     self.over_pvalue = True
        # elif binom.pmf(cluster2_count, n=total_count, p=0.5) < ignore_pvalue:
        #     self.over_pvalue = True
        # else:
        #     self.over_pvalue = False

    def __str__(self):
        return f'{self.top_group_item}/{self.bottom_group_item}'

    def get_counters(self):
        # if self.bottom_group_item.est_type == ALTERNATE:
        #     return (self.top_group_item.counter, self.bottom_group_item.counter)
        return (self.bottom_group_item.counter, self.top_group_item.counter)


class Cluster:
    def __init__(self, center_xy, dist_area_percent=None, est_type=None):
        self.center_xy = center_xy
        self.dist_area_percent = round(dist_area_percent, 2)
        self.est_type = est_type


class ManifoldXY:
    # def __init__(self, name=None, all_xy=None, bams_xy=None):
    def __init__(self, name='Raw'):
        self.set_scale = True

        self.name = name
        self.scaled_xy = None
        self._all_xy = None
        self.scaled_bams_xy = {}
        self._bams_xy = None # original if raw
        self.bams_group = {}
        # self.bams_group_ratio = {}
        self.est_cluster_counter = 0
        self.clusters = []

        self.bams_over_cutoff_coverage = {}

        # self.dist_area_percents = []

        # self.est_bams_genotype = {}

    def __str__(self):
        return f'{self.name}'

    @property
    def all_xy(self):
        return self._all_xy

    @all_xy.setter
    def all_xy(self, all_xy_):
        self._all_xy = all_xy_

        # if not self.scaled_xy and self.set_scale:
        #     #self.scaled_xy = np.array(StandardScaler().fit_transform(self._all_xy))
        #     #self.scaled_xy = StandardScaler().fit_transform(self._all_xy)
        #     if self.name == 'Raw':
        #         self.scaled_xy = self._all_xy
        #     else:
        #         self.scaled_xy = MinMaxScaler().fit_transform(self._all_xy)
        #         # self.scaled_xy = StandardScaler().fit_transform(self._all_xy)

        if self.name == 'Raw':
            self.scaled_xy = self._all_xy
        else:
            if len(self._all_xy) == 0:
                self.scaled_xy = []
            else:
                self.scaled_xy = MinMaxScaler().fit_transform(self._all_xy)

    @property
    def bams_xy(self):
        return self._bams_xy

    @bams_xy.setter
    def bams_xy(self, bams_xy_):
        self._bams_xy = bams_xy_

        # if not self.scaled_bams_xy and self.set_scale:
        #     for bam_file in self._bams_xy:
        #         print('bam_file&&&&&&&&&&&&&&&&&')
        #         print(bam_file)
        #         # self.scaled_bams_xy[bam_file] = StandardScaler().fit_transform(self._bams_xy[bam_file])
        #         if self.name == 'Raw':
        #             self.scaled_bams_xy[bam_file] = self._bams_xy[bam_file]
        #         else:
        #             self.scaled_bams_xy[bam_file] = MinMaxScaler().fit_transform(self._bams_xy[bam_file])
        #             # self.scaled_bams_xy[bam_file] = StandardScaler().fit_transform(self._bams_xy[bam_file])


        for bam_file in self._bams_xy:
            # print('bam_file&&&&&&&&&&&&&&&&&')
            # print(bam_file)
            # self.scaled_bams_xy[bam_file] = StandardScaler().fit_transform(self._bams_xy[bam_file])
            if self.name == 'Raw':
                self.scaled_bams_xy[bam_file] = self._bams_xy[bam_file]
            else:
                if len(self._bams_xy[bam_file]) == 0:
                    self.scaled_bams_xy[bam_file] = []
                else:
                    self.scaled_bams_xy[bam_file] = MinMaxScaler().fit_transform(self._bams_xy[bam_file])


class AnalysisData:
    def __init__(self, filter_option=None, sv=None):
        self.filter_option = filter_option
        self.sv = sv
        self.set_filter(filter_option)

        # parameters
        self.IGNORE_RATIO = self.filter_option['ignore_ratio']
        self.IGNORE_ERROR = self.filter_option['ignore_error']
        self.CUTOFF_COVERAGE = self.filter_option['cutoff_coverage'] * 2

        # read information
        self.consesus_from_bams = None
        self.consesus_reads_from_bams = None
        self.consesus_reads_attr_from_bams = None
        self._coverage_from_bams = None
        self.over_cutoff_coverage_from_bams = {} #PassValue
        
        self.read_counters_bams = None # Group object
        # self.read_counter_ratios_bams = None
        self.read_counter = 0
        self.n_bam_files = len(filter_option['bam_files'])

        self.over_cutoff_coverage = True

        # manifold information
        self.raw_xy = None
        self.manifold_xys = []
        self.est_cluster_counter = 0
        self.force = False

        # genotype
        self.est_genotype = {}

    @property
    def coverage_from_bams(self):
        return self._coverage_from_bams

    @coverage_from_bams.setter
    def coverage_from_bams(self, coverage_from_bams_):
        self._coverage_from_bams = coverage_from_bams_

        for bam_file in coverage_from_bams_:
            count_coverage_bam = coverage_from_bams_[bam_file]
            avg_count_coverage = sum(count_coverage_bam) / len(count_coverage_bam)
            # print(f'avg_count_coverage:{avg_count_coverage}')

            is_pass = True if avg_count_coverage < self.CUTOFF_COVERAGE else False
            self.over_cutoff_coverage_from_bams[bam_file] = PassValue(avg_count_coverage, is_pass)


    def set_filter(self, filter_option):
        if not filter_option.get('chrom', None):
            self.filter_option['chrom'] = self.sv.chrom

        if not filter_option.get('query_start', None):
            # self.filter_option['query_start'] = sv.pos - 100  if sv.pos > 100  else 0
            # self.filter_option['query_start'] = self.sv.pos
            self.filter_option['query_start'] = self.sv.query_pos

        if not filter_option.get('query_end', None):
            self.filter_option['query_end'] = self.sv.query_pos_end

        if not filter_option.get('query_svtype', None):
            self.filter_option['query_svtype'] = self.sv.t


    def get_epochs(self):
        EPOCHS_ADJ = self.filter_option.get('epochs_adj', 0)
        # ((max_epochs - min_epochs 100) * seq_len) / (min_len - max_len) + max_epochs
        epochs = round((((100 - 30) * self.sv.query_l) / (50 - 800)) + 120, 0)
        return int(epochs) + int(EPOCHS_ADJ)


    def add_raw_xy(self, raw_xy):
        if self.raw_xy:
            raise Exception("Sorry, you can set it only once.")

        self.raw_xy = raw_xy
        self.manifold_xys.append(self.raw_xy)

    def get_title(self):
        sv = self.sv
        filter_option = self.filter_option

        title = (
                f'Chrom:{self.sv.chrom}, '
                f'Position:{self.sv.pos:,}, '
                f'Length:{self.sv.l:,}, '
                f'Type:{self.sv.t}, '
                f'Query:{self.sv.query_pos:,}~{self.sv.query_pos_end:,}, '
                f'Seq size:{self.sv.query_l:,}, '
                f'Reads:{self.read_counter}, '
                # f'Cluster algo:{self.filter_option.get("cluster_algo", "")}, '
                f'Epochs:{self.get_epochs()}\n',
                f'Est. cluster:{self.est_cluster_counter}',
                f'Genotype:{self.get_genotypes(",")}',
                f'Note:{self.get_note()}',
            )

        return ' '.join(title)

    def estimate_genotypes_with_reads_attr(self):
        read_counters_bams = self.read_counters_bams

        for bam_file in read_counters_bams:
            ref_counter, alter_counter = read_counters_bams[bam_file].get_counters()
            self.est_genotype[bam_file] = self.likelihood(ref_counter, alter_counter)

    def estimate_genotypes_with_reads_group_attr(self):
        bams_group = self.raw_xy.bams_group # Group object

        # read group ratios
        for bam_file in bams_group:
            ref_counter, alter_counter = bams_group[bam_file].get_counters()
            self.est_genotype[bam_file] = self.likelihood(ref_counter, alter_counter)

    def get_genotypes(self, seperator='\t'):
        if self.read_counter == 0:
            return 'No reads'

        # if self.analysis_data.over_read_ratio = False:

        return seperator.join([self.est_genotype[bf] for bf in self.est_genotype])

    def get_note(self, seperator=', \n'):
        # cluster 
        if self.raw_xy and self.raw_xy.clusters:
            return self.get_ss() + ', ' + seperator + ', '.join([f'{cluster.est_type}:{cluster.dist_area_percent}%' for i, cluster in enumerate(self.raw_xy.clusters)]) + ' | ' + self.get_genotypes_with_reads_group_attr()

        # non cluster
        return 'No cluster | ' + self.get_ss() + ' | ' +self.get_genotypes_with_reads_attr()

    def write_genotypes(self):
        # return f'{self.sv.chrom}\t{self.sv.pos:,}\t{self.sv.pos_end:,}\t{self.sv.l:,}\t{self.sv.t}\t{self.get_genotypes()}\t{self.get_note()}|{self.get_ss()}\n'
        # return f'{self.sv.chrom}\t{self.sv.pos:,}\t{self.sv.pos_end:,}\t{self.sv.l:,}\t{self.sv.t}\t{self.get_genotypes()}\t{self.get_note()}|{self.get_genotypes_with_reads_group_attr()}\n'
        return f'{self.sv.chrom}:{self.sv.pos}-{self.sv.pos_end}\t{self.sv.l:,}\t{self.sv.t}\t{self.get_genotypes()}\t{self.get_note(",")}|{self.get_genotypes_with_reads_group_attr()}\n'

    def get_genotypes_with_reads_group_attr(self):
        bams_group = self.raw_xy.bams_group
        return '|'.join([ f'{bams_group[bf]}' for bf in bams_group])

    def get_genotypes_with_reads_attr(self):
        read_counters_bams = self.read_counters_bams
        return '|'.join([f'{read_counters_bams[bf]}' for bf in read_counters_bams])

    def plot_name(self):
        return self.sv.fname2()

    def get_bams(self):
        return ', '.join(self.consesus_from_bams.keys())

    # def is_over_read_ignore_error(self, bam_file):
    #     read_counters_bams_group = self.read_counters_bams[bam_file]
    #     print(read_counters_bams_file)
    #     t = sum(read_counters_bams_file)
    #     print(sum(read_counters_bams))
    #     print(f'{read_counters_bams[bam_file][0]}/{read_counters_bams[bam_file][1]}')
    #     if binom.pmf(read_counters_bams_file[0], n=t, p=0.5) < self.IGNORE_ERROR:
    #         return True

    #     if binom.pmf(read_counters_bams_file[1], n=t, p=0.5) < self.IGNORE_ERROR:
    #         return True

    #     return False

    # def is_over_read_group_ignore_error(self, bam_file):
    #     bams_group = self.raw_xy.bams_group[bam_file]

    #     return '|'.join([ f'{bams_group[bf]}' for bf in bams_group])


    #     if binom.pmf(read_counters_bams_file[0], n=t, p=0.5) < self.IGNORE_ERROR:
    #         return True

    #     if binom.pmf(read_counters_bams_file[1], n=t, p=0.5) < self.IGNORE_ERROR:
    #         return True

    #     return False

    def get_ss(self):
        TARGET_BAM_FILE = self.filter_option['target_bam_file']
        # if '44' in self.filter_option['vcf_file']:
        #     s = 0
        # elif '40' in self.filter_option['vcf_file']:
        #     s = 1
        # elif '09' in self.filter_option['vcf_file']:
        #     s = 2
        # m = ''
        matched = True
        decision = 'MATCHED'

        for k, bf in enumerate(self.est_genotype):
            if k == TARGET_BAM_FILE:
                if self.sv.gt == '-/-':
                    matched = False
                    decision = 'No REF GENOTYPE'
                elif self.est_genotype[bf] == self.sv.gt:
                    # m = self.sv.gt
                    matched = True
                    decision = 'PAIRED'
                elif self.est_genotype[bf] == '0/1' and self.sv.gt in ['0/1', '1/0']:
                    # m = self.sv.gt
                    matched = True
                    decision = 'PAIRED'
                else:
                    # m = f'UNMATCHED {self.sv.gt}'
                    matched = False
                    decision = 'UNMATCHED'
                break

        # s = f'{self.sv.chrom}:{self.sv.pos}-{self.sv.pos_end}' if matched else f'UNMATCHED {self.sv.gt}| {self.sv.chrom}:{self.sv.query_pos}-{self.sv.query_pos_end}'
        s = f'{decision} {self.sv.gt}| {self.sv.chrom}:{self.sv.query_pos}-{self.sv.query_pos_end}'

        # return f'{s}, {self.get_pvalues()}'
        return f'{s}'

    def process_info(self):
        return f'Reads:{self.read_counter}, Est cluster:{self.est_cluster_counter}, Epochs:{self.get_epochs()}'

    # def get_pvalues(self):
    #     # read pvalue
    #     read_counters_bams = self.read_counters_bams
    #     s1 = ', '.join([f'{read_counters_bams[bam_file].pvalue.get_value()}' for bam_file in read_counters_bams])

    #     # read group pvalue
    #     group = self.raw_xy.bams_group
    #     s2 = ', '.join([f'{group[bam_file].pvalue.get_value()}' for bam_file in group])
    #     return f'read:{s1}, group:{s2}'

    # def get_coverages(self):
    #     over_cutoff_coverage_from_bams = self.over_cutoff_coverage_from_bams
    #     return ', '.join([f'{over_cutoff_coverage_from_bams[bam_file].get_value()}' for bam_file in over_cutoff_coverage_from_bams])

    def likelihood(self, counter1, counter2):
        e = self.filter_option['likelihood_e']
        # e = 0.12
        # e = 0.00005
        # e = 0.00001
        # e = 0.0001
        # e = 0.005
        # e = 0.001 #48
        # e = 0.01 #45
        # e = 0.1 #26 | 5
        # e = 0.2 #67
        # e = 0.15 #26
        # e = 0.12 #238
        # e = 0.08 #26 |5
        # e = 0.03 # 26
        # e = 0.05 # 26
        # QUERY_SVTYPE = self.filter_option['query_svtype']
        # if QUERY_SVTYPE == 'DEL':
        #     # e = 0.00005 #0 
        #     # e = 0.0005 #0
        #     # e = 0.005 # 0
        #     # e = 0.05 #200
        #     # e = 0.1 #92
        #     e = 0.12 #0
        #     # e = 0.15 #3
        # elif QUERY_SVTYPE == 'INS':
        #     e = 0.12 # for del 6(10), 1(20)
        #     # e = 0.05 # for del 6(10), 1(20)
        #     # e = 0.00005 # default for del 61(10), (20)
        #     # e = 0.005 # for del 11(10), 7(20)
        #     # e = 0.05 # for del 6(10), 1(20)
        #     # e = 0.01 # for  11(10), 6(20)
        #     # e = 0.001

        # print(counter1, counter2)

        # 0 means no reads (?) TODO
        if counter1 + counter2 == 0:
            print('@@@@@@@@@@@@@@@@@@')
            return '0/0'

        # normalise 
        ncounter1 = (counter1*100) / (counter1 + counter2)
        ncounter2 = (counter2*100) / (counter1 + counter2)
        counter1 = int(ncounter1)
        counter2 = int(ncounter2)

        likelihood00 = Decimal(counter1*math.log10(1-e)) + Decimal(counter2*math.log10(e))
        likelihood01 = Decimal((counter1+counter2)*math.log10(0.5))
        likelihood11 = Decimal(counter1*math.log10(e)) + Decimal(counter2*math.log10(1-e))
       
        likelihoods = [likelihood00, likelihood01, likelihood11]
        m_likelihood_idx = likelihoods.index(max(likelihoods))

        est_genotype = './.'

        if m_likelihood_idx == 0:
            est_genotype = '0/0'
        elif m_likelihood_idx == 1:
            est_genotype = '0/1'
        elif m_likelihood_idx == 2:
            est_genotype = '1/1'

        combination = Decimal(math.log10(math.comb(counter1+counter2, counter1)))

        # phred scaled likelihoods
        prob00 = -10 * (likelihood00 + combination)
        prob01 = -10 * (likelihood01 + combination)
        prob11 = -10 * (likelihood11 + combination)

        prob = [int(prob00), int(prob01), int(prob11)]

        # print("@@@=====================")
        # print(likelihoods)
        # print(est_genotype, prob)
        return est_genotype
