import argparse

from loaddata import loaddata, get_svs_from_vcf
from utils import plot_latent, save_reads, get_filename, reset_filter_option, shorten, create_result_dir
from models import  AnalysisData
from cluster import generate_xys, analysis_reads


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--chrom', type=str, help='chromosome')
    parser.add_argument('--query_start', type=int, help='query start')
    parser.add_argument('--query_end', type=int, help='query end')
    parser.add_argument('--query_svtype', type=str, help='query SV type')
    parser.add_argument('--query_len_min', type=int, default=50, help='query len min')
    parser.add_argument('--query_len_max', type=int, default=800, help='query len max')
    parser.add_argument('--pre_query_len', type=int, default=5, help='pre query len')
    parser.add_argument('--post_query_len', type=int, default=2, help='post query len')
    parser.add_argument('--fetch_sv_len', type=int, default=40, help='fetch sv len')
    parser.add_argument('--ignore_ratio', type=float, default=0.20, help='ignore ratio')
    parser.add_argument('--ignore_error', type=float, default=0.05, help='ignore error')
    parser.add_argument('--cutoff_coverage', type=float, default=16, help='cutoff coverage')
    # parser.add_argument('--temp_vcf_attr', type=str, help='temp vcf attr')
    # parser.add_argument('--seq_size', type=int, default=512, help='seq size')
    parser.add_argument('--bam_files', type=str, help='BAM files', action='append', nargs='+')
    parser.add_argument('--vcf_file', type=str, help='VCF file')
    parser.add_argument('--target_bam_file', type=int, help='target bam file') # todo remove
    # parser.add_argument('--bed_files', type=str, help='BED file')
    parser.add_argument('--result_dir', type=str, default='./1phd/data/sim-depth8-acc8595', help='folder to save')
    parser.add_argument('-v', '--verbose', action='store_false')
    # parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--epochs_adj', type=int, default=0, help='epochs adj')
    parser.add_argument('--likelihood_e', type=float, default=0.1, help='likelihood e')
    parser.add_argument('--quality_threshold', type=int, default=15, help='quality threshold')
    parser.add_argument('--cluster_algo', type=str, default='bgmm', help='cluster algorithm')

    args = parser.parse_args()
    filter_option = args.__dict__

    filter_option['bam_files'] = [bam_file for bam_files in filter_option['bam_files'] for bam_file in bam_files]

    # todo remove
    # reset_filter_option(filter_option)

    svs = get_svs_from_vcf(filter_option, simvcf=True)

    print(f'svs len:{len(svs)}')
    run(filter_option, svs)


def run(filter_option, svs):
    r = None
    # r = (26939849, 26939849)

    create_result_dir(filter_option, r)
    fname = get_filename(filter_option)

    bamfiles = ', '.join([shorten(bf) for bf in filter_option['bam_files']])
    bamfiles_long = ', '.join([shorten(bf, size=40) for bf in filter_option['bam_files']])

    with open(fname, 'w') as f:
        f.write(f'# \n')
        f.write(f'# Bam files:{bamfiles_long} \n')
        f.write(f'# Pre query len:{filter_option["pre_query_len"]}, Post query len:{filter_option["post_query_len"]}, Ignore ratio:{filter_option["ignore_ratio"]} \n')
        f.write(f'# \n')
        f.write(f'#Chrom\tStart pos\tEnd pos\tLength\tType\tGenotype({bamfiles_long})\tNote\n')
        for i, sv in enumerate(svs):
            # if sv.pos != 535755:
            # if sv.pos !=  1567550:
            # if sv.pos !=  1370160:
            #     continue

            # if r and i < r[0]:
            #     continue
            # elif r and i > r[1]:
            #     break
            # if i > 10: break

            print(f'{i}======')
            # filter_option['seq_size'] = sv.l
            # print("filter_option['seq_size']")
            # print(filter_option['seq_size'])
            s = run_sv(filter_option, sv)
            f.write(s)

            # try:
            #     s = run_sv(filter_option, sv)
            #     f.write(s)
            # except Exception as e:
            #     # print(str(e))
            #     f.write(sv.title_for_exception())
            # reset_filter_option(filter_option)


def run_sv(filter_option, sv=None):
    analysis_data = AnalysisData(filter_option, sv=sv)


    # # load data
    # loaddata(analysis_data)

    # # autoencoder model generates data
    # generate_xys(analysis_data)

    # # analysis reads and clusters
    # analysis_reads(analysis_data)

    # # plot
    # if filter_option['verbose']:
    #     plot_latent(analysis_data)
    #     #save_reads(consesus_reads_from_bams, filter_option)

    # print(analysis_data.process_info())
    # print(analysis_data.write_genotypes())

    # return analysis_data.write_genotypes()


    try:
        loaddata(analysis_data)

        # autoencoder model generates data
        generate_xys(analysis_data)

        # analysis reads and clusters
        analysis_reads(analysis_data)

        # plot
        if filter_option['verbose']:
            plot_latent(analysis_data)
            #save_reads(consesus_reads_from_bams, filter_option)

        print(analysis_data.process_info())
        print(analysis_data.write_genotypes())

        s = analysis_data.write_genotypes()
    except Exception as e:
        print(e)
        s = str(e)
    finally:
        return s


if __name__ == '__main__':
    main()

