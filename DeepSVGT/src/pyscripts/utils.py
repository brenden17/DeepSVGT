import os
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter

from PIL import Image

from cluster import get_mainfold_xy_groups, get_read_counters
from models import ALTERNATE, REFERENCE, EST_ALTERNATE, EST_REFERENCE, SV


def reset_filter_option(filter_option):
    filter_option['chrom'] = None
    filter_option['query_start'] = None
    filter_option['query_end'] = None
    filter_option['query_svtype'] = None
    filter_option['seq_size'] = 0


def logtime(msg=''):
    print(f'==== {datetime.now()} | {msg}')


def shorten(s, size=10, split=True):
    l = s.split('/')[-1] if split else s
    return l if len(l) < size else l[:size] + '...'


def find_height(groups1, groups2, refs, alters):
    a = [sum(x) for x in zip(groups1, groups2)]
    b = [sum(x) for x in zip(refs, alters)]

    max_a = max(a) if a else 0
    max_b = max(b) if b else 0
    return max_a if max_a > max_b else max_b


# def save_2d(X):
#     with open('1phd/tmp9227.txt', 'wb') as f:
#          np.save(f, X)


# def load_2d():
#     with open('1phd/tmp6606.txt', 'rb') as f:
#         X = np.load(f)

#     return X


# def convert_read_ratios(read_ratios):
#     # [[1, 2], [1,2], [1,2]] -> [[1, 1, 1], [2, 2, 2]]
#     read_ratio_list = [read_ratios[bam_file] for bam_file in read_ratios]

#     read_ratio1 = [read_ratio[0] for read_ratio in read_ratio_list]
#     read_ratio2 = [read_ratio[1] for read_ratio in read_ratio_list]

#     return read_ratio1, read_ratio2


# def convert_read_ratio(read_counter_property_bams):
#     labels = []
#     mapped_reads = []
#     unmapped_reads = []
#     for bam_file in read_counter_property_bams:
#         counter = read_counter_property_bams[bam_file]
       
#         labels.append(bam_file)
#         mapped_reads.append(counter[0])
#         unmapped_reads.append(counter[1])

#     return labels, mapped_reads, unmapped_reads


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/model.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')


# def plot_latent2(analysis_data):
#     filter_option = analysis_data.filter_option 
#     coverage_from_bams = analysis_data.coverage_from_bams
#     raw_xy = analysis_data.raw_xy

#     title = analysis_data.get_title(),

#     fig = plt.figure(figsize=(15, 10))
#     fig.suptitle(' '.join(title))

#     ncols = 2
#     width_ratios = [1 for _ in range(ncols)]
    
#     gs = gridspec.GridSpec(2, ncols, width_ratios=width_ratios, height_ratios=[1, 1])

#     # coverage plot
#     axes = plt.subplot(gs[0, :])
#     axes.set_title('Coverage')

#     # line
#     x = np.array(range(filter_option['query_start'], filter_option['query_start']+filter_option['seq_size']))
#     for k in coverage_from_bams:
#         axes.plot(x, coverage_from_bams[k], label=shorten(k), linestyle='--')

#     # sv range
#     sv = analysis_data.sv
#     axes.axvspan(sv.pos, sv.pos_end, color='gray', alpha=0.5)

#     at = AnchoredText(sv.t, prop=dict(size=15), frameon=True, loc='upper center')
#     at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
#     axes.add_artist(at)

#     # show whole big number
#     axes.get_xaxis().get_major_formatter().set_useOffset(False)
#     axes.get_xaxis().get_major_formatter().set_scientific(False)

#     axes.grid(axis='y', linestyle='--')
#     axes.legend()

#     # manifold and counter plot
#     refs, alters, bam_labels = get_read_counters(analysis_data)
#     mainfold_xy_groups = get_mainfold_xy_groups(analysis_data)

#     axes = plt.subplot(gs[-1, 0])
#     axes.set_title(f'{raw_xy.name}')
#     scaled_bams_xy = raw_xy.scaled_bams_xy
#     est_cluster_counter = raw_xy.est_cluster_counter
#     clusters = raw_xy.clusters

#     for bam_file in scaled_bams_xy:
#         # print('#####################################3')
#         # print(bam_file)
#         # print(scaled_bams_xy[bam_file])
#         if len(scaled_bams_xy[bam_file]) == 0:
#             continue

#         axes.scatter(scaled_bams_xy[bam_file][:, 0], scaled_bams_xy[bam_file][:, 1], label=shorten(bam_file), alpha=0.5)

#     if est_cluster_counter > 1:
#         # axes.scatter(clusters[0].xy[0], clusters[0].xy[1], label='Cluster 1', marker="D", s=80, c='r') 
#         # axes.scatter(clusters[1].xy[0], clusters[1].xy[1], label='Cluster 2', marker="D", s=80, c='y')

#         axes.scatter(clusters[0].center_xy[0], clusters[0].center_xy[1], label=clusters[0].est_type, marker="D", s=80, c='y') 
#         axes.scatter(clusters[1].center_xy[0], clusters[1].center_xy[1], label=clusters[1].est_type, marker="D", s=80, c='r')


#     axes.grid(axis='x', linestyle='-')
#     axes.grid(axis='y', linestyle='-')
#     axes.legend()

#     # counter
#     groups1, groups2 = mainfold_xy_groups.get(raw_xy.name, ([], []))
#     #print('groups1, groups2.............')
#     #print(groups1, groups2)

#     s = len(bam_labels)
#     dx = np.arange(s)
#     width = round(1/s, 1)
#     axes = plt.subplot(gs[-1, 1])
#     axes.set_title(f'{raw_xy.name} read information')

#     axes.bar(dx, refs, width=width, label=REFERENCE, alpha=0.2)
#     axes.bar(dx, alters, width=width, label=ALTERNATE, alpha=0.2, bottom=refs)

#     # print(groups1)
#     # print(groups2)

#     if groups1 or groups2:
#         axes.bar(dx+width, groups1, width=width, label=EST_REFERENCE, color='y', alpha=0.2)
#         axes.bar(dx+width, groups2, width=width, label=EST_ALTERNATE, color='r', alpha=0.2, bottom=groups1)

#     # max_el = max([sum(x) for x in zip(groups1, groups2, refs, alters)])
#     max_el = find_height(groups1, groups2, refs, alters)
#     axes.set_ylim(0, max_el + (max_el/5))
#     axes.set_xticks(dx+width/2)
#     axes.set_xticklabels([shorten(bl) for bl in bam_labels])
#     axes.set_ylabel('Number of reads')

#     axes.legend()

#     plt.tight_layout()

#     result_dir = filter_option['result_dir']
#     plt_img = os.path.join(result_dir, f'{analysis_data.plot_name()}')
#     plt.savefig(plt_img)


def plot_latent(analysis_data):
    filter_option = analysis_data.filter_option
    coverage_from_bams = analysis_data.coverage_from_bams
    manifold_xys = analysis_data.manifold_xys
    sv = analysis_data.sv

    title = analysis_data.get_title(),

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(' '.join(title))

    n = len(manifold_xys) # only show raw
    ncols = 1 if n == 0 else n * 2
    width_ratios = [1 for _ in range(ncols)]
    
    gs = gridspec.GridSpec(2, ncols, width_ratios=width_ratios, height_ratios=[1, 1])

    # coverage plot
    axes = plt.subplot(gs[0, :])
    axes.set_title('Coverage')

    # x = np.array(range(sv.query_pos, sv.query_filter_option['query_start']+filter_option['seq_size']))
    x = np.array(range(sv.query_pos, sv.query_pos+sv.query_l))

    for k in coverage_from_bams:
        axes.plot(x, coverage_from_bams[k], label=shorten(k), linestyle='--')

    axes.axvspan(sv.pos, sv.pos_end, color='gray', alpha=0.5)

    at = AnchoredText(sv.t, prop=dict(size=15), frameon=True, loc='upper center')
    at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axes.add_artist(at)

    # show whole big number
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_scientific(False)

    axes.grid(axis='y', linestyle='--')
    axes.legend()

    # manifold and counter plot
    refs, alters, bam_labels = get_read_counters(analysis_data)
    mainfold_xy_groups = get_mainfold_xy_groups(analysis_data)

    for i, manifold_xy in enumerate(manifold_xys):
        # manifold
        axes = plt.subplot(gs[-1, i*2])
        axes.set_title(f'{manifold_xy.name}')
        scaled_bams_xy = manifold_xy.scaled_bams_xy
        est_cluster_counter = manifold_xy.est_cluster_counter
        clusters = manifold_xy.clusters

        for bam_file in scaled_bams_xy:
            # print('#####################################3')
            # print(bam_file)
            # print(scaled_bams_xy[bam_file])
            if len(scaled_bams_xy[bam_file]) == 0:
                continue

            axes.scatter(scaled_bams_xy[bam_file][:, 0], scaled_bams_xy[bam_file][:, 1], label=shorten(bam_file), alpha=0.5)

        if est_cluster_counter > 1:
            # axes.scatter(clusters[0].xy[0], clusters[0].xy[1], label='Cluster 1', marker="D", s=80, c='r') 
            # axes.scatter(clusters[1].xy[0], clusters[1].xy[1], label='Cluster 2', marker="D", s=80, c='y')
            axes.scatter(clusters[0].center_xy[0], clusters[0].center_xy[1], label=clusters[0].est_type, marker="D", s=80, c='r') 
            axes.scatter(clusters[1].center_xy[0], clusters[1].center_xy[1], label=clusters[1].est_type, marker="D", s=80, c='y')


        axes.grid(axis='x', linestyle='-')
        axes.grid(axis='y', linestyle='-')
        axes.legend()

        # counter
        groups1, groups2 = mainfold_xy_groups.get(manifold_xy.name, ([], []))
        #print('groups1, groups2.............')
        #print(groups1, groups2)

        s = len(bam_labels)
        dx = np.arange(s)
        width = round(1/s, 1)
        axes = plt.subplot(gs[-1, i*2+1])
        axes.set_title(f'{manifold_xy.name} read information')

        axes.bar(dx, refs, width=width, label=REFERENCE, alpha=0.2)
        axes.bar(dx, alters, width=width, label=ALTERNATE, alpha=0.2, bottom=refs)

        # print(groups1)
        # print(groups2)

        if groups1 or groups2:
            axes.bar(dx+width, groups1, width=width, label=EST_REFERENCE, color='y', alpha=0.2)
            axes.bar(dx+width, groups2, width=width, label=EST_ALTERNATE, color='r', alpha=0.2, bottom=groups1)

        # max_el = max([sum(x) for x in zip(groups1, groups2, refs, alters)])
        max_el = find_height(groups1, groups2, refs, alters)
        axes.set_ylim(0, max_el + (max_el/5))
        axes.set_xticks(dx+width/2)
        axes.set_xticklabels([shorten(bl) for bl in bam_labels])
        axes.set_ylabel('Number of reads')

        axes.legend()

    plt.tight_layout()

    result_dir = filter_option['result_dir']
    plt_img = os.path.join(result_dir, f'{analysis_data.plot_name()}')
    plt.savefig(plt_img)
    fig.clf()
    plt.close()


def plot_sim_latent(latent_xys, tsne_xy, title):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title)

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    axes_1 = plt.subplot(gs[0, 0])
    axes_1.set_title('Original latent')
    for k in latent_xys:
        axes_1.scatter(latent_xys[k][:, 0], latent_xys[k][:, 1], label=k.split('/')[-1], alpha=0.5)
    axes_1.legend()

    axes_2 = plt.subplot(gs[0, -1])
    axes_2.set_title('Mainfolding')
    axes_2.scatter(tsne_xy[:, 0], tsne_xy[:, 1], alpha=0.5)
    
    plt.tight_layout()

    plt_img = os.path.join('./1phd/img', datetime.now().strftime("%m-%d-%H-%M-%S") + '.png')
    plt.savefig(plt_img)
    return plt_img


def create_gif_plots(plots):
    frames = [Image.open(fname) for fname in plots]
    frame_one = frames[0]
    plt_img = os.path.join('./1phd/img', datetime.now().strftime("%m-%d-%H-%M-%S") + '.gif')

    frame_one.save(plt_img,
                    format="GIF",
                    append_images=frames,
                    save_all=True, duration=1500, loop=0)


def save_reads(consesus_reads_from_bams, filter_option):
    result_dir = filter_option['result_dir']
    fname = os.path.join(result_dir, datetime.now().strftime("%m-%d-%H-%M-%S") + 'reads.txt')

    with open(fname, 'w') as fw:
        for bam_file in consesus_reads_from_bams:
            fw.write(bam_file + '=========================\n')
            for read in consesus_reads_from_bams[bam_file]:
                fw.write(read + '\n')


def get_filename(filter_option):
    result_dir = filter_option['result_dir']
    vcf_file = filter_option['vcf_file']

    l = vcf_file.split('/')[-1]
    fname = os.path.join(result_dir, f'{l}.out.{datetime.now().strftime("%m_%d_%H_%M_%S")}.txt')

    return fname


def create_result_dir(filter_option, ext=None):
    if ext:
        result_dir = filter_option['result_dir'] + f'-{ext[0]}-{ext[1]}'
        filter_option['result_dir'] = result_dir
    else:
        result_dir = filter_option['result_dir']

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


# @dataclass
# class TempSV:
#     chrom: str = ''
#     svtype: str = ''
#     pos: int = 0
#     gt: str = ''


# def read_simvcf(filter_option):
#     vcf_filename = filter_option['vcf_file']
#     lines = None
#     with open(vcf_filename) as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.find('#') == 0:
#                 continue
#             # ch, pos, l, t, g, v = line.split('\t')
#             sv = SV(filter_option, line)

#     return lines