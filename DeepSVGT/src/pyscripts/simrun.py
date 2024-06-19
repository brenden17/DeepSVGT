import argparse

import torch
import torch.nn as nn

from sklearn.manifold import TSNE, MDS

from loaddata import loaddata, get_svs_from_vcf
from datasets import build_dataloader, kmer, BASE_SIZE
from utils import plot_latent, create_gif_plots, generate_latent, plot_sim_latent, convert_2d
from simulation import generate_two_groups

from cluster import get_optimal_nclusters_bgmm, get_optimal_nclusters_gmm

from vae import *
from cnnvae import CNNAutoencoder


def main():
 
    gif_plots = []
 
    count = 10
    for i in range(count):
        print(i)
        p = run(idx=i, count=count)
        gif_plots.append(p)

#    if count > 1:
#        create_gif_plots(gif_plots)

def run(idx=1, count=9):
    SEQ_SIZE = 512
    KMER_SIZE = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_SIZE = SEQ_SIZE
    LATENT_DIMS = 2
    
    SUB_RATE = 0.0
    # EPOCHS = 10
    EPOCHS = 80

    #######################################
    # load data 
    #######################################
    group_subs_rate = 0.1 * idx
    seqs, group1, group2 = generate_two_groups(SEQ_SIZE, group_subs_rate=group_subs_rate, subs_rate=SUB_RATE, kmer_k_size=KMER_SIZE)

    # TODO data
    data, _ = build_dataloader(seqs, KMER_SIZE)

    #######################################
    # train model
    #######################################
    # model = CNNVariationalAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
    # model = Variational6LayersAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
    model = CNNAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
    # model = Variational4LayersAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
    # model = Variational2LayersAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)

    model = train(model, data, DEVICE, epochs=EPOCHS)

    latent_xy = generate_latent(model, data, DEVICE)

    latent_xys = {}
    temp_data, _ = build_dataloader(group1, KMER_SIZE)
    temp_latent_xy = generate_latent(model, temp_data, DEVICE)
    latent_xys['group 1'] = temp_latent_xy


    temp_data, _ = build_dataloader(group2, KMER_SIZE)
    temp_latent_xy = generate_latent(model, temp_data, DEVICE)
    latent_xys['group 2'] = temp_latent_xy
    #######################################
    # Manifold 
    #######################################
    # X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    #tsne_xy = TSNE(n_components=2).fit_transform(latent_xy)
    
    tsne_xy = MDS(n_components=2).fit_transform(latent_xy).T
    tsne_xy = convert_2d(tsne_xy)

    #######################################
    # Cluster
    #######################################
    nclusters = []
    # nclusters.extend(get_optimal_nclusters_bgmm(latent_xy, tsne_xy))
    # nclusters.extend(get_optimal_nclusters_gmm(latent_xy, tsne_xy))

    # print(nclusters)

    #######################################
    # plot
    #######################################

    title = (
            f'Frame:{idx+1}/{count}, '
            f'Seq/Kmer size:{SEQ_SIZE:,}/{KMER_SIZE:,}, '
            f'Diff %: {(group_subs_rate*100)}%, \n'
            f'Model:{model.__class__.__name__}, '
            f'Epochs:{EPOCHS}, '
            f'Mainfold:MDS'
            #f'Clusters: {"/".join(map(lambda x: str(x), nclusters))}, '
            )
    p = plot_sim_latent(latent_xys, tsne_xy, title)
    return p


if __name__ == '__main__':
    main()
