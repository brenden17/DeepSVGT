import torch

import numpy as np
import sklearn

#from scipy import stats
#import scipy

# from sklearn import cluster
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN, Birch, OPTICS
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import linalg

from models import AnalysisData, ManifoldXY, Cluster, Group
# from models import IGNORE_RATIO, EST_ALTERNATE, EST_REFERENCE
from models import EST_ALTERNATE, EST_REFERENCE
from cnnvae import CNN2LayerAutoencoder, CNNAutoencoder, train
from datasets import build_dataloader#, kmer, BASE_SIZE


import warnings
warnings.filterwarnings('ignore') 

random_seed = 1
np.random.seed(random_seed)


EPS=0.4


def convert_2d(X):
    return np.array(list(zip(X[0], X[1])))


def convert_3d(X):
    return np.array(list(zip(X[0], X[1], X[2])))


# def get_distribution_area(scaled_xy, bgmm):
#     min_scaled_xy = np.amin(scaled_xy, axis=0)
#     max_scaled_xy = np.amax(scaled_xy, axis=0)
#     min_max_scaled_xy = max_scaled_xy - min_scaled_xy
#     scaled_xy_area = min_max_scaled_xy[0] * min_max_scaled_xy[1]

#     dist_area_percents = []
#     for n in range(bgmm.means_.shape[0]):
#         eig_vals, eig_vecs = np.linalg.eigh(bgmm.covariances_[n])
#         eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
#         dist_area = eig_vals[0] * eig_vals[1]

#         dist_area_percent = ((dist_area * 100) / scaled_xy_area)
#         dist_area_percents.append(round(dist_area_percent, 2))

#     return dist_area_percents


# def create_clusters(scaled_xy, bgmm, cluster_algo='bgmm'):
def create_clusters(scaled_xy, cluster_algo='bgmm'):
    # print('create_clusters.........................................................1111')
    min_scaled_xy = np.amin(scaled_xy, axis=0)
    max_scaled_xy = np.amax(scaled_xy, axis=0)
    min_max_scaled_xy = max_scaled_xy - min_scaled_xy
    scaled_xy_area = min_max_scaled_xy[0] * min_max_scaled_xy[1]

    clusters = []

    if cluster_algo == 'bgmm':
        from sklearn.mixture import BayesianGaussianMixture
        bgmm = BayesianGaussianMixture(n_components=2, n_init=50, random_state=0).fit(scaled_xy)

        for i in range(bgmm.means_.shape[0]):
            eig_vals, eig_vecs = np.linalg.eigh(bgmm.covariances_[i])
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
            dist_area = eig_vals[0] * eig_vals[1]

            dist_area_percent = ((dist_area * 100) / scaled_xy_area)
            c = Cluster(center_xy=bgmm.means_[i], dist_area_percent=dist_area_percent)
            clusters.append(c)
    elif cluster_algo == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++").fit(scaled_xy)

        for i in range(kmeans.cluster_centers_.shape[0]):
            c = Cluster(center_xy=kmeans.cluster_centers_[i], dist_area_percent=80)
            clusters.append(c)
    elif cluster_algo == 'BisectingKMeans':
        from sklearn.cluster import BisectingKMean
        bkmeams = BisectingKMeans(n_clusters=2, random_state=0).fit(scaled_xy)

        for i in range(bkmeams.cluster_centers_.shape[0]):
            c = Cluster(center_xy=bkmeams.cluster_centers_[i], dist_area_percent=80)
            clusters.append(c)
    elif cluster_algo == 'MeanShift':
        from sklearn.cluster import MeanShift
        ms = MeanShift(n_clusters=2, random_state=0).fit(scaled_xy)

        for i in range(ms.cluster_centers_.shape[0]):
            c = Cluster(center_xy=ms.cluster_centers_[i], dist_area_percent=80)
            clusters.append(c)
    elif cluster_algo == 'SpectralClustering':
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(scaled_xy)
        print(sc.labels_)

    elif cluster_algo == 'AgglomerativeClustering':
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=2).fit(scaled_xy)
        print(ac.labels_)
    elif cluster_algo == 'AffinityPropagation':
        from sklearn.cluster import AffinityPropagation
        ac = AffinityPropagation(random_state=0).fit(scaled_xy)
        print(ac.labels_)
    elif cluster_algo == 'DBSCAN':
        ac = DBSCAN(random_state=0).fit(scaled_xy)
        print(ac.labels_)
    else:
        pass

    return clusters


def create_clusters_bgmm(scaled_xy, bgmm):
    min_scaled_xy = np.amin(scaled_xy, axis=0)
    max_scaled_xy = np.amax(scaled_xy, axis=0)
    min_max_scaled_xy = max_scaled_xy - min_scaled_xy
    scaled_xy_area = min_max_scaled_xy[0] * min_max_scaled_xy[1]

    clusters = []

    for i in range(bgmm.means_.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(bgmm.covariances_[i])
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        dist_area = eig_vals[0] * eig_vals[1]

        dist_area_percent = ((dist_area * 100) / scaled_xy_area)
        c = Cluster(center_xy=bgmm.means_[i], dist_area_percent=dist_area_percent)
        clusters.append(c)

    return clusters

def create_clusters_kmeans(scaled_xy, kmeans):
    min_scaled_xy = np.amin(scaled_xy, axis=0)
    max_scaled_xy = np.amax(scaled_xy, axis=0)
    min_max_scaled_xy = max_scaled_xy - min_scaled_xy
    scaled_xy_area = min_max_scaled_xy[0] * min_max_scaled_xy[1]

    clusters = []

    for i in range(kmeans.cluster_centers_.shape[0]):
        c = Cluster(center_xy=kmeans.cluster_centers_[i], dist_area_percent=80)
        clusters.append(c)

    return clusters


def generate_xys(analysis_data):
    filter_option = analysis_data.filter_option
    SEQ_SIZE = analysis_data.sv.query_l
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DEVICE = 'cpu'
    LATENT_DIMS = 2
    # EPOCHS = filter_option["epochs"]
    EPOCHS = analysis_data.get_epochs()
    # print('EPOCHS----------------')
    # print(EPOCHS)

    all_reads = []
    all_reads_counter = []

    consesus_reads_from_bams = analysis_data.consesus_reads_from_bams
    # over_cutoff_coverage = True

    for bam_file in filter_option['bam_files']:
        all_reads_counter.append(f'{len(consesus_reads_from_bams[bam_file])}')
        all_reads.extend(consesus_reads_from_bams[bam_file])

        # check ratio, if under 0.1, do not need to cluster
        # if analysis_data.read_counter_ratios_bams[bam_file] > 0.1:
        #     over_cutoff_coverage = True

            
    # print(f'all_reads_counter:{"/".join(all_reads_counter)}')
    # print(f'{len(all_reads)} are fetched.')
    # print(f'read ratio:{analysis_data.read_counter_ratios_bams}')

    analysis_data.read_counter = len(all_reads)
    
    if analysis_data.read_counter == 0:
        return

    # if over_cutoff_coverage == False:
    #     analysis_data.over_cutoff_coverage = False
    #     analysis_data.set_genotypes_with_reads_attr()
    #     return

    # TODO data
    all_encoded_data, _ = build_dataloader(all_reads)

    # train model
    model = CNNAutoencoder(SEQ_SIZE, LATENT_DIMS).to(DEVICE)
    # model = CNN2LayerAutoencoder(SEQ_SIZE, LATENT_DIMS).to(DEVICE)

    trained_model = train(model, all_encoded_data, DEVICE, epochs=EPOCHS)

    latent_xy = generate_latent(trained_model, all_encoded_data, DEVICE)
    
    # to show dots from each bam_files
    coverage_from_bams = analysis_data.coverage_from_bams
    latent_xys = {}
    bams_over_cutoff_coverage = {}

    for bam_file in filter_option['bam_files']:
        # print('@@@@@@@@@@@@@@@00000000000000000')
        # print(consesus_reads_from_bams[bam_file])
        # print('coverage_from_bams')
        # print(coverage_from_bams[bam_file])
        count_coverage_bam = coverage_from_bams[bam_file]
        avg_count_coverage = sum(count_coverage_bam) / len(count_coverage_bam)
        # print(count_coverage_bam)
        # print(avg_count_coverage)

        # if consesus_reads_from_bams[bam_file] or avg_count_coverage < analysis_data.over_cutoff_coverage_from_bams[bam_file].get_is_pass():
        # print('analysis_data.over_cutoff_coverage_from_bams[bam_file].get_is_pass()')
        # print(analysis_data.over_cutoff_coverage_from_bams[bam_file].get_is_pass())
        # if consesus_reads_from_bams[bam_file] or analysis_data.over_cutoff_coverage_from_bams[bam_file].get_is_pass():
        # if consesus_reads_from_bams[bam_file] and analysis_data.over_cutoff_coverage_from_bams[bam_file].get_is_pass():
        if consesus_reads_from_bams[bam_file] and analysis_data.over_cutoff_coverage_from_bams[bam_file].is_pass:
            # print('@@@@@@@@@@@@@@@11111111111111111')
            bam_encoded_data, _ = build_dataloader(consesus_reads_from_bams[bam_file])
            latent_xys[bam_file] = generate_latent(trained_model, bam_encoded_data, DEVICE)
        else:
            latent_xys[bam_file] = []

    #raw_xy = ManifoldXY(name='Raw', all_xy=latent_xy, bams_xy=latent_xys)
    raw_xy = ManifoldXY(name='Raw')
    raw_xy.all_xy = latent_xy
    raw_xy.bams_xy = latent_xys
   
    analysis_data.add_raw_xy(raw_xy)


def generate_latent(model, data, device):
    latent_x = []
    latent_y = []
    latent_z = []

    for i, (x, y) in enumerate(data):
        mu, log_var = model.encoder(x.to(device))
        z = model.reparameterize(mu, log_var)
        z = z.to('cpu').detach().numpy()

        if z.shape[1] >= 2:
            latent_x.extend(z[:, 0])
            latent_y.extend(z[:, 1])
        if z.shape[1] >= 3:
            latent_z.extend(z[:, 2])

    if latent_z:
        return convert_3d([latent_x, latent_y, latent_z])

    return convert_2d([latent_x, latent_y, latent_z])


def get_manifold_xys(analysis_data):
    mds_bams_xy = {} 
    bams_xy = analysis_data.raw_xy.bams_xy
 
    for bam_file in bams_xy:
        xy = bams_xy[bam_file]
        # print('xy....')
        # print(xy)
        if len(xy) == 0:
            mds_bams_xy[bam_file] = []
            continue
        # MDS
        mds_xy = MDS(n_components=2, random_state=0, normalized_stress='auto').fit_transform(xy)
        #mds_bams_xy[bam_file] = np.array(mds_xy)
        mds_bams_xy[bam_file] = mds_xy

    # create new a manifold, MDS
    mds_manifold_xy = ManifoldXY('MDS')
    mds_manifold_xy.all_xy = np.array(MDS(n_components=2, random_state=0, normalized_stress='auto').fit_transform(analysis_data.raw_xy.all_xy))
    mds_manifold_xy.bams_xy = mds_bams_xy

    analysis_data.manifold_xys.append(mds_manifold_xy)


def bgmm(xy):
    bgmm = BayesianGaussianMixture(n_components=2, n_init=50, random_state=0).fit(xy)

    bgmm_weights = bgmm.weights_
    return -100


def dbscan(xy):
    db = DBSCAN(eps=EPS).fit(xy)
    labels = db.labels_
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return n_clusters


def birch(xy):
    birch = Birch(n_clusters=None).fit(xy)
    p = birch.predict(xy)

    n_clusters = len(set(p))
    return n_clusters


def optics(xy):
    optics = OPTICS(eps=0.4).fit(xy)
    l = optics.labels_

    n_clusters = len(set(l))
    return n_clusters



def estimate_cluster_count(analysis_data):
    cluster_algo = analysis_data.filter_option.get('cluster_algo', None)
    manifold_xys = analysis_data.manifold_xys

    # cluster_funcs = (dbscan, birch, optics, bgmm)
    # dbscan is the best
    cluster_funcs = (dbscan,) 
    est_clusters = []

    # find major estimated cluster count
    for manifold_xy in manifold_xys:
        if manifold_xy.name == 'Raw':
            continue
        scaled_xy = manifold_xy.scaled_xy

        est_clusters.extend([cluster_func(scaled_xy) for cluster_func in cluster_funcs])

    est_cluster_counter = est_clusters[0]
    # print(f'est_cluster_counter: {est_cluster_counter}')

    # check read ratio, if read ratio
    # read_counter_ratios = [analysis_data.read_counter_ratios_bams[bf] for bf in analysis_data.read_counter_ratios_bams]
    read_counter_ratios = [analysis_data.read_counters_bams[bf].group_ratio for bf in analysis_data.read_counters_bams]
    # read_counter_ratio = get_read_counterdd_ratio(analysis_data)

    # print(read_counter_ratios)
    read_counter_ratios_sum = sum(list((map(lambda x: 1 if x > analysis_data.IGNORE_RATIO else 0, read_counter_ratios))))
    # print('read_counter_ratios_sum')
    # print(read_counter_ratios_sum)

    major_n_bam_files = round(analysis_data.n_bam_files/2)

    # find centers
    for manifold_xy in manifold_xys:
        scaled_xy = manifold_xy.scaled_xy

        # if read ratio is higher than normal, check it again
        # if est_cluster_counter == 2 or read_counter_ratios_sum >= 1:
        if est_cluster_counter == 2 or read_counter_ratios_sum >= major_n_bam_files:
            # manifold_xy.clusters = create_clusters(scaled_xy, f)
            manifold_xy.clusters = create_clusters(scaled_xy)

            manifold_xy.est_cluster_counter = 2
            analysis_data.est_cluster_counter = 2
            if read_counter_ratios_sum > 0:
                analysis_data.force = True
        else:
            manifold_xy.est_cluster_counter = 1
            analysis_data.est_cluster_counter = 1


# def estimate_cluster_count_with_kmean(analysis_data):
#     print('estimate_cluster_count_with_kmean...')
#     manifold_xys = analysis_data.manifold_xys

#     # cluster_funcs = (dbscan, birch, optics, bgmm)
#     # dbscan is the best
#     cluster_funcs = (dbscan,) 
#     est_clusters = []

#     # find major estimated cluster count
#     for manifold_xy in manifold_xys:
#         if manifold_xy.name == 'Raw':
#             continue
#         scaled_xy = manifold_xy.scaled_xy

#         est_clusters.extend([cluster_func(scaled_xy) for cluster_func in cluster_funcs])

#     est_cluster_counter = est_clusters[0]
#     # print(f'est_cluster_counter: {est_cluster_counter}')

#     # check read ratio, if read ratio
#     read_counter_ratios = [analysis_data.read_counter_ratios_bams[bf] for bf in analysis_data.read_counter_ratios_bams]
#     # read_counter_ratio = get_read_counterdd_ratio(analysis_data)

#     # print(read_counter_ratios)
#     read_counter_ratios_sum = sum(list((map(lambda x: 1 if x > IGNORE_RATIO else 0, read_counter_ratios))))
#     # print('read_counter_ratios_sum')
#     # print(read_counter_ratios_sum)

#     # find centers
#     for manifold_xy in manifold_xys:
#         scaled_xy = manifold_xy.scaled_xy

#         # if read ratio is higher than normal, check it again
#         if est_cluster_counter == 2 or read_counter_ratios_sum >= 1:
#             kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++").fit(scaled_xy)

#             # manifold_xy.centers = kmeans.cluster_centers_
#             manifold_xy.clusters = create_clusters_kmeans(scaled_xy, kmeans)

#             manifold_xy.est_cluster_counter = 2
#             analysis_data.est_cluster_counter = 2
#             if read_counter_ratios_sum > 0:
#                 analysis_data.force = True
#         else:
#             manifold_xy.est_cluster_counter = 1
#             analysis_data.est_cluster_counter = 1



# def estimate_cluster_count_with_kmean2(analysis_data):
#     print('estimate_cluster_count_with_kmean...')
#     manifold_xys = analysis_data.manifold_xys

#     # cluster_funcs = (dbscan, birch, optics, bgmm)
#     # dbscan is the best
#     cluster_funcs = (dbscan,) 
#     est_clusters = []

#     # find major estimated cluster count
#     for manifold_xy in manifold_xys:
#         if manifold_xy.name == 'Raw':
#             continue
#         scaled_xy = manifold_xy.scaled_xy

#         est_clusters.extend([cluster_func(scaled_xy) for cluster_func in cluster_funcs])

#     est_cluster_counter = est_clusters[0]
#     # print(f'est_cluster_counter: {est_cluster_counter}')

#     # check read ratio, if read ratio
#     read_counter_ratios = [analysis_data.read_counter_ratios_bams[bf] for bf in analysis_data.read_counter_ratios_bams]
#     # read_counter_ratio = get_read_counterdd_ratio(analysis_data)

#     # print(read_counter_ratios)
#     read_counter_ratios_sum = sum(list((map(lambda x: 1 if x > IGNORE_RATIO else 0, read_counter_ratios))))
#     # print('read_counter_ratios_sum')
#     # print(read_counter_ratios_sum)

#     # find centers
#     for manifold_xy in manifold_xys:
#         scaled_xy = manifold_xy.scaled_xy

#         # if read ratio is higher than normal, check it again
#         if est_cluster_counter == 2 or read_counter_ratios_sum >= 1:
#             kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_xy)
#             manifold_xy.centers = kmeans.cluster_centers_

#             print(kmeans.cluster_centers_)

#             manifold_xy.est_cluster_counter = 2
#             analysis_data.est_cluster_counter = 2
#             if read_counter_ratios_sum > 0:
#                 analysis_data.force = True
#         else:
#             manifold_xy.est_cluster_counter = 1
#             analysis_data.est_cluster_counter = 1


# def estimate_cluster_count_with_bgmm(analysis_data):
#     print('estimate_cluster_count_with_bgmm...')
#     manifold_xys = analysis_data.manifold_xys

#     # cluster_funcs = (dbscan, birch, optics, bgmm)
#     # dbscan is the best
#     cluster_funcs = (dbscan,) 
#     est_clusters = []

#     # find major estimated cluster count
#     for manifold_xy in manifold_xys:
#         if manifold_xy.name == 'Raw':
#             continue
#         scaled_xy = manifold_xy.scaled_xy

#         est_clusters.extend([cluster_func(scaled_xy) for cluster_func in cluster_funcs])

#     # dbscan
#     est_cluster_counter = est_clusters[0]
#     print(f'est_cluster_counter: {est_cluster_counter}')

#     # check read ratio, if read ratio
#     read_counter_ratios = [analysis_data.read_counter_ratios_bams[bf] for bf in analysis_data.read_counter_ratios_bams]
#     # read_counter_ratio = get_read_counterdd_ratio(analysis_data)

#     # print(read_counter_ratios)
#     read_counter_ratios_sum = sum(list((map(lambda x: 1 if x > analysis_data.IGNORE_RATIO else 0, read_counter_ratios))))
#     print('read_counter_ratios_sum=====================================')
#     print(read_counter_ratios_sum)

#     # find centers
#     for manifold_xy in manifold_xys:
#         scaled_xy = manifold_xy.scaled_xy

#         # if read ratio is higher than normal, check it again
#         if est_cluster_counter == 2 or read_counter_ratios_sum >= 1:
#         # if est_cluster_counter == 2:

#             bgmm = BayesianGaussianMixture(n_components=2, n_init=50, random_state=0).fit(scaled_xy)

#             manifold_xy.clusters = create_clusters(scaled_xy)

#             manifold_xy.est_cluster_counter = 2
#             analysis_data.est_cluster_counter = 2
#             if read_counter_ratios_sum > 0:
#                 analysis_data.force = True
#         else:
#             print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
#             manifold_xy.est_cluster_counter = 1
#             analysis_data.est_cluster_counter = 1


def analysis_reads(analysis_data):
    # print(f'analysis_reads^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^{analysis_data.over_cutoff_coverage}')
    if analysis_data.read_counter == 0:
        return

    get_manifold_xys(analysis_data)

    estimate_cluster_count(analysis_data)

    analysis_read_groups(analysis_data)

    estimate_genotype(analysis_data)


def analysis_read_groups(analysis_data):
    if analysis_data.est_cluster_counter == 1:
        return

    manifold_xys = analysis_data.manifold_xys

    ignore_pvalue = analysis_data.IGNORE_ERROR


    for manifold_xy in manifold_xys:
        group = {}
        group_ratio = {}
        scaled_bams_xy = manifold_xy.scaled_bams_xy

        G1_alter_counter = 0
        G1_ref_counter = 0
        G2_alter_counter = 0
        G2_ref_counter = 0

        # find cluster's type such as alter or ref
        for bam_file in scaled_bams_xy:
            # print('bam_file....1111111111111')
            # print(bam_file)
            scaled_xy = scaled_bams_xy[bam_file]

            # print(scaled_xy)
            if len(scaled_xy) == 0:
                continue
            cluster1 = manifold_xy.clusters[0].center_xy
            cluster2 = manifold_xy.clusters[1].center_xy

            cluster1_dist = np.sqrt(np.sum((scaled_xy-cluster1)**2, axis=1)) # c1 distannce
            cluster2_dist = np.sqrt(np.sum((scaled_xy-cluster2)**2, axis=1)) # c2 distannce

            closed_cluster2 = cluster1_dist >= cluster2_dist # close to G2
            closed_cluster1 = cluster1_dist < cluster2_dist # close to G1
            # print('closed_cluster1')
            # print(bam_file)
            # print(len(analysis_data.consesus_reads_attr_from_bams[bam_file]))
            # print(sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster2]))
            # print(sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster2]==0))
            a = sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster2])
            G2_alter_counter += a
            c = sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster2]==0)
            G2_ref_counter += c
            # G2_ref_counter += (len(analysis_data.consesus_reads_attr_from_bams[bam_file])-a) 
            # print(sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster1]))
            # print(sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster1]==0))
            b = sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster1])
            G1_alter_counter += b
            d = sum(np.array(analysis_data.consesus_reads_attr_from_bams[bam_file])[closed_cluster1]==0)
            G1_ref_counter += d

        
        if abs(G1_alter_counter - G1_ref_counter) > abs(G2_alter_counter - G1_alter_counter):
            g1 = EST_ALTERNATE if G1_alter_counter > G1_ref_counter else EST_REFERENCE
            g2 = EST_REFERENCE if g1 == EST_ALTERNATE else EST_ALTERNATE
        else:
            g2 = EST_ALTERNATE if G2_alter_counter > G1_alter_counter else EST_REFERENCE
            g1 = EST_REFERENCE if g2 == EST_ALTERNATE else EST_ALTERNATE

        # print('###############')
        # print(G1_alter_counter, G1_ref_counter, G2_alter_counter, G2_ref_counter)
        # print(g1, g2)


        manifold_xy.clusters[0].est_type = g1
        manifold_xy.clusters[1].est_type = g2


        # find each group's type
        for bam_file in scaled_bams_xy:
            # print('bam_file%%%%%%%%%%%%%%%%%%')
            # print(bam_file)
            scaled_xy = scaled_bams_xy[bam_file]

            if len(scaled_xy) == 0:
                group[bam_file] = Group(0, '', 0, '', ignore_pvalue)
                continue

            cluster1 = manifold_xy.clusters[0].center_xy
            cluster1_est_type = manifold_xy.clusters[0].est_type
            cluster2 = manifold_xy.clusters[1].center_xy
            cluster2_est_type = manifold_xy.clusters[1].est_type

            cluster1_dist = np.sqrt(np.sum((scaled_xy-cluster1)**2, axis=1)) # c1 distannce
            cluster2_dist = np.sqrt(np.sum((scaled_xy-cluster2)**2, axis=1)) # c2 distannce

            closed_cluster1_count = sum(cluster1_dist < cluster2_dist) # close to G1
            closed_cluster2_count = sum(cluster1_dist >= cluster2_dist) # close to G2

            # g = Group(closed_cluster1_count, cluster1_est_type, closed_cluster2_count, cluster2_est_type)

            # group[bam_file] = (closed_cluster1_count, closed_cluster2_count) if closed_cluster2_count > closed_cluster1_count else (closed_cluster2_count, closed_cluster1_count)
            # group_ratio[bam_file] = 0 if closed_cluster1_count == 0 else group[bam_file][0]/(group[bam_file][1]+group[bam_file][0])
            # print(closed_cluster1_count, cluster1_est_type, closed_cluster2_count, cluster2_est_type)
            group[bam_file] = Group(closed_cluster1_count, cluster1_est_type, closed_cluster2_count, cluster2_est_type, ignore_pvalue)
            # group_ratio[bam_file] = 0 if closed_cluster1_count == 0 else group[bam_file][0]/(group[bam_file][1]+group[bam_file][0])


        manifold_xy.bams_group = group
        # manifold_xy.bams_group_ratio = group_ratio


def estimate_genotype(analysis_data):
    est_cluster_counter = analysis_data.est_cluster_counter

    manifold_xys = analysis_data.manifold_xys

    if est_cluster_counter == 1:
        # analysis_data.set_genotypes_with_reads_attr()
        analysis_data.estimate_genotypes_with_reads_attr()
        return

    # analysis_data.set_genotypes_with_reads_group_attr()
    analysis_data.estimate_genotypes_with_reads_group_attr()


def get_mainfold_xy_groups(analysis_data):
    if analysis_data.est_cluster_counter == 1:
        return {}

    mainfold_xys = analysis_data.manifold_xys
    
    mainfold_xy_groups = {}

    # [(1, 2), (1,2), (1,2)]
    for manifold_xy in mainfold_xys:
        group1 = []
        group2 = []

        for bam_file in manifold_xy.bams_group:
            group = manifold_xy.bams_group[bam_file]
            # print('!!!!!!!!!!!!!!')
            # print(bam_file)
            # print(group)
            group1.append(group.bottom_group_item.counter)
            group2.append(group.top_group_item.counter)

        mainfold_xy_groups[manifold_xy.name] = (group1, group2)

    return mainfold_xy_groups


def get_read_counters(analysis_data):
    read_counters_bams = analysis_data.read_counters_bams

    refs = []
    alters = []
    bam_labels = []

    for bam_file in read_counters_bams:
        ref_counter, alter_counter = read_counters_bams[bam_file].get_counters()
        refs.append(ref_counter)
        alters.append(alter_counter)
        bam_labels.append(bam_file)

    return refs, alters, bam_labels
