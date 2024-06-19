import os
from datetime import datetime
from collections import Counter

import numpy as np

from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering 
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def c2d(X):
    return np.array(list(zip(X[0], X[1])))

def mp(data):
#https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py
    n_neighbors = 10

    manifold_xys = [data]
    
    #MDS
    mds =  manifold.MDS(2, max_iter=100, n_init=1)
    d = mds.fit_transform(data)
    manifold_xys.append(d)

    #TSNE
    #tsne = manifold.TSNE(n_components=2, random_state=0)
    #d = tsne.fit_transform(data)
    #manifold_xys.append(d)

    # SE
    #se = manifold.SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
    #d = se.fit_transform(data)
    #manifold_xys.append(d)

    # ISOMAP
    #iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
    #d = iso.fit_transform(data)
    #manifold_xys.append(d)

    return manifold_xys


def ploting(manifold_xys):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('xx')

    ncols =len(manifold_xys) 
    width_ratios = [1 for _ in range(ncols)]
    
    gs = gridspec.GridSpec(2, ncols, width_ratios=width_ratios, height_ratios=[1, 1])

    # manifold plot
    for i, k in enumerate(manifold_xys):
        axes_3 = plt.subplot(gs[0, i])
        #axes_3.set_title(k + f' Cluster:{cluster_counters[k]}')
        axes_3.scatter(k[:, 0], k[:, 1], alpha=0.5)
        #_, center = read_weight_centers[k]
        #if cluster_counters[k] > 1:
        #    axes_3.scatter(center[0][0], center[0][1], marker="D", s=100, c='r')
        #    axes_3.scatter(center[1][0], center[1][1], marker="D", s=100, c='r')

   
    axes_3.legend()

    plt_img = os.path.join('./1phd/', datetime.now().strftime("%m-%d-%H-%M-%S") + '.png')
    plt.savefig(plt_img)
#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
def bgmm(xy):
    bgmm = BayesianGaussianMixture(n_components=2, n_init=50, random_state=42).fit(xy)

    bgmm_weights = bgmm.weights_
    print('np.round(bgmm_weights, 2)')
    print(np.round(bgmm_weights, 2))

def dbscan(xy):
    db = DBSCAN(eps=0.4).fit(xy)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("dbscan clusters: %d" % n_clusters_)


def birch(xy):
    b = Birch(n_clusters=None)
    b.fit(xy)
    l = b.predict(xy)

    s = len(set(l))
    print('birch :%d' % s)

def ap(xy):
    b = AffinityPropagation(random_state=5).fit(xy)
    b.fit(xy)
    l = b.predict(xy)

    s = len(set(l))
    print('ap :%d' % s)


def optics(xy):
    b = OPTICS(eps=0.4).fit(xy)
    b.fit(xy)
    l = b.labels_

    s = len(set(l))
    print('optics :%d' % s)

def sc(X):
    c = SpectralClustering(n_clusters=2,
                assign_labels='discretize',
                random_state=0).fit(X)
    l = c.labels_
    print(Counter(l).values())

def ac(X):
    c = AgglomerativeClustering().fit(X)
    l = c.labels_
    print(Counter(l).values())

#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
def clustering(manifold_xys):
    #manifold_xys = [StandardScaler().fit_transform(xy) for xy in manifold_xys]
    manifold_xys = [MinMaxScaler().fit_transform(xy) for xy in manifold_xys]

    ploting(manifold_xys)
    for xy in manifold_xys:
        print("########################################")
        bgmm(xy)
        dbscan(xy)
        birch(xy)
        optics(xy)
        #sc(xy)
        #ap(xy)
        #ac(xy)
        

def load_2d(fname=None):

    if not fname:
        #fname = './1phd/tmp8681.txt'
        fname = './1phd/tmp8681.txt' #1
        #fname = './1phd/tmp9227.txt' #1
        #fname = './1phd/tmp9900-2d.txt' #2
        #fname = './1phd/tmp5606-3d.txt'
        #fname = './1phd/tmp6606.txt'

    with open(fname, 'rb') as f:
        X = np.load(f)

    return X

if __name__ == '__main__':
    data = load_2d()
    
    manifold_xys = mp(data)
    clustering(manifold_xys)


    
