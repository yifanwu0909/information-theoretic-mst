import numpy as np
from time import time

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering

# Normalized mutual information is only available
# in the current development version. See if we can import,
# otherwise use dummy.

from sklearn.metrics import normalized_mutual_info_score

from tree_entropy import tree_information
from itm import ITM
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons


def do_experiments(X,y):

    n_clusters = len(np.unique(y))

    classes = [ITM(n_clusters=n_clusters, infer_dimensionality=False),
               KMeans(n_clusters=n_clusters)]
    names = ["ITM","KMeans"]

    for clusterer, method in zip(classes, names):
        clusterer.fit(X)
        y_pred = clusterer.labels_

        title = 'Clustering Using '+method
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
        plt.title(title)
        plt.show()




# The first dataset 
X, y = make_blobs(n_samples=300, centers=3, n_features=2,cluster_std=2)
do_experiments(X,y)

# The second dataset 
X, y = make_circles(n_samples=300, noise=0.05,factor=.5)
do_experiments(X,y)

# The third dataset 
X, y = make_moons(n_samples=300, noise=0.1)
do_experiments(X,y)
