import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn import cluster

def calc_spectral_clust(datasets, default_base):

    plot_num = 1
    
    for i_dataset, (dataset, algo_params) in enumerate(datasets):

        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # Create cluster objects
        algorithm = cluster.SpectralClustering(
            n_clusters=params['n_clusters'],
            affinity=params['affinity']
        )


        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        y_pred = algorithm.labels_.astype(int)

        plt.subplot(2, 3, plot_num)

        name = "Spectral Clustering"
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))

        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

    plt.show()
