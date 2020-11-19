import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


# Read data
def read_data():
    data = np.loadtxt("mouse.csv", delimiter=",")

    return data


# Scale numerical data
def scaling(data):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    scaled_data = sc.fit_transform(data)

    return scaled_data


# Draw a scatter plot to visualize the clustering result
def c_scatter(data, c, title):
    x = data[:, 0]
    y = data[:, 1]

    plt.scatter(x=x, y=y, c=c)
    plt.title(title)
    # plt.show()
    plt.savefig(title + '.png')


# Use K-means model to cluster dataset.
def k_means(data):
    # hyper-parameter
    params = {
        'n_cluster': [3, 4, 5, 6, 8, 10],
        'max_iter': [50, 100, 200, 300]
    }

    # Try various combinations of parameters
    for n_cluster in params['n_cluster']:
        for max_iter in params['max_iter']:
            kmeans = KMeans(n_clusters=n_cluster, max_iter=max_iter)
            c = kmeans.fit_predict(data)
            title = 'K-Means (n_clusters={}, max_iter={})'.format(n_cluster, max_iter)
            c_scatter(data, c, title)


# Use DBSCAN model to cluster dataset.
def dbscan(data):
    # hyper-parameter
    params = {
        'eps': [0.05, 0.1, 0.2, 0.5, 1],
        'min_samples': [3, 5, 10, 15, 20, 30, 50, 100, 200, 300]
    }

    # Try various combinations of parameters
    # DBSCAN will output an array of -1’s and 0’s, where -1 indicates an outlier.
    for eps in params['eps']:
        for min_sample in params['min_samples']:
            dbs = DBSCAN(eps=eps, min_samples=min_sample)
            c = dbs.fit_predict(data)
            title = 'DBSCAN (eps={}, min_samples={})'.format(eps, min_sample)
            c_scatter(data, c, title)


# Use gaussian mixture model to cluster dataset.
def em(data):
    # hyper-parameter
    params = {
        'n_components': [2, 3, 4, 5, 6, 7],
        'max_iter': [10, 30, 50, 100, 200]
    }

    # Try various combinations of parameters
    for n_components in params['n_components']:
        for max_iter in params['max_iter']:
            em = GaussianMixture(n_components=n_components, max_iter=max_iter)
            c = em.fit_predict(data)
            title = 'EM (n_components={}, max_iter={})'.format(n_components, max_iter)
            c_scatter(data, c, title)


data = read_data()
sc_data = scaling(data)
k_means(sc_data)
dbscan(sc_data)
em(data)
