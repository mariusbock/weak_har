# ------------------------------------------------------------------------
# All clustering related functions
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
import numpy as np
import os
import matplotlib
import scipy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture


def apply_gmm(clu_feat, k, cov_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
              weights_init=None, means_init=None, precisions_init=None, seed=1, warm_start=False, outlier_removal=False):
    
    """
    Applies Gaussian Mixture Model (GMM) clustering to the given feature data.

    Args:
    - clu_feat: numpy array, shape (n_samples, n_features)
        The input feature data to be clustered.
    - k: int
        The number of clusters to create.
    - cov_type: str, optional (default='full')
        The type of covariance parameters to use in the GMM. Possible values are 'full', 'tied', 'diag', and 'spherical'.
    - tol: float, optional (default=1e-3)
        The convergence threshold for the EM algorithm.
    - reg_covar: float, optional (default=1e-6)
        The regularization added to the diagonal of the covariance matrices.
    - max_iter: int, optional (default=100)
        The maximum number of iterations for the EM algorithm.
    - n_init: int, optional (default=1)
        The number of initializations to perform. The best result is kept.
    - init_params: str, optional (default='kmeans')
        The method used to initialize the weights, means, and covariances of the GMM. Possible values are 'kmeans' and 'random'.
    - weights_init: array-like, shape (n_components,), optional (default=None)
        The initial weights of the GMM components.
    - means_init: array-like, shape (n_components, n_features), optional (default=None)
        The initial means of the GMM components.
    - precisions_init: array-like, shape (n_components, n_features, n_features), optional (default=None)
        The initial precisions of the GMM components.
    - seed: int, optional (default=1)
        The random seed used for initialization.
    - warm_start: bool, optional (default=False)
        If True, the initialization is performed using the previous fit as initialization.
    - outlier_removal: bool, optional (default=False)
        If True, outliers are removed from the input feature data before clustering.

    Returns:
    - lbls: numpy array, shape (n_samples,)
        The cluster labels assigned to each input feature vector.
    - dist: numpy array, shape (n_samples,)
        The distances of each input feature vector to its respective cluster center.
    - gmm: GaussianMixture object
        The trained GMM model.

    """
    # define the GMM model and fit it to the data
    print("Applying GMM clustering ...")
    gmm = GaussianMixture(n_components=k, covariance_type=cov_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params, weights_init=weights_init, means_init=means_init,precisions_init=precisions_init, random_state=seed, warm_start=warm_start).fit(clu_feat)
    
    # compute distances (source code: https://stackoverflow.com/questions/47412749/how-can-i-get-a-representative-point-of-a-gmm-cluster)
    centers = np.empty(shape=(gmm.n_components, clu_feat.shape[1]))
    for i in range(gmm.n_components):
        # find the point in the cluster with the highest density
        # the density is calculated as the log of the pdf of the multivariate normal distribution
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(clu_feat)
        # the center of the cluster is the point with the highest density
        centers[i, :] = clu_feat[np.argmax(density)]
    # compute the pairwise distances between the cluster centers and the points in the dataset
    pair_dist = pairwise_distances(centers, clu_feat)

    # assign each embedding vector to a cluster
    lbls = gmm.predict(clu_feat)

    # compute the distances of each point to its respective cluster center
    idx = np.array(range(len(lbls)))
    dist = []
    for (id, lbl) in zip(idx, lbls):
        dist.append(pair_dist[lbl, id])
    return lbls, np.array(dist), gmm


def sort_idx_and_dist_to_centroid(dist, n_clusters):
    """
    Sorts the indices and distances to the centroid for each cluster.

    Args:
        dist (numpy.ndarray): The distance matrix of shape (n_clusters, n_samples).
        n_clusters (int): The number of clusters.

    Returns:
        tuple: A tuple containing the sorted index array and the sorted distance array.
            idx_array (numpy.ndarray): The sorted index array of shape (n_clusters, n_samples).
            dist_array (numpy.ndarray): The sorted distance array of shape (n_clusters, n_samples).
    """
    idx_array = np.empty(dist.shape)
    dist_array = np.empty(dist.shape)
    for c in range(n_clusters):
        c_dist = np.argsort(dist[c, :])
        sorted_dist = dist[c, :][c_dist]
        idx_array[c, :] = c_dist
        dist_array[c, :] = sorted_dist
    return idx_array, dist_array


def sample_cluster_feat(cluster_labels, cluster_dist, samples=10, clusters=30, sampling_strategy=None):
    """
    Samples cluster features based on the given cluster labels and distances.

    Args:
        cluster_labels (numpy.ndarray): Array of cluster labels.
        cluster_dist (numpy.ndarray): Array of cluster distances.
        samples (int, optional): Number of samples to be selected from each cluster. Defaults to 10.
        clusters (int, optional): Number of clusters. Defaults to 30.
        sampling_strategy (str, optional): Sampling strategy to be used. Can be 'distance', 'random', or None.
            If 'distance', samples are selected based on the distance within each cluster.
            If 'random', samples are randomly selected from the entire dataset.
            If None, all samples are selected. Defaults to None.

    Returns:
        numpy.ndarray: Boolean mask indicating the selected samples.

    """
    output = np.stack((cluster_labels, cluster_dist), axis=1)
    s_mask = np.zeros(output.shape[0], dtype=bool)
    if sampling_strategy == 'distance':
        s_idx = None
        output = np.concatenate((np.array(range(len(output)))[:, None], output), axis=1)
        for c in range(clusters):
            if s_idx is None:
                temp = output[output[:, 1] == c]
                s_idx = temp[temp[:, -1].argsort()][:samples, 0].astype(int)
            else:
                temp = output[output[:, 1] == c]
                temp = temp[temp[:, -1].argsort()][:samples, 0].astype(int)
                s_idx = np.concatenate((s_idx, temp))
        s_mask[s_idx] = True
    elif sampling_strategy == 'random':
        s_idx = np.random.choice(output.shape[0], size=samples * clusters, replace=False)
        s_mask[s_idx] = True
    else:
        s_mask[:] = True
    return s_mask


def create_correlation_matrix(activities, clusters, n_clusters, n_classes, distances=None, normalize_by_distance=False):
    """
    Create a correlation matrix based on the given activities and clusters.

    Args:
        activities (numpy.ndarray): The activities array.
        clusters (numpy.ndarray): The clusters array.
        n_clusters (int): The number of clusters.
        n_classes (int): The number of classes.
        distances (numpy.ndarray, optional): The distances array. Defaults to None.
        normalize_by_distance (bool, optional): Whether to normalize the correlation matrix by distance. Defaults to False.

    Returns:
        numpy.ndarray: The normalized correlation matrix.
    """

    print("Creating Cluster-to-Activity Matrix...")
    dets = np.concatenate((np.argmax(activities, axis=1)[:, None], clusters[:, None]), axis=1)
    corr_frame = pd.DataFrame(columns=list(range(n_classes)), index=list(range(n_clusters)))
    corr_frame = corr_frame.fillna(0)
    for c in range(n_clusters):
        if normalize_by_distance:
            fil_dets = dets[dets[:, 1] == c]
            fil_dist = distances[dets[:, 1] == c]
            for det, dis in zip(fil_dets, fil_dist):
                corr_frame.iloc[c, int(det[0])] += dis
        else:
            fil_dets = dets[dets[:, 1] == c]
            for det in fil_dets:    
                corr_frame.iloc[c, int(det[0])] += 1
    
    return normalize(corr_frame, axis=1, norm='l1')

