'''
Created on Nov 8, 2016

@author: optas
'''

import numpy as np
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
import itertools

def bench_clustering(estimator, name, gt_labels):
    '''Compares a clustering produced by an algorithm like kmeans or spectral clustering
    with a ground-truth clustering under some popular fitness metrics.
    '''    
    print('%30s   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name,
             metrics.homogeneity_score(gt_labels, estimator.labels_),
             metrics.completeness_score(gt_labels, estimator.labels_),
             metrics.v_measure_score(gt_labels, estimator.labels_),
             metrics.adjusted_rand_score(gt_labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(gt_labels,  estimator.labels_),)
         )

def kmeans_cluster_pseudo_probability(estimator):
    cluster_dists = squareform(pdist(estimator.cluster_centers_))
    cluster_sims = 1.0 / 1.0 + cluster_dists
    np.fill_diagonal(cluster_sims, 0)    
    divisor = np.sum(cluster_sims, 0)
    divisor[divisor == 0] = 1
    cluster_probs = cluster_sims / divisor
    return cluster_probs

def spectral_cluster_pseudo_probability(estimator):
    labels = estimator.labels_
    class_ids = np.unique(labels)
    n_classes = len(class_ids)
    similarity = np.zeros((n_classes, n_classes))
    affinity = estimator.affinity_matrix_ 
    for c1, c2 in itertools.combinations(class_ids, 2):
        # Sum edge weights accross clusters (double counts an edge)
        similarity[c1, c2] = np.sum((labels == c1) * affinity * (labels == c2).T) / 2.0 
    divisor = np.sum(similarity, 0)
    divisor[divisor == 0] = 1
    cluster_prob = similarity / divisor
    return cluster_prob


