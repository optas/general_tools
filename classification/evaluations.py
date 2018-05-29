import numpy as np

def average_per_class(predicted_labels, gt_labels):    
    gt_labels = np.array(gt_labels)
    scores_per_class = []

    for c in np.unique(gt_labels):
        index_c = gt_labels == c
        n_class = float(np.sum(index_c))
        s = np.sum(gt_labels[index_c] == predicted_labels[index_c])
        s /= n_class
        scores_per_class.append(s)
    return np.mean(scores_per_class), scores_per_class