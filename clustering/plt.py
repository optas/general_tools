"""
Created on December 26, 2016

@author: optas
"""
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
from past.utils import old_div
from PIL import Image
from sklearn.manifold import TSNE

import itertools
import numpy as np
import matplotlib.pylab as plt
import cv2

from ..plotting.in_out import read_transparent_png

def plot_confusion_matrix(cm, classes, plt_nums=True, normalize=False,
                          cmap=plt.cm.Blues, figsize=(10, 10),
                          fontsize=20, save_file=None):
    """
    Make a figure of a confusion matrix as returned from sklearn.metrics.confusion_matrix
    :param cm:
    :param classes: list of strings corresponding to the class-labels
    :param plt_nums: plot numbers over heatmap figure
    :param normalize: the numbers shows will be the percentage of the confusion
    :param cmap:
    :param figsize:
    :param fontsize:
    :param save_file:
    :return:
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.xticks(np.arange(len(classes)), classes, rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(len(classes)), classes, fontsize=fontsize)

    if normalize:
        cm = old_div(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
        str_formatter = '{:0.2f}'
    else:
        str_formatter = '{}'

    if plt_nums:
        row_ranks = np.argsort(cm, axis=1)[:, -2:]
        for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
            if j == row_ranks[i][0]:
                color = 'red'
            elif j == row_ranks[i][1]:
                color = 'green'
            else:
                color = "black"
            plt.text(j, i, str_formatter.format(cm[i, j]),
                     horizontalalignment="center",
                     color=color, fontsize=fontsize/2)

    if save_file is not None:
        plt.savefig(save_file)


def plot_tsne(embedding, colors=None, words=None, figsize=(20, 20), scatter_size=20, **tsne_kwargs):
    '''Creates and TSNE model and plots it. Works only with 2D. TODO. Fix to work 3D TSNE.
    Example:
    tsne_kwargs={'perplexity':40, 'random_state': 42}
    m, f = plot_tsne(embediing, color=['r','r','r','r','b'], words=['a','b',3,4,5], **tsne_kwargs);
    '''
    # TSNE defaults
    for key, val in zip(['init', 'n_iter', 'random_state'], ['pca', 2500, None]):
        if key not in tsne_kwargs:
            tsne_kwargs[key] = val
    
    tsne_model = TSNE(**tsne_kwargs)
    print(tsne_model)
    
    new_emb = tsne_model.fit_transform(embedding)
    n_ex = len(new_emb)
        
    plt.figure(figsize=figsize)
    if colors is None:
        colors = ['b'] * n_ex
        
    fig = plt.figure()
    for i in range(n_ex):        
        plt.scatter(new_emb[i, 0], new_emb[i, 1], c=colors[i], s=scatter_size)
        if words is not None:
            plt.annotate(words[i], xy=(new_emb[i, 0], new_emb[i, 1]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom')
    return tsne_model, fig
        
        
def _scale_2d_embedding(two_dim_emb):
    two_dim_emb -= np.min(two_dim_emb, axis=0)  # scale x-y in [0,1]
    two_dim_emb /= np.max(two_dim_emb, axis=0)
    return two_dim_emb


def plot_2d_embedding_in_grid_greedy_way(two_dim_emb, image_files, big_dim=2500, small_dim=200, save_file=None, transparent=True):
    '''
    Input:
        two_dim_emb: (N x 2) numpy array: arbitrary 2-D embedding of data.
        image_files: (list) of strings pointing to images. Specifically image_files[i] should be an image associated with
                     the datum whose coordinates are given in two_dim_emb[i].
        big_dim:     (int) height of output 'big' grid rectangular image.
        small_dim:   (int) height to which each individual rectangular image/thumbnail will be resized.
    '''
    ceil = np.ceil
    mod = np.mod
    floor = np.floor
    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.ones((big_dim, big_dim, 3), dtype='uint8')
    
    if transparent:
        occupy_val = 255
        im_loader = read_transparent_png
    else:
        occupy_val = 0
        im_loader = cv2.imread
            
    out_image *= occupy_val 
    for i, im_file in enumerate(image_files):
        #  Determine location on grid
        a = ceil(x[i, 0] * (big_dim - small_dim) + 1)
        b = ceil(x[i, 1] * (big_dim - small_dim) + 1)
        a = int(a - mod(a - 1, small_dim) - 1)
        b = int(b - mod(b - 1, small_dim) - 1)
                
        if out_image[a, b, 0] != occupy_val:
            continue    # Spot already filled (drop=>greedy).
        
        fig = im_loader(im_file)
        fig = cv2.resize(fig, (small_dim, small_dim))
        try:
            out_image[a:a + small_dim, b:b + small_dim, :] = fig
        except:
                print('the code here fails. fix it.')
                print(a)
        continue

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)
    
    return out_image



def plot_2d_embedding_in_grid_forceful(two_dim_emb, image_files, big_dim=2500, small_dim=200, save_file=None):
    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.zeros((big_dim, big_dim, 3), dtype='uint8')
    N = two_dim_emb.shape[0]
    xnum = int(big_dim / float(small_dim))
    ynum = int(big_dim / float(small_dim))
    free = np.ones(N, dtype=np.bool)

    grid_2_img = np.ones((xnum, ynum), dtype='int') * -1
    res = float(small_dim) / float(big_dim)
    for i in range(xnum):
        for j in range(ynum):
            sorted_indices = np.argsort((x[:, 0] - i * res)**2 + (x[:, 1] - j * res)**2)
            possible = sorted_indices[free[sorted_indices]]

            if len(possible) > 0:
                picked = possible[0]
                free[picked] = False
                grid_2_img[i, j] = picked
            else:
                break

    for i in range(xnum):
        for j in range(ynum):
            if grid_2_img[i, j] > -1:
                im_file = image_files[grid_2_img[i, j]]
                fig = cv2.imread(im_file)
                fig = cv2.resize(fig, (small_dim, small_dim))
                try:
                    out_image[i * small_dim:(i + 1) * small_dim, j * small_dim:(j + 1) * small_dim, :] = fig
                except:
                    print('the code here fails. fix it.')
                    print(im_file)
                continue

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)

    return out_image
