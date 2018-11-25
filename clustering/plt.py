'''
Created on December 26, 2016

@author: optas
'''

import itertools
import numpy as np
import matplotlib.pylab as plt
import cv2
from PIL import Image
from general_tools.plotting import read_transparent_png
from sklearn.manifold import TSNE

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(10, 10), fontsize=20, save_file=None, plt_nums=False):
    '''This function prints and plots the confusion matrix.'''
    plt.figure(figsize=figsize)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=80)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if plt_nums:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize)    
    plt.tight_layout()

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
    print tsne_model
    
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
                print 'the code here fails. fix it.'
                print a
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
    for i in xrange(xnum):
        for j in xrange(ynum):
            sorted_indices = np.argsort((x[:, 0] - i * res)**2 + (x[:, 1] - j * res)**2)
            possible = sorted_indices[free[sorted_indices]]

            if len(possible) > 0:
                picked = possible[0]
                free[picked] = False
                grid_2_img[i, j] = picked
            else:
                break

    for i in xrange(xnum):
        for j in xrange(ynum):
            if grid_2_img[i, j] > -1:
                im_file = image_files[grid_2_img[i, j]]
                fig = cv2.imread(im_file)
                fig = cv2.resize(fig, (small_dim, small_dim))
                try:
                    out_image[i * small_dim:(i + 1) * small_dim, j * small_dim:(j + 1) * small_dim, :] = fig
                except:
                    print 'the code here fails. fix it.'
                    print im_file
                continue

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)

    return out_image
