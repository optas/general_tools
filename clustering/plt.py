'''
Created on December 26, 2016

@author: optas
'''

import itertools
import numpy as np
import matplotlib.pylab as plt
import cv2
from PIL import Image


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def _scale_2d_embedding(two_dim_emb):
    two_dim_emb -= np.min(two_dim_emb, axis=0)  # scale x-y in [0,1]
    two_dim_emb /= np.max(two_dim_emb, axis=0)
    return two_dim_emb


def plot_2d_embedding_in_grid_greedy_way(two_dim_emb, image_files, big_dim=2500, small_dim=200, save_file=None):
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
    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.zeros((big_dim, big_dim, 3), dtype='uint8')

    for i, im_file in enumerate(image_files):
        #  Determine location on grid
        a = ceil(x[i, 0] * (big_dim - small_dim) + 1)
        b = ceil(x[i, 1] * (big_dim - small_dim) + 1)
        a = int(a - mod(a - 1, small_dim) + 1)
        b = int(b - mod(b - 1, small_dim) + 1)

        if out_image[a, b, 0] != 0:
            continue    # Spot already filled.

        fig = cv2.imread(im_file)
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
    used = np.zeros(N, dtype=np.bool)

    grid_2_img = np.ones((xnum, ynum), dtype='int') * -1
    res = float(small_dim) / float(big_dim)
    for i in xrange(xnum):
	for j in xrange(ynum):
		sorted_indices = np.argsort( (x[:,0] - i * res) ** 2 + (x[:,1] - j * res) ** 2)
		k = 0
		while used[sorted_indices[k]]:
			k  = k + 1
		used[sorted_indices[k]] = True
		grid_2_img[i,j] = sorted_indices[k]
    
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
# MATLAB CODE
#     S = 2000; % size of final image
#     G = zeros(S, S, 3, 'uint8');
#     s = 50; % size of every image thumbnail
#     
#     xnum = S/s;
#     ynum = S/s;
#     used = false(N, 1);
#     
#     qq=length(1:s:S); // x or y contains small s
#     abes = zeros(qq*2,2); // In abes every row contains x,y of left-high corner
#     i=1;
#     for a=1:s:S
#         for b=1:s:S
#             abes(i,:) = [a,b];
#             i=i+1;
#         end
#     end
#     %abes = abes(randperm(size(abes,1)),:); % randperm
#     
#     for i=1:size(abes,1)
#         a = abes(i,1);
#         b = abes(i,2);
#         %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
#         %yf = ((b-1)/S - 0.5)/2 + 0.5;
#         xf = (a-1)/S;
#         yf = (b-1)/S;
#         dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
#         dd(used) = inf; % dont pick these
#         [dv,di] = min(dd); % find nearest image
#     
#         used(di) = true; % mark as done
#         I = imread(fs{di});
#         if size(I,3)==1, I = cat(3,I,I,I); end
#         I = imresize(I, [s, s]);
#     
#         G(a:a+s-1, b:b+s-1, :) = I;
#     
#         if mod(i,100)==0
#             fprintf('%d/%d\n', i, size(abes,1));
#         end
#     end
#     
#     imshow(G);    
=======

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)

    return out_image
>>>>>>> bf9425ba935f4d3240f44eb608af50906219c011
