__author__ = "Panos Achlioptas"

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from PIL import Image
from matplotlib import transforms


from . colors import scalars_to_colors

def stack_images_horizontally(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    '''
    images = list(map(Image.open, file_names))
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_file is not None:
        new_im.save(save_file)
    return new_im


def stack_images_in_square_grid(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new grid-square image that plots them in individual cells.
    The behavior is as expected when the sizes of the images are the same.
    '''
    images = list(map(Image.open, file_names))
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    max_height = max(heights)
    n_images = len(images)
    im_per_row = int(np.floor(np.sqrt(n_images)))
    total_width = im_per_row * max_width
    total_height = im_per_row * max_height
    new_im = Image.new('RGBA', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    in_row = 0

    for im in images:
        if in_row == im_per_row:
            y_offset += im.size[1]
            x_offset = 0
            in_row = 0

        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        in_row += 1

    y_offset += im.size[1]

    if save_file is not None:
        new_im.save(save_file)
    return new_im



def colored_text(in_text, scores=None, colors=None, colormap='jet', for_html=False, space_char=' '):
    if colors is None:
        colors = scalars_to_colors(scores, colormap)
        
    codes = []
    for c in colors:
        codes.append(mcolors.to_hex(c))   # color as hex.
    
    res = ''
    if for_html:
        for token, code in zip(in_text, codes):
            if len(res) > 0:
                res += space_char # add space                
            res += '<span style="color:{};"> {} </span>'.format(code, token)
    else:
        res = codes
    return res


def colored_text_to_figure(in_text, scores=None, colors=None, figsize=(10, 0.5), colormap='jet', **kw):
    """
    Input: in_text: (list) of strings
            scores: same size list/array of floats, if None: colors arguement must be not None.
            colors: if not None, it will be used instead of scores.
    """
    fig = plt.figure(frameon=False, figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    t = plt.gca().transData

    if colors is None:
        colors = scalars_to_colors(scores, colormap)

    for token, col in zip(in_text, colors):
        text = plt.text(0, 0, ' ' + token + ' ', color=col, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')
    return fig
