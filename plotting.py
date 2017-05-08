import numpy as np
from PIL import Image


def stack_images_horizontally(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    '''
    images = map(Image.open, file_names)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

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
    images = map(Image.open, file_names)
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    n_images = len(images)
    im_per_row = int(np.floor(np.sqrt(n_images)))
    total_width = im_per_row * max_width
    total_height = im_per_row * max_height
    new_im = Image.new('RGB', (total_width, total_height))

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