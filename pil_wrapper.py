"""
Created on November 17, 2018

@author: optas
"""

from __future__ import print_function

import warnings
from builtins import zip
from PIL import Image


def square_image(img_file, desired_size, im_type='RGB', bg='white'):
    ''' Will load and resize the image to a square one, while keeping the original
    aspect ratio and add padding if necessary to achieve this.
    Input:
        im_type: (string) PIL compatible image-type, e.g, RGBA, LA, L.
    '''
    try:
        image = Image.open(img_file)
    except IOError:
        print("Cannot open image-file '%s'" % img_file)
        return
            
    w, h = image.size
    if w < desired_size or h < desired_size:
        warning.warn('Image has a side with smaller size than the desires_size, use ```resize_image_keep_aspect```.') 
    
    image.thumbnail((desired_size, desired_size), Image.ANTIALIAS)

    new_size = image.size
    new_im = Image.new(im_type, (desired_size, desired_size), color=bg)
    new_im.paste(image, ((desired_size - new_size[0]) // 2, 
                         (desired_size - new_size[1]) // 2))

    assert(new_im.size == (desired_size, desired_size))
    return new_im


def center_with_padding(image, new_width, new_height, mode='RGB', bg='white'):
    ''' Places the image in the center of a "frame" that contains padded pixels 
    to meet the desired size.   
    '''
    w, h = image.size
    if w > new_width or h > new_height:
        raise ValueError('Padding does not make sense. Image is bigger than specified size.')
            
    new_size = image.size    
    new_im = Image.new(mode, (new_width, new_height), color=bg)
    new_im.paste(image, ((new_width - w) // 2, 
                         (new_height - h) // 2))
    
    assert(new_im.size == (new_width, new_height))
    return new_im


def alpha_to_rgb(image, bg_color=(255, 255, 255)):
    """Convert RGBA Image to RGB.
    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    bg_color -- Tuple r, g, b (default 255, 255, 255)
    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, bg_color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def stack_images_horizontally(images, x_pad=0, bg_color='white'):
    ''' 
        images: list with PIL Images
        x_pad: pixels separating each stacked image        
    '''    
    widths, heights = list(zip(*(i.size for i in images)))

    extra_pixels = (len(images) - 1) * x_pad
    total_width = sum(widths) + extra_pixels
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), color=bg_color)
    x_offset = 0 # first image is stack left-most
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + x_pad
    return new_im

    
def resize_image_keep_aspect(image, fixed_dim, force_min=True, resample=Image.ANTIALIAS):
    """ 
    The height or the width of the resulting image will  be `fixed_dim`.
    If force_min, then the side that is smaller will become `fixed_dim`.  
    """
    
    initial_width, initial_height = image.size

    # Take the greater value, and use it for the ratio
    if force_min:    
        min_ = min([initial_width, initial_height])
        ratio = min_ / float(fixed_dim)
    else:
        max_ = max([initial_width, initial_height])
        ratio = max_ / float(fixed_dim)
        
    new_width = int((initial_width / ratio))
    new_height = int((initial_height / ratio))
    
    image = image.resize((new_width, new_height), resample=resample)                     
    return image
    
# def png_img_to_rgb(img_file):
#     im = Image.open(img_file)    
#     IM = np.array(im, dtype=np.float32)
#     IM /=  255.0
#     print 'y'
#     all_white = np.ones((224, 224), np.float32)
    
#     R = IM[:,:, 0]
#     G = IM[:,:, 1]
#     B = IM[:,:, 2]
#     A = IM[:,:, 3]

#     R = R * A  + (1-A * all_white)
#     G = G * A  + (1-A * all_white)
#     B = B * A  + (1-A * all_white)
#     im = np.stack([R, G, B])
#     im = im.transpose([1, 2, 0])
#     return im