'''
Created on November 17, 2018

@author: optas
'''

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
        print "Cannot open image-file '%s'" % img_file
        return
            
    image.thumbnail((desired_size, desired_size), Image.ANTIALIAS)

    new_size = image.size
    new_im = Image.new(im_type, (desired_size, desired_size), color=bg)
    new_im.paste(image, ((desired_size - new_size[0]) // 2, 
                         (desired_size - new_size[1]) // 2))

    assert(new_im.size == (desired_size, desired_size))
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
    widths, heights = zip(*(i.size for i in images))
    extra_pixels = (len(images) - 1) * x_pad
    total_width = sum(widths) + extra_pixels
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), color=bg_color)
    x_offset = 0 # first image is stack left-most
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + x_pad
    return new_im

# TODO def scale_aspect_ratio_preserving:

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