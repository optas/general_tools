'''
Created on November 17, 2018

@author: optas
'''

from PIL import Image


def square_image(img_file, desired_size, im_type='RGB', bg='white'):
    ''' Will load and resize the image to a square one, while keeping the original
    aspect ratio and add padding if necessary to achieve this.
    Input:
        im_type: (string) PIL compatible image-type, e.e.g, RGBA, LA, L.
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

# TODO def scale_aspect_ratio_preserving: