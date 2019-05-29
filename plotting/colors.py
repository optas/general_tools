__author__ = "Panos Achlioptas"


import colorsys
import numpy as np
import matplotlib.cm as cm

from .. arrays import is_true

def rgb_to_hex_string(r, g, b):
    all_ints = is_true.is_integer(r) and is_true.is_integer(g) and is_true.is_integer(b)
    in_range = np.all(np.array([r, g, b]) <= 255) and np.all(np.array([r, g, b]) >= 0)
    if not all_ints or not in_range:
        raise ValueError('Expects integers in [0, 255]')

    return "#{0:02x}{1:02x}{2:02x}".format(int(r), int(g), int(b))


def scalars_to_colors(float_vals, colormap='jet'):
    cmap = cm.get_cmap(colormap)
    mappable = cm.ScalarMappable(cmap=cmap)
    colors = mappable.to_rgba(float_vals)
    return colors


def hsl_to_hsv(color):
    """
    >>> hsl_to_hsv((120, 100, 50))
    (120.0, 100.0, 100.0)
    >>> hsl_to_hsv((0, 100, 100))
    (0.0, 0.0, 100.0)
    Saturation in HSV is undefined and arbitrarily 0 for black:
    >>> hsl_to_hsv((240, 100, 0))
    (240.0, 0.0, 0.0)
    NOTE: 
    Taken from : https://github.com/futurulus/colors-in-context/blob/master/colorutils.py
    """
    hi, si, li = [float(d) for d in color]
    ho = hi
    si *= (li / 100.0) if li <= 50.0 else (1.0 - li / 100.0)
    vo = li + si
    so = (200.0 * si / vo) if vo else 0.0
    return (ho, so, vo)


def hsv_to_rgb(color):
    """color is a triplet (H, S, V) for hue in degrees (max=360) 
    and saturation and value in(0..100%)
    """    
    c = (color[0] / 360.0, color[1] / 100.0, color[2] / 100.0)
    c = colorsys.hsv_to_rgb(*c)
    c = tuple(int(round(i * 255)) for i in c)
    return c


def hsl_to_rgb(color):
    """color is a triplet (H, S, L) for hue in degrees (max=360) 
    saturation and value (0..100%)"""
    c = (color[0] / 360.0, color[2] / 100.0, color[1] / 100.0)
    c = colorsys.hls_to_rgb(*c)
    c = tuple(int(round(i * 255.0)) for i in c)
    return c


def rgb_to_hsv(rgb):
    rgb_0_1 = [d / 255.0 for d in rgb[:3]]
    hsv_0_1 = colorsys.rgb_to_hsv(*rgb_0_1)
    return tuple(d * r for d, r in zip(hsv_0_1, [360.0, 100.0, 100.0]))