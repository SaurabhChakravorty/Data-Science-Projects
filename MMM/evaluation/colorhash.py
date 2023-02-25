"""
Doesn't follow usual code structure of this project.
Taken from: https://bitbucket.org/fk/python-color-hash/src/default/
[i] Doesn't exist in Conda
[i] Removed comments
"""

from __future__ import division

import sys
from binascii import crc32
from numbers import Number

PY2 = sys.version_info[0] <= 2


def crc32_hash(obj):
    if PY2:
        bs = str(obj)
    else:
        bs = str(obj).encode('utf-8')
    return crc32(bs) & 0xffffffff


def hsl2rgb(hsl):
    try:
        h, s, l = hsl
    except TypeError:
        raise ValueError(hsl)
    try:
        h /= 360
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
    except TypeError:
        raise ValueError(hsl)

    rgb = []
    for c in (h + 1 / 3, h, h - 1 / 3):
        if c < 0:
            c += 1
        elif c > 1:
            c -= 1

        if c < 1 / 6:
            c = p + (q - p) * 6 * c
        elif c < 0.5:
            c = q
        elif c < 2 / 3:
            c = p + (q - p) * 6 * (2 / 3 - c)
        else:
            c = p
        rgb.append(round(c * 255))

    return tuple(rgb)


def rgb2hex(rgb):
    try:
        return '#%02x%02x%02x' % rgb
    except TypeError:
        raise ValueError(rgb)


def color_hash(obj, hashfunc=crc32_hash,
               lightness=(0.35, 0.5, 0.65), saturation=(0.35, 0.5, 0.65),
               min_h=None, max_h=None):
    if isinstance(lightness, Number):
        lightness = [lightness]
    if isinstance(saturation, Number):
        saturation = [saturation]

    if min_h is None and max_h is not None:
        min_h = 0
    if min_h is not None and max_h is None:
        max_h = 360

    hash = hashfunc(obj)
    h = (hash % 359)
    if min_h is not None and max_h is not None:
        h = (h / 1000) * (max_h - min_h) + min_h
    hash //= 360
    s = saturation[hash % len(saturation)]
    hash //= len(saturation)
    l = lightness[hash % len(lightness)]

    return (h, s, l)


class ColorHash:

    def __init__(self, *args, **kwargs):
        self.hsl = color_hash(*args, **kwargs)

    @property
    def rgb(self):
        return hsl2rgb(self.hsl)

    @property
    def hex(self):
        return rgb2hex(self.rgb)
