from moviepy.editor import ImageSequenceClip
import os
import numpy as np
from PIL import Image

def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    clip.write_videofile(fname+".mp4", fps=fps)
    return clip



def mp4(filename, array, fps=10, scale=1.0):
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname+".mp4"

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_videofile(filename, fps=fps)
    return clip


def color(z, i, R):
    # From: https://www.reddit.com/r/math/comments/2abwyt/smooth_colour_mandelbrot/
    if abs(z) < R:
        return 0, 0, 0
    v = np.log2(i + R - np.log2(np.log2(abs(z)))) / 5
    if v is np.nan:
        a=1
    if v < 1.0:
        return v**4, v**2.5, v
    else:
        v = max(0, 2 - v)
        return v, v**1.5, v**3


def create_image(z, step, R):
    r, g, b = np.frompyfunc(color, 3, 3)(z, step, R)
    img_array = np.dstack((r, g, b))
    img = Image.fromarray(np.uint8(img_array * 255))
    return img