import numpy as np
import tensorflow as tf
from PIL import Image
from moviepy.editor import ImageSequenceClip
import os

R = 4
ITER_NUM = 200

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
    return clip



def color(z, i):
    # From: https://www.reddit.com/r/math/comments/2abwyt/smooth_colour_mandelbrot/
    if abs(z) < R:
        return 0, 0, 0
    v = np.log2(i + R - np.log2(np.log2(abs(z)))) / 5
    if v < 1.0:
        return v**4, v**2.5, v
    else:
        v = max(0, 2 - v)
        return v, v**1.5, v**3

def julia_set(Z, p):
    xs = tf.constant(Z.astype(np.complex64))
    zs = tf.Variable(xs)

    ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    for i in range(ITER_NUM):
        zs = tf.where(tf.abs(zs) < R, zs**2 + 0.7885*tf.cast(tf.exp(p), tf.complex64), zs)
        not_diverged = tf.abs(zs) < R
        ns = ns + tf.cast(not_diverged, tf.float32)

    final_step = ns.numpy()
    final_z = zs.numpy()

    r, g, b = np.frompyfunc(color, 2, 3)(final_z, final_step)
    img_array = np.dstack((r, g, b))
    return Image.fromarray(np.uint8(img_array * 255))

if __name__ == "__main__":
    start_x = -2.5  # x range
    end_x = 2.5
    start_y = -2.5  # y range
    end_y = 2.5
    width = 1000  # image width
    step = (end_x - start_x) / width
    Y, X = np.mgrid[start_y:end_y:step, start_x:end_x:step]
    Z = X + 1j * Y

    n = 30
    seqs = np.zeros([n] + list(Z.shape) + [3])
    for i, p in enumerate(np.linspace(0, 2*np.pi, n)):
        print(i, p)
        p = tf.Variable(1j * p)
        img = julia_set(Z, p)
        seqs[i, :, :] = np.array(img)


    gif('julia.gif', seqs, 8)