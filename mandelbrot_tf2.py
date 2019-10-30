import numpy as np
import tensorflow as tf
from PIL import Image
from utils import create_image

R = 4
ITER_NUM = 200

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

def mandelbrot(Z):
    xs = tf.constant(Z)
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    for i in range(ITER_NUM):
        zs = tf.where(tf.abs(zs) < R, zs**2 + xs, zs)
        not_diverged = tf.abs(zs) < R
        ns = ns + tf.cast(not_diverged, tf.float32)
    return zs, ns

if __name__ == "__main__":
    start_x = -2.5  # x range
    end_x = 1
    start_y = -1.2  # y range
    end_y = 1.2
    width = 1000  # image width
    step = (end_x - start_x) / width
    Y, X = np.mgrid[start_y:end_y:step, start_x:end_x:step]
    #Y, X = tf.meshgrid(tf.range(start_y, end_y, step), tf.range(start_x, end_x, step))
    Z = X + 1j * Y
    Z = Z.astype(np.complex64)

    zs, ns = mandelbrot(Z)
    final_step = ns.numpy()
    final_z = zs.numpy()

    img = create_image(final_z, final_step, R)
    img.save('img/mandelbrot.png')
