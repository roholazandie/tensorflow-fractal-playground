import numpy as np
import tensorflow as tf
from utils import gif, mp4, create_image
from tqdm import tqdm

R = 4
n_iteration = 200


# @tf.function
# def julia_set(zs, ns, phase):
#     for _ in range(n_iteration):
#         zs = tf.where(tf.abs(zs) < R, zs ** 2 + 0.7885 * tf.cast(tf.exp(phase), tf.complex64), zs)
#         not_diverged = tf.abs(zs) < R
#         ns = ns + tf.cast(not_diverged, tf.float32)
#     return ns, zs

def julia_set(Z, phase):
    zs = tf.Variable(Z)
    ns = tf.Variable(tf.zeros_like(Z, tf.float32))
    for _ in range(n_iteration):
        zs = tf.where(tf.abs(zs) < R, zs ** 2 + 0.7885 * tf.cast(tf.exp(phase), tf.complex64), zs)
        not_diverged = tf.abs(zs) < R
        ns = ns + tf.cast(not_diverged, tf.float32)
    return ns, zs


def julia_set_np(zs, phase):
    ns = np.zeros_like(Z, dtype=np.float32)
    for i in range(n_iteration):
        zs = np.where(np.abs(zs) < R, zs**2 + 0.7885 * np.exp(phase).astype(np.complex64), zs)
        not_diverged = np.abs(zs) < R
        ns = ns + not_diverged.astype(np.float32)

    return ns, zs


if __name__ == "__main__":
    n = 2
    start_x = -2.5  # x range
    end_x = 2.5
    start_y = -2.5  # y range
    end_y = 2.5
    width = 3000  # image width
    step = (end_x - start_x) / width
    Y, X = np.mgrid[start_y:end_y:step, start_x:end_x:step]
    Z = X + 1j * Y
    Z = Z.astype(np.complex64)

    import time
    t1 = time.time()
    seqs = np.zeros([n] + list(Z.shape) + [3])
    for i, phase in enumerate(tqdm(np.linspace(0, 2 * np.pi, n))):
        phase = tf.Variable(1j * phase)
        ns, zs = julia_set(Z, phase)
        final_step = ns.numpy()
        final_z = zs.numpy()
        img = create_image(final_z, final_step, R)
        seqs[i, :, :] = np.array(img)
    t2 = time.time()
    #gif('julia', seqs, 8)
    mp4('julia', seqs, 8)

    t3 = time.time()
    seqs = np.zeros([n] + list(Z.shape) + [3])
    for i, phase in enumerate(tqdm(np.linspace(0, 2 * np.pi, n))):
        ns, zs = julia_set_np(Z, 1j*phase)
        final_step = ns
        final_z = zs
        img = create_image(final_z, final_step, R)
        seqs[i, :, :] = np.array(img)
    t4 = time.time()
    mp4('julianp', seqs, 8)

    print(t2-t1)
    print(t4-t3)