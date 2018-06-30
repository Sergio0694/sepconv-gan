import argparse
import numpy as np
import tensorflow as tf
from ops import SEPCONV_MODULE

def test_sepconv():

    # setup
    image = np.random.uniform(0.0, 1.0, [1, 12, 12, 3])
    kv = np.random.uniform(-1.0, 1.0, [1, 12, 12, 5])
    kh = np.random.uniform(-1.0, 1.0, [1, 12, 12, 5])
    with tf.Session():
        image_t = tf.constant(image, tf.float32)
        kv_t = tf.constant(kv, tf.float32)
        kh_t = tf.constant(kh, tf.float32)
        y = SEPCONV_MODULE.sepconv(image_t, kv_t, kh_t)

        # dC/dkv
        error = tf.test.compute_gradient_error(kv_t, kv_t.shape, y, image_t.shape, kv)
        print('dC/dkv error: {}'.format(error))
        assert error < 1e-5

        error = tf.test.compute_gradient_error(kh_t, kh_t.shape, y, image_t.shape, kh)
        print('dC/dkh error: {}'.format(error))
        assert error < 1e-5

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser(description='Tests the gradient of a custom op.')
    parser.add_argument('-op', help='The name of the op to test', required=True)
    args = vars(parser.parse_args())

    # exec
    test = {
        'sepconv': test_sepconv
    }.get(args['op'])
    if not test:
        print('The op {} does not exist'.format(args['op']))
        exit(-1)
    test()