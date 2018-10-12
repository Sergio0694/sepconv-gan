import argparse
import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpu_ops import NEAREST_SHADER_MODULE

def preview(frame0_path, interpolated_path, frame1_path):

    # load the source images
    frame0_bgr = np.expand_dims(cv2.imread(frame0_path).astype(np.float32), 0)
    interpolated_bgr = np.expand_dims(cv2.imread(interpolated_path).astype(np.float32), 0)
    frame1_bgr = np.expand_dims(cv2.imread(frame1_path).astype(np.float32), 0)

    # apply the shader
    shaded_f32 = NEAREST_SHADER_MODULE(interpolated_bgr, frame0_bgr, frame1_bgr)
    shaded_uint8 = tf.cast(shaded_f32, tf.uint8)[0]
    with tf.Session() as session:
        return session.run(shaded_uint8)

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser(description='Previews the nearest shader between two input frames and an interpolated frame.')
    parser.add_argument('-source', help='The path of the first frame. Its filename must end with a sequence number, so that the script will ' \
                        'be able to automatically retrieve the path for the following frame. This script expects three frames, where the one' \
                        'at the center should be one processed by the network, with the _ suffix and the same sequence number as the first frame.', required=True)
    args = vars(parser.parse_args())

    # validate
    if not os.path.isfile(args['source']):
        print('The input file does not exist')
        exit(-1)
    match = re.findall('([0-9]+)[.](jpg|png|bmp)$', args['source'])
    if not match:
        print('The input path is not valid')
        exit(-1)
    index, extension = match[0]
    interpolated = re.sub('[0-9]+[.](?:jpg|png|bmp)$', '{:03d}_.{}'.format(int(index), extension), args['source'])
    if not os.path.isfile(interpolated):
        print('Couldn\'t find the interpolated frame for the one in input')
        exit(-1)
    following = re.sub('[0-9]+[.](?:jpg|png|bmp)$', '{:03d}.{}'.format(int(index) + 1, extension), args['source'])
    if not os.path.isfile(following):
        print('Couldn\'t find the following frame for the one in input')
        exit(-1)
    
    # process and display
    shaded_bgr = preview(args['source'], interpolated, following)
    shaded_rgb = cv2.cvtColor(shaded_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(shaded_rgb)
    plt.show()
