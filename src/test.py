import dataset_builder
import image_helper
import cv2_helper
import os
import tensorflow as tf
import numpy as np
import dataset.training_dataset as train

path = 'E:\\qBittorrent'

TRAINING_DATASET_PATH = 'D:\\ML\\frames\\data'
IMAGE_DIFF_THRESHOLD = 1000
BATCH_SIZE = 2

cv2_helper.diff('D:\\ML\\frames\\data\\v2_s0_113.jpg', 'D:\\ML\\frames\\data\\v2_s0_116.jpg') # OK
cv2_helper.diff('D:\\ML\\frames\\data\\v3_s0_006.jpg', 'D:\\ML\\frames\\data\\v2_s0_116.jpg') # NO
cv2_helper.diff('D:\\ML\\frames\\data\\v0_s0_064.jpg', 'D:\\ML\\frames\\data\\v0_s0_065.jpg') # NO

pipeline = train.load(TRAINING_DATASET_PATH, IMAGE_DIFF_THRESHOLD, BATCH_SIZE)

print(pipeline)

it = pipeline.make_initializable_iterator()
el = it.get_next()

with tf.Session() as sess:
    sess.run(it.initializer)
    r = sess.run(el)
    print(r[0].shape, r[1].shape)

#print(cv2_helper.diff('{}\\{}'.format(output, 'v0_s0_001.jpg'), '{}\\{}'.format(output, 'v0_s0_005.jpg'))) # almost same
#print(cv2_helper.diff('{}\\{}'.format(output, 'v1_s0_064.jpg'), '{}\\{}'.format(output, 'v1_s0_068.jpg'))) # change
#print(cv2_helper.diff('{}\\{}'.format(output, 'v1_s0_066.jpg'), '{}\\{}'.format(output, 'v1_s0_070.jpg'))) # change
print('DONE')