import dataset_builder
import image_helper
import cv2_helper
import os
import tensorflow as tf
import numpy as np

path = 'E:\\qBittorrent'

TRAINING_DATASET_PATH = 'D:\\ML\\frames\\data'
IMAGE_DIFF_THRESHOLD = 1000

cv2_helper.diff('D:\\ML\\frames\\data\\v2_s0_113.jpg', 'D:\\ML\\frames\\data\\v2_s0_116.jpg') # OK
cv2_helper.diff('D:\\ML\\frames\\data\\v3_s0_006.jpg', 'D:\\ML\\frames\\data\\v2_s0_116.jpg') # NO
cv2_helper.diff('D:\\ML\\frames\\data\\v0_s0_064.jpg', 'D:\\ML\\frames\\data\\v0_s0_065.jpg') # NO
exit(0)
l = os.listdir(TRAINING_DATASET_PATH)
print(len(l))

groups = [
    l[i:i + 5]
    for i in range(len(l) - 4)
]
print(len(groups))
print(groups[0])

def ensure_same_video_origin(paths):
    parts1, parts2 = str(paths[0]).split('_'), str(paths[-1]).split('_')
    return parts1[0] == parts2[0] and parts1[1] == parts2[1]

x = tf.data.Dataset.from_tensor_slices(groups) \
    .shuffle(len(groups)) \
    .filter(lambda g: tf.py_func(ensure_same_video_origin, inp=[g], Tout=[tf.bool])) \
    .map(lambda g: tf.py_func(cv2_helper.load_images, inp=[g, TRAINING_DATASET_PATH], Tout=[tf.float32])) \
    .filter(lambda g: tf.py_func(cv2_helper.ensure_difference_threshold, inp=[g, IMAGE_DIFF_THRESHOLD], Tout=[tf.bool])) \
    .batch(1)
  #  
   # 
  #  
  #  
    

print(x)

it = x.make_initializable_iterator()
el = it.get_next()

with tf.Session() as sess:
    sess.run(it.initializer)
    r = sess.run(el)
    print(r[0].shape)

#print(cv2_helper.diff('{}\\{}'.format(output, 'v0_s0_001.jpg'), '{}\\{}'.format(output, 'v0_s0_005.jpg'))) # almost same
#print(cv2_helper.diff('{}\\{}'.format(output, 'v1_s0_064.jpg'), '{}\\{}'.format(output, 'v1_s0_068.jpg'))) # change
#print(cv2_helper.diff('{}\\{}'.format(output, 'v1_s0_066.jpg'), '{}\\{}'.format(output, 'v1_s0_070.jpg'))) # change
print('DONE')