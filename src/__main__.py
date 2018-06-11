import tensorflow as tf
from __MACRO__ import *
import dataset.training_dataset as train

pipeline = train.load(TRAINING_DATASET_PATH, IMAGE_DIFF_THRESHOLD, BATCH_SIZE)
print(pipeline)

it = pipeline.make_initializable_iterator()
el = it.get_next()

with tf.Session() as sess:
    sess.run(it.initializer)
    r = sess.run(el)
    print(r[0].shape, r[1].shape)
