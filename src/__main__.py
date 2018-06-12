import os
import train
from __MACRO__ import *
from dataset.dataset_loader import calculate_samples_data
from helpers.debug_tools import calculate_image_difference

os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6' # Enable all GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'      # See issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# validate test samples
_, groups, _ = calculate_samples_data(TEST_DATASET_PATH, 1) # TODO: make the window size a macro too
for group in groups:
    _, _, error = calculate_image_difference('{}\\{}'.format(TEST_DATASET_PATH, group[0]), '{}\\{}'.format(TEST_DATASET_PATH, group[-1]))
    assert IMAGE_DIFF_MIN_THRESHOLD < error < IMAGE_DIFF_MAX_THRESHOLD

# debug run
train.run()
