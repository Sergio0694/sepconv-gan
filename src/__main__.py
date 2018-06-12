import os
import train

os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6' # Enable all GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'      # See issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# debug run
train.run()
