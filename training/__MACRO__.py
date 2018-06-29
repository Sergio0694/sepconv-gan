import os
from datetime import datetime
import networks.generators.deep_motion_cnn as nn
from helpers._cv2 import OpticalFlowEmbeddingType

# paths
TRAINING_DATASET_PATH = '/media/sergio/SSD/ML/th/datasets/480p'
TEST_DATASET_PATH = '/media/sergio/SSD/ML/th/datasets/test_1080'

# preprocessing parameters
TRAINING_IMAGES_SIZE = 240
IMAGE_DIFF_MAX_THRESHOLD = 3000
IMAGE_DIFF_MIN_THRESHOLD = 22
IMAGE_MIN_VARIANCE_THRESHOLD = 8
MAX_FLOW = 6
IMAGES_WINDOW_SIZE = 1
FLOW_MODE = OpticalFlowEmbeddingType.NONE
INPUT_CHANNELS = {
    OpticalFlowEmbeddingType.NONE: 6,
    OpticalFlowEmbeddingType.DIRECTIONAL: 8,
    OpticalFlowEmbeddingType.BIDIRECTIONAL: 10,
    OpticalFlowEmbeddingType.BIDIRECTIONAL_PREWARPED: 12
}.get(FLOW_MODE)

# training parameters
INITIAL_GENERATOR_LR = 0.0001
GENERATOR_LR_DECAY_RATE = 0.995
GENERATOR_ADAM_OPTIMIZER = True
DISCRIMINATOR_LR = 0.00001
TRAINING_TOTAL_SAMPLES = 10000000
TENSORBOARD_LOG_INTERVAL = 10000
BATCH_SIZE = 32
DISCRIMINATOR_ACTIVATION_EPOCH = 1
GENERATOR_GRADIENT_CLIP = 5.0
DISCRIMINATOR_GRADIENT_CLIP = 5.0
NETWORK_BUILDER = nn.get_network_v2

# debug
VERBOSE_MODE = True
TRAINING_PROGRESS_BAR_LENGTH = 10
SHOW_TEST_SAMPLES_INFO_ON_LOAD = True
TENSORBOARD_ROOT_DIR = '/home/sergio/Documents/tensorboard'
MODEL_ID = '{}.{}_({})'.format(nn.__name__.split('.')[-1], NETWORK_BUILDER.__name__, datetime.now().strftime('%d-%m-%Y_%H-%M'))
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_ROOT_DIR, MODEL_ID)
MAX_MODELS_TO_KEEP = 1
