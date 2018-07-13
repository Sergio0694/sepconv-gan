from enum import Enum
import os
from datetime import datetime
import networks.generators.deep_motion_sepconv as nn
from helpers._cv2 import OpticalFlowEmbeddingType

# paths
TRAINING_DATASET_PATH = '/media/sergio/SSD/ML/th/datasets/720p'
TEST_DATASET_PATH = '/media/sergio/SSD/ML/th/datasets/test_1080'

# preprocessing parameters
TRAINING_IMAGES_SIZE = 160
IMAGE_DIFF_MIN_THRESHOLD = 220
IMAGE_MEAN_VARIANCE = 16
MAX_FLOW = 24
IMAGES_WINDOW_SIZE = 1
FLOW_MODE = OpticalFlowEmbeddingType.NONE
INPUT_CHANNELS = {
    OpticalFlowEmbeddingType.NONE: 6,
    OpticalFlowEmbeddingType.DIRECTIONAL: 8,
    OpticalFlowEmbeddingType.BIDIRECTIONAL: 10,
    OpticalFlowEmbeddingType.BIDIRECTIONAL_PREWARPED: 12
}.get(FLOW_MODE)

# generator loss
class LossType(Enum):
    L1 = 0
    L2 = 1
    PERCEPTUAL = 2
    L1_PERCEPTUAL = 3
    L2_PERCEPTUAL = 4
GENERATOR_LOSS_TYPE = LossType.L1_PERCEPTUAL
PERCEPTUAL_LOSS_ENABLED = GENERATOR_LOSS_TYPE.value > 1
L_LOSS_FACTOR = 0.8
PERCEPTUAL_LOSS_FACTOR = 1.0
DISCRIMINATOR_LOSS_FACTOR = 1.0
LUMINANCE_LOSS_FACTOR = 0.05

# generator parameters
INITIAL_GENERATOR_LR = 0.0006
GENERATOR_LR_DECAY_RATE = 0.99
GENERATOR_ADAM_OPTIMIZER = True
GENERATOR_GRADIENT_CLIP = 5.0
NETWORK_BUILDER = nn.get_network_v1

# discriminator
DISCRIMINATOR_ENABLED = True
DISCRIMINATOR_LR = 0.0001
DISCRIMINATOR_GRADIENT_CLIP = 5.0

# training parameters
TRAINING_TOTAL_SAMPLES = 10000000
TENSORBOARD_LOG_INTERVAL = 10000
BATCH_SIZE = 6

# debug
VERBOSE_MODE = True
TRAINING_PROGRESS_BAR_LENGTH = 10
SHOW_TEST_SAMPLES_INFO_ON_LOAD = False
SHOW_TENSORS_LIST = False
TENSORBOARD_ROOT_DIR = '/media/sergio/Misc/tensorboard'
MODEL_ID = '{}.{}_({})'.format(nn.__name__.split('.')[-1], NETWORK_BUILDER.__name__, datetime.now().strftime('%d-%m-%Y_%H-%M'))
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_ROOT_DIR, MODEL_ID)
MACRO_PATH = __file__
MAX_MODELS_TO_KEEP = 1
