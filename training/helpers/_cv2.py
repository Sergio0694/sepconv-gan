from enum import Enum
import cv2
import numpy as np

class OpticalFlowType(Enum):
    DIRECTIONAL = 0
    BIDIRECTIONAL = 1

def get_optical_flow_from_rgb(before, after, flow_type):
    '''Returns the appropriate optical flow estimation between two RB images.'''

    # convert to grayscale
    before_g = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_g = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    if flow_type == OpticalFlowType.DIRECTIONAL:
        return get_optical_flow_from_grayscale(before_g, after_g)
    return get_optical_flow_from_grayscale(before_g, after_g), get_optical_flow_from_grayscale(after_g, before_g)

def get_optical_flow_from_grayscale(before, after):
    '''Reads two grayscale images and returns the RGB optical flow between them.'''

    flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

    return np.expand_dims(angle * 180 / np.pi / 2, -1), np.expand_dims(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX), -1)
