from enum import Enum
import cv2
import numpy as np

class OpticalFlowType(Enum):
    DIRECTIONAL = 0
    BIDIRECTIONAL = 1

def get_optical_flow_rgb(before, after, flow_type):
    '''Returns the appropriate optical flow estimation between two RB images.'''

    # convert to grayscale
    before_g = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_g = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    if flow_type == OpticalFlowType.DIRECTIONAL:
        return get_rgb_flow(before, before_g, after_g)
    else:
        return get_rgb_flow(before, before_g, after_g), get_rgb_flow(before, after_g, before_g)

def get_rgb_flow(original, before, after):
    '''Reads two grayscale images and returns the RGB optical flow between them.'''

    flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv = np.zeros_like(original)

    hsv[:, :, 0] = angle * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_bgr
