from enum import Enum
import cv2
import numpy as np

class OpticalFlowEmbeddingType(Enum):
    NONE = 0
    DIRECTIONAL = 1
    BIDIRECTIONAL = 2
    BIDIRECTIONAL_PREWARPED = 3

def prewarp_frame(flow, frame):
    '''Warps a frame given its optical flow data.

    flow(np.array) -- the optical flow data
    frame(np.array) -- the image to warp
    '''

    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(frame, flow, None, cv2.INTER_LINEAR)

def get_bidirectional_prewarped_frames(before, after):
    '''Returns a pair of warped frames, from A to B and from B to A.

    before(np.array) -- the first frame
    after(np.array) -- the second frame
    '''

    before_g = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_g = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    flow_forward = cv2.calcOpticalFlowFarneback(before_g, after_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_backward = cv2.calcOpticalFlowFarneback(after_g, before_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return prewarp_frame(flow_forward, before), prewarp_frame(flow_backward, after)

def get_optical_flow_from_rgb(before, after, flow_type):
    '''Returns the appropriate optical flow estimation between two RB images.'''

    # convert to grayscale
    before_g = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_g = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    if flow_type == OpticalFlowType.DIRECTIONAL:
        return get_optical_flow_from_grayscale(before_g, after_g)
    if flow_type == OpticalFlowType.BIDIRECTIONAL: 
        return get_optical_flow_from_grayscale(before_g, after_g), get_optical_flow_from_grayscale(after_g, before_g)
    raise ValueError('Invalid flow type')

def get_optical_flow_from_grayscale(before, after):
    '''Reads two grayscale images and returns the RGB optical flow between them.'''

    flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h = cv2.normalize(flow[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
    v = cv2.normalize(flow[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
    return h, v
