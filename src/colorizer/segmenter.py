import cv2
import numpy as np
from scipy.interpolate import NearestNDInterpolator

class BaseSegmenter:
    """
    Base [Segmenter] to get segmentation map from line art.
    """
    def __call__(self, lineart) -> np.ndarray:
        raise NotImplementedError
        return
    

class SimpleSegmenter(BaseSegmenter):
    def __init__(self, connectivity=8, **kwargs):
        self.connectivity = connectivity
        return
    """
    A simple [Segmenter] class using 'cv2.connectedComponents()'
    """

    def __call__(self, lineart):
        return cv2.connectedComponents(lineart, connectivity=self.connectivity)

    """
    def __call__(self, lineart):
        
        downscaled = cv2.resize(lineart, dsize=(0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)
        num_regions, downscaled_segmap = cv2.connectedComponents(downscaled, connectivity=self.connectivity)
        
        segmap = cv2.resize(downscaled_segmap, dsize=(lineart.shape[-1], lineart.shape[-2]), interpolation=cv2.INTER_NEAREST)
        mask = np.where(~(segmap == 0))
        interp = NearestNDInterpolator(np.transpose(mask), segmap[mask])
        segmap = interp(*np.indices(segmap.shape))
        segmap[lineart==0] = 0
        return num_regions, segmap.astype(np.uint8)
    """