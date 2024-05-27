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
