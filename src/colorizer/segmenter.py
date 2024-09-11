import cv2
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from src.colorizer.utils import *
from skimage.morphology import skeletonize, dilation

def get_skeleton(region_map):
    Xp = np.pad(region_map, [[0, 1], [0, 0], [0, 0]], 'symmetric').astype(np.float32)
    Yp = np.pad(region_map, [[0, 0], [0, 1], [0, 0]], 'symmetric').astype(np.float32)
    X = np.sum((Xp[1:, :, :] - Xp[:-1, :, :]) ** 2.0, axis=2) ** 0.5
    Y = np.sum((Yp[:, 1:, :] - Yp[:, :-1, :]) ** 2.0, axis=2) ** 0.5
    edge = np.zeros_like(region_map)[:, :, 0]
    edge[X > 0] = 255
    edge[Y > 0] = 255
    edge[0, :] = 255
    edge[-1, :] = 255
    edge[:, 0] = 255
    edge[:, -1] = 255
    skeleton = 1.0 - dilation(edge.astype(np.float32) / 255.0)
    skeleton = skeletonize(skeleton)
    skeleton = (skeleton * 255.0).clip(0, 255).astype(np.uint8)
    field = np.random.uniform(low=0.0, high=255.0, size=edge.shape).clip(0, 255).astype(np.uint8)
    field[skeleton > 0] = 255
    field[edge > 0] = 0
    filter = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],
        dtype=np.float32) / 5.0
    height = np.random.uniform(low=0.0, high=255.0, size=field.shape).astype(np.float32)
    for _ in range(512):
        height = cv2.filter2D(height, cv2.CV_32F, filter)
        height[skeleton > 0] = 255.0
        height[edge > 0] = 0.0
    return height.clip(0, 255).astype(np.uint8)

def get_regions(skeleton_map):
    marker = skeleton_map[:, :, 0]
    normal = topo_compute_normal(marker) * 127.5 + 127.5
    marker[marker > 100] = 255
    marker[marker < 255] = 0
    labels, nil = label(marker / 255)
    water = cv2.watershed(normal.clip(0, 255).astype(np.uint8), labels.astype(np.int32)) + 1
    water = thinning(water)
    all_region_indices = find_all(water)
    regions = np.zeros_like(skeleton_map, dtype=np.uint8)
    regions = np.repeat(regions, 3, axis=-1)
    for region_indices in all_region_indices:
        regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
    return nil, water


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
    #def __call__(self, lineart):
    #    if lineart.ndim==2:
    #        lineart = np.repeat(lineart[:,:,None], 3, axis=-1)
    #    skeleton = get_skeleton(lineart)
    #    return get_regions(skeleton[:,:,None])


    

class SkeletonSegmenter(BaseSegmenter):
    def __init__(self):
        return
    def __call__(self, lineart):
        if lineart.ndim==2:
            lineart = np.repeat(lineart[:,:,None], 3, axis=-1)
        skeleton = get_skeleton(lineart)
        return get_regions(skeleton[:,:,None])