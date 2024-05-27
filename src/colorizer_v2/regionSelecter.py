import cv2
import numpy as np

class BaseRegionSelecter:
    def decide_next(self) -> np.ndarray:
        raise NotImplementedError
        return 
    def get_color_mask(self) -> np.ndarray:
        raise NotImplementedError
        return 
    
class RandomRegionSelecter(BaseRegionSelecter):
    def __init__(self, **kwargs):
        return
    
    def decide_next(self, color, segmentation_map, n_target=1):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        target_mask = np.zeros_like(gray)
        uncolored_labels = np.unique(segmentation_map[(segmentation_map > 0) & (gray == 255)])
        n_next = min(len(uncolored_labels), n_target)
        next_labels = np.random.choice(uncolored_labels, size=n_next, replace=False)
        for label in next_labels: 
            target_mask[(segmentation_map==label)] = 1
        target_mask[gray<250] = 0
        return target_mask
    
    def get_color_mask(self, color, segment_map=None):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        mask = ((color[:, :, 0] == color[:,:,1]) & (color[:,:,1]== color[:,:,2])).astype(np.uint8)
        
        if segment_map is None:
            mask[(mask==1) & (gray < 128)] = 0
        else:
            mask[segment_map==0] = 0
        return mask