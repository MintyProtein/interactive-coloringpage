import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from src.colorizer.processing import *
from src.colorizer.utils import instantiate_from_config
from src.colorizer.regionSelecter import *
from src.colorizer.colorizer import *
from src.colorizer.segmenter import *

class InteractiveColoringPipeLine:
    def __init__(self, colorizer: BaseColorizer,
                 segmenter: BaseSegmenter, 
                 regionSelecter: BaseRegionSelecter, 
                 resolution=512):        
        
        self.segmenter = segmenter
        self.colorizer = colorizer
        self.regionSelecter = regionSelecter
        
        self.resolution = resolution

        self._lineart = None
        self._num_regions = None
        self._segmentation_map = None
        self._color = None
    
    def from_config(config):
        segmenter = instantiate_from_config(OmegaConf.to_object(config.segmenter_config))
        colorizer = instantiate_from_config(OmegaConf.to_object(config.colorizer_config))
        regionSelecter = instantiate_from_config(OmegaConf.to_object(config.regionSelecter_config))
        
        return InteractiveColoringPipeLine(colorizer=colorizer,
                                           segmenter=segmenter,
                                           regionSelecter=regionSelecter,
                                           resolution=config.resolution)
    
    def set_lineart(self, lineart: np.ndarray, description_prompt = "", verbose=True) -> None:
        self._original_lineart_shape = lineart.shape
        self._lineart, self._pad_info = preprocess_lineart(lineart, self.resolution)
        self.num_regions, self._segmentation_map = self.segmenter(self._lineart)
        
        self._color = HWC3(self._lineart)
        self._lineart_mask = (self._lineart[:, :, None].copy() / 255).astype(np.uint8)
        self._description_prompt = description_prompt
        
        if verbose:
            print(f"Succesfully set new lineart. # of Segmentation Regions: {self.num_regions}")
  
    def update_color(self, new_color: np.ndarray) -> None:
        new_color = preprocess_color(new_color, self.resolution)
        self._color = new_color
        
    def user_turn_coloring(self, user_coloring: np.ndarray) -> None:
        assert user_coloring.shape == self._color.shape
        new_color = user_coloring.copy()
        new_color[self._lineart_mask] = 0
        self.update_color(new_color)
    
    def robot_turn_coloring(self, n_target) -> np.ndarray:
        target_mask = self.regionSelecter.decide_next(self._color, self._segmentation_map, n_target)
        color_mask = self.regionSelecter.get_color_mask(self._color, self._segmentation_map)
        robot_coloring = self.colorizer.colorize_target(self._color,
                                                        target_mask,
                                                        self._description_prompt,
                                                        color_mask=color_mask)[0].astype(np.uint8)
        return robot_coloring
        
        
        
        
