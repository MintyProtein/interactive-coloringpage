import importlib
import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x

def show_segmap(segmentation_map, n_labels=None):
    assert segmentation_map.ndim == 2
    if n_labels is None:
        n_labels = len(np.unique(segmentation_map))
  
    map_color = np.zeros_like(segmentation_map)[:, :, None]
    map_color = np.concatenate((map_color, map_color, map_color), axis=2)
    
    for i in range(1, n_labels):
        map_color[segmentation_map==i] = [int(j) for j in np.random.randint(0,255,3)]
        
    plt.imshow(map_color)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
