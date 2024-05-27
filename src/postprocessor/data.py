import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.dropout.functional import cutout

class LineErasing(ImageOnlyTransform):
    def __init__(self, max_holes, max_height, max_width, fill_value, always_apply=False, p=1):
        super(LineErasing, self).__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value
    
    def get_holes(self, image):
        if image.ndim == 3:
            C, H, W = image.shape
        elif image.ndim == 2:
            H, W = image.shape
            
        num_holes = np.random.randint(0, self.max_holes)
        holes = []
        line_pixels = np.argwhere(image < 150)
        
        for i in range(num_holes):
            key = np.random.randint(0,len(line_pixels))
            key_h = line_pixels[key][0]
            key_w = line_pixels[key][1]
            
            hole_height = np.random.randint(1, self.max_height)
            hole_width = np.random.randint(1, self.max_width)
            
            x2 = np.max((0, key_w + hole_width//2))
            x1 = np.min((W, x2 - hole_width))
            y2 = np.max((0, key_h + hole_height//2))
            y1 = np.min((H, y2 - hole_height))
            holes.append((x1, y1, x2, y2))
        
        return holes

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.max_holes == 0:
            return img
        holes = self.get_holes(img)
        return cutout(img, holes, self.fill_value)


class PostprocessorDataset(Dataset):
    def __init__(self, img_paths, max_holes):
        self.img_paths = img_paths
        self.augmentation = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90()
        ])
        self.downscale_and_noise = A.Compose([
            LineErasing(max_holes=max_holes, max_height=15, max_width=15, fill_value=255,p=1),
            A.Downscale(scale_min=0.25, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=1)
        ])
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE) 
        img = self.augmentation(image=img)['image']
        img_x = self.downscale_and_noise(image=img)['image']
        
        img_x = torch.FloatTensor(img_x).unsqueeze(0) / 255
        img_y = torch.FloatTensor(img).unsqueeze(0) / 255
        return img_x, img_y