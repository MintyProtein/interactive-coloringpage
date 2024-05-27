import cv2
import numpy as np


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    
    
def resize_longest_side(input_img, target_size):
    if input_img.ndim == 3:
        H, W, C = input_img.shape
    elif input_img.ndim == 2:
        H, W  = input_img.shape

    longest_length = max(H, W)
    scale = target_size / longest_length
    new_H = int(H * scale)
    new_W = int(W * scale)
    
    return cv2.resize(input_img, (new_W, new_H), cv2.INTER_AREA)
    
def pad_to_square(input_img, pad_value=0):
    if input_img.ndim == 3:
        H, W, C = input_img.shape
    elif input_img.ndim == 2:
        H, W  = input_img.shape
        C = None
    
    # Resize the image. 
    longest_length = max(H, W)
    
    # Pad the image
    pad_height = longest_length - H
    pad_width = longest_length - W
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    
    if C is not None:
        image_padded = np.pad(input_img, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), mode='constant', constant_values=pad_value)
    else:
        image_padded = np.pad(input_img, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=pad_value)

    return image_padded, (top_pad, bottom_pad, left_pad, right_pad)


def preprocess_lineart(lineart: np.ndarray, resolution, pad_value=0):
    if lineart.ndim == 3:
        gray = cv2.cvtColor(lineart, cv2.COLOR_BGR2GRAY)
    else:
        gray = lineart.copy()
    gray = resize_longest_side(gray, target_size=resolution)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5).astype(np.uint8)
    gray, pad_info = pad_to_square(gray, pad_value=pad_value)
    return gray, pad_info

def preprocess_color(img, resolution, pad_value=0):
    color = resize_longest_side(img, target_size=resolution)
    color, pad_info = pad_to_square(color, pad_value=pad_value)
    return color

def postprocess():
    return NotImplementedError()