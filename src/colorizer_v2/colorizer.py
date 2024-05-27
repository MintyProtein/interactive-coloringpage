from abc import abstractmethod
import cv2
import numpy as np
import torch
import einops
from diffusers import (AutoencoderKL, StableDiffusionXLControlNetPipeline,  StableDiffusionControlNetInpaintPipeline,
                       EulerAncestralDiscreteScheduler, ControlNetModel)

class BaseColorizer:
    @abstractmethod
    def colorize_target():
        raise NotImplementedError
        return 


class ControlNetColorizer(BaseColorizer):  
    def __init__(self, pipe: StableDiffusionXLControlNetPipeline, device=torch.device('cuda')):
        self.pipe = pipe
        self.device = device
        self.pipe.to(self.device)
        return
    
    def load_from_config(config_path, device=torch.device('cuda')):
        config = OmegaConf.load(config_path)
        
        contolnet = ControlNetModel.from_pretrained(config.controlnet_checkpoint)
        vae = AutoencoderKL.from_pretrained(config.vae_checkpoint)
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(scheduler_checkpoint)
        
        return StableDiffusionXLControlNetPipeline.from_pretrained(
            config.sd_checkpoint,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            safety_checker=None,
        )

    def colorize_target(self, 
                        base_canvas, 
                        current_color,
                        target_mask,
                        prompt,
                        negative_prompt, 
                        ):
        return