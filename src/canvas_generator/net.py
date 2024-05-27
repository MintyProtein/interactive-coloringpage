from omegaconf import OmegaConf
import torch
from diffusers import StableDiffusionPipeline


class CanvasGenerator:
    def __init__(self, pipe: StableDiffusionPipeline, device):
        self.pipe = pipe
        self.device = device
        self.pipe.to(self.device)
    
    @staticmethod
    def load_from_config(config_path, device=torch.device('cuda')):
        config = OmegaConf.load(config_path)
        pipe = StableDiffusionPipeline.from_pretrained(config.canvas_generator_checkpoint, 
                                                       safety_checker=None)
        return CanvasGenerator(pipe, device)
    
    @torch.inference_mode()
    def generate(self, 
                 prompt: str, 
                 negative_prompt: str, 
                 inference_steps: int, 
                 guidance_scale: float, 
                 n_images: int, 
                 seed=None,
                 **kwargs
                 ):
        generator = None if seed is None else torch.manual_seed(seed)
        return self.pipe(prompt=prompt,
                         negative_prompt=negative_prompt,
                         num_inference_steps=inference_steps,
                         num_images_per_prompt=n_images,
                         guidance_scale=guidance_scale,
                         generator=generator,
                        ).images 
    