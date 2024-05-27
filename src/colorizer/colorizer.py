import cv2
import numpy as np
import torch
import einops
from src.colorizer.ldm.util import instantiate_from_config
from src.colorizer.utils import *
from src.colorizer.cldm.ddim_hacked import DDIMSampler
from src.colorizer.cldm.model import load_state_dict
class BaseColorizer:
    def colorize_target(self, color_current, target_mask):
        raise NotImplementedError
        return 


class ControlNetColorizer(BaseColorizer):  
    def __init__(self, model_config, prompt, n_prompt, *args, **kwargs):
        self.model = instantiate_from_config(model_config)
        self.model.cuda()
        self.sampler = DDIMSampler(self.model)
        
        self.prompt = prompt
        self.n_prompt = n_prompt
        return
    
    def load_checkpoint(self, ldm_path="./models/anything-v3-full.safetensors",
                        cldm_path="./models/control_v11p_sd15s2_lineart_anime.pth"):
        self.model.load_state_dict(load_state_dict(ldm_path, location='cuda'), strict=False)
        self.model.load_state_dict(load_state_dict(cldm_path, location='cuda'), strict=False)

    def colorize_target(self, input_img,
                        target_mask, prompt="", ddim_steps=20,
                        color_mask=None, num_samples=1, 
                        eta=1.0, scale=6.0, strength=1.3, guess_mode=False):
        
        with torch.no_grad():
            H, W, C = input_img.shape
            assert (H,W) == target_mask.shape
            img_pixel_batched = input_img.copy()[None]
            target_mask_batched = target_mask.copy()[None, :, :, None]
            
            if color_mask is not None:    
                assert (H,W) == color_mask.shape
                mask_latent = cv2.resize(color_mask, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
                
                mask_latent = 1.0 - torch.from_numpy(mask_latent.copy()).float().cuda()
                mask_latent = torch.stack([mask_latent for _ in range(num_samples)], dim=0)
                mask_latent = einops.rearrange(mask_latent, 'b h w -> b 1 h w').clone()
                
                x0 = torch.from_numpy(input_img.copy()).float().cuda() / 127.0 - 1.0
                x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
                x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()
                x0 = self.model.get_first_stage_encoding(self.model.encode_first_stage(x0))
                
                self.sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)
            else:
                mask_latent = None
                x0 = None
                
            control = 1.0 - torch.from_numpy(input_img.copy()).float().cuda() / 255.0
            control = torch.unsqueeze(control, 0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ","+ self.prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)
            
            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else [strength] * 13 
            
            samples, intermediates = self.sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, mask=mask_latent, x0=x0,
                                                        verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
            
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            x_samples = x_samples * target_mask_batched + img_pixel_batched * (1.0 - target_mask_batched)
            results = [x_samples[i] for i in range(num_samples)]
            
            return results