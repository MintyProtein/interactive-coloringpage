import argparse
import sys
import cv2
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
import gradio as gr
from omegaconf import OmegaConf
from src.colorizer import InteractiveColoringPipeLine
from src.canvas_generator import CanvasGenerator


CANVAS_PROMPT_PREFIX = "A simple coloring page of "
COLORIZER_PROMPT_POSTFIX = "aesthetic, masterpiece, simple"
COLORIZER_ITERATION = 5

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_config', default='configs/generator_config.yaml')
    parser.add_argument('--postprocessor_config', default='configs/postprocessor_config.yaml')
    parser.add_argument('--colorizer_config', default='configs/colorizer_config.yaml')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    
    # Load Canvas Generator
    canvas_generator = CanvasGenerator.load_from_config(args.generator_config)
    
    # Load colorization model
    color_config = OmegaConf.load(args.colorizer_config)
    color_pipe = InteractiveColoringPipeLine.from_config(color_config)
    color_pipe.colorizer.load_checkpoint(ldm_path="checkpoints/colorizer/anything-v3-full.safetensors",
                                         cldm_path="checkpoints/colorizer/control_v11p_sd15s2_lineart_anime.pth")
    
    with gr.Blocks() as demo:
        with torch.no_grad():            
            with gr.Row():
                # Col A: Canvas Generation
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Canvas Generation</center></h3>")
                    canvas_image = gr.Image(label='Canvas', interactive=False)
                    text_prompt = gr.Textbox(label='Text prompt')
                    canvas_btn = gr.Button(value='Generate')
                            
                # Col B: Segmentation
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Segmentation</center></h3>")
                    seg_img = gr.Image(label='Segmentation map', interactive=False)
                    seg_btn = gr.Button(value='Check Segmentation map')
                    
                #Col C: Inpainted image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Output</center></h3>")
                    colored_img = gr.Image(label='Inpainted image', interactive=False) 
                    color_btn = gr.Button(value='Inpaint')


            def generate_canvas(text_prompt,
                                negative_prompt="nsfw, watermark, text, bad quality, disfigured, color", 
                                steps=30, 
                                guidance_scale=8):
                prompt = CANVAS_PROMPT_PREFIX + text_prompt
                result = canvas_generator.generate(prompt = prompt, 
                                                  negative_prompt = negative_prompt, 
                                                  inference_steps = steps, 
                                                  guidance_scale = guidance_scale,
                                                  n_images=1)[0]
            
                return result
            canvas_btn.click(generate_canvas, inputs=text_prompt, outputs=canvas_image)
        
            def generate_segmap(input_img):
                input_img[input_img>127] = 255
                input_img[input_img<=127] = 0
                color_pipe.set_lineart(input_img)
           
                segmentation_map = color_pipe._segmentation_map
                n_labels = len(np.unique(segmentation_map))
                map_color = np.zeros_like(segmentation_map)[:, :, None]
                map_color = np.concatenate((map_color, map_color, map_color), axis=2)
                for i in range(1, n_labels):
                    map_color[segmentation_map==i] = [int(j) for j in np.random.randint(0,255,3)]
                return map_color
  
            seg_btn.click(generate_segmap, inputs=canvas_image, outputs=seg_img)
            
                                                            
            def colorize_canvas(input_img, text_prompt):
                input_img[input_img>127] = 255
                input_img[input_img<=127] = 0
                
                prompt = text_prompt + COLORIZER_PROMPT_POSTFIX
                color_pipe.set_lineart(input_img, prompt)
                output_img = input_img
                for i in range(COLORIZER_ITERATION+1):
                    color_pipe.update_color(output_img)
                    output_img = color_pipe.robot_turn_coloring(n_target = color_pipe.num_regions // COLORIZER_ITERATION)
                    yield output_img
            
            color_btn.click(colorize_canvas,
                            inputs=[canvas_image, text_prompt],
                            outputs=colored_img)
        demo.launch(share=True)