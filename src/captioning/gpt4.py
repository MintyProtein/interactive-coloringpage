import base64
import requests
from io import BytesIO
import json
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

def encode_image(image_path, resolution):
    img = Image.open(image_path)
    ratio = resolution / max(img.width, img.height)
    if ratio < 1:
        img = img.resize((int(img.width*ratio), (int(img.height*ratio))))
    img = img.convert("RGB")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
 
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')


class GPTBatchAPI:
    def __init__(self, model_name, api_key, resolution=256, max_tokens=300):
        self.model_name = model_name
        self.api_key = api_key
        self.resolution = resolution
        self.max_tokens = max_tokens
       
       
    def prepare_request(self, custom_id, text, image_path):
        base64_image = encode_image(image_path, self.resolution)
        req = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions", 
            "body": {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            },
                            {
                                "type": "image_url",
                                "image_url": 
                                    {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                            }
                        ]
                    }
                ]
            }
        }
        return req
    
    
    def prepare_batch(self, output_path, image_paths, prompt):
        with open(output_path, "a") as f:
            for i, path in enumerate(tqdm(image_paths)):
                keyword = path.split('/')[-2]
                custom_id = '/'.join(path.split('/')[-3:])
                text = prompt % keyword
                
                req = self.prepare_request(custom_id, text, path)
                f.write(json.dumps(req, ensure_ascii=False,) + "\n")
        return
    def upload_batch():
        return
    
    def generate(self, text, image_path, resolution=256, max_tokens=400):
        try:
            base64_image = encode_image(image_path, resolution)
            # API 호출
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(response.json())
                raise AssertionError
        except Exception as e:
            print(f"Request FAILED: {image_path}", e)
            pass