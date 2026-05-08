### Vibecoded by Koala Nation ###
import requests
import json
import torch
import numpy as np
from PIL import Image
import io
import base64
import time
from .pipelines import Payload_ZIT, RunPodClient

class ZIT_Simple:
    """
    Simple node using Payload_ZIT. Provided a text prompt, 
    generates an image with Z-Image Turbo using a worker from Runpod.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_token": ("STRING", {"default": "Enter RunPod API Key"}),
                "endpoint_id": ("STRING", {"default": "Enter Endpoint ID"}),
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic cyborg cat, neon lights, high resolution"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional":{
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600, "step": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "serverless/ZIT"

    def generate(self, api_token, endpoint_id, prompt, width, height, steps, seed, timeout=300):
       
        payload = Payload_ZIT.txt2img(prompt, width, height, steps, seed, guidance_scale=0, cfg_norm=True, sigmas=None)         

        try:
            result = RunPodClient.send_and_poll(api_token, endpoint_id, payload, timeout=300)            
            image_tensor = RunPodClient.process_output_image(result["output"].get("image"))            
            if image_tensor is None:
                return (self.empty_image(width, height),)

            return (image_tensor,)

        except Exception as e:
            print(f"❌ ZIT Simple 2 Error: {e}")
            return (self.empty_image(width, height),)

    def empty_image(self, width=1024, height=1024):
        return torch.zeros((1, height, width, 3))


class ZIT_Advanced:
    """
    Advanced node, including image to image and inpainting
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {                
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "text": ("STRING", {"multiline": True, "default": "a cat"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "cfg_normalization": ("BOOLEAN", {"default": True}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "api_token": ("STRING", {"default": "Enter RunPod API Key"}),
                "endpoint_id": ("STRING", {"default": "Enter Endpoint ID"}),
            },
            "optional": {                                
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "sigmas": ("SIGMAS",),
                "lora_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "serverless/ZIT"

    def generate(self, width, height, api_token, endpoint_id, seed, steps, guidance, denoise, 
                 cfg_normalization,text="", image=None, mask=None, sigmas=None, lora_path=""):
        
        '''# 1. Resolve Prompt (Exact same logic as before)
        prompt = text
        if conditioning is not None:
            try:
                if "n_pixels" in conditioning[0][1]:
                     prompt = conditioning[0][1].get("text", text)
                else:
                     prompt = text
            except:
                prompt = text'''
        prompt = text        
        
        # 2. Resolve Input Image and Dimensions (Exact same logic as before)
        input_image_b64 = None
        mask_image_b64 = None
        active_image = None
                           
        if image is not None:     
            height = image.shape[1] #overrides value of height]
            width = image.shape[2] #overrides value of widht     
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.uint8(i))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            input_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8") #image as data
        
        if mask is not None:
            m = 255. * mask[0].cpu().numpy()
            m_img = Image.fromarray(np.uint8(m)).convert("L")
            m_buf = io.BytesIO()
            m_img.save(m_buf, format="PNG")
            mask_image_b64 = base64.b64encode(m_buf.getvalue()).decode("utf-8") #mask as data

        sigma_list = None
        if sigmas is not None:
            sigma_list = sigmas.cpu().numpy().tolist()
        

        #Payloads, we create them from 'pipelines'
        '''
        if input_image_b64 and mask_image_b64: #inpaint pipeline
            payload = Payload_ZIT.inpaint(prompt, input_image_b64, mask_image_b64, steps, seed, guidance)        
        elif input_image_b64: #img2img pipeline
            payload = Payload_ZIT.img2img(prompt, denoise, steps, seed, guidance, input_image_b64)        
        else: #txt2img pipeline            
            payload = Payload_ZIT.txt2img(prompt, width, height, steps, seed, guidance) #no denoise is used
        '''
        # Payloads updated with cfg_normalization and sigmas
        if input_image_b64 and mask_image_b64: # inpaint pipeline
            payload = Payload_ZIT.inpaint(prompt, input_image_b64, mask_image_b64, steps, seed, guidance, cfg_normalization, sigma_list, lora_path)        
        elif input_image_b64: # img2img pipeline
            payload = Payload_ZIT.img2img(prompt, denoise, steps, seed, guidance, input_image_b64, cfg_normalization, sigma_list, lora_path)        
        else: # txt2img pipeline            
            payload = Payload_ZIT.txt2img(prompt, width, height, steps, seed, guidance, cfg_normalization, sigma_list, lora_path)   
        
        try:
            result = RunPodClient.send_and_poll(api_token, endpoint_id, payload, timeout=300) #sends request to endpoing runpod        
            image_tensor = RunPodClient.process_output_image(result["output"].get("image")) #convert image from request to image
            latent_out = RunPodClient.process_output_latent(result["output"].get("latent")) #convert latent from request to latent
            if image_tensor is None:
                image_tensor = self.empty_image(width, height)
            if latent_out is None:
                latent_out = self.empty_latent(width, height)                          
            return (image_tensor, latent_out)

        except Exception as e:
            print(f"❌ ZIT Advanced 2 Error: {e}")
            return (self.empty_image(width, height), self.empty_latent(width, height))
    
    def empty_image(self, width=1024, height=1024):
        return torch.zeros((1, height, width, 3))

    def empty_latent(self, width=1024, height=1024):
        return {"samples": torch.zeros([1, 16, height // 8, width // 8])}

NODE_CLASS_MAPPINGS = {
    "ZIT_Simple": ZIT_Simple,    
    "ZIT_Advanced": ZIT_Advanced   
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZIT_Simple": "[Simple] Z-Image Turbo API",
    "ZIT_Advanced": "[Advanced] Z-Image Turbo API"
}