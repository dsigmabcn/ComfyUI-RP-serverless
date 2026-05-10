### Vibecoded by Koala Nation ###
import requests
import json
import torch
import numpy as np
from PIL import Image
import io
import base64
import time
from .pipelines import Payload_ZIT, Payload_Wan, RunPodClient


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
            }#,
            #"optional":{
            #    "timeout": ("INT", {"default": 120, "min": 10, "max": 600, "step": 10}),
            #}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "serverless/ZIT"

    def generate(self, api_token, endpoint_id, prompt, width, height, steps, seed):
       
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
                #"lora_path": ("STRING",{"forceInput": True}),
                "zit_lora": ("serverless_LORA",), # New input type
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "serverless/ZIT"

    def generate(self, width, height, api_token, endpoint_id, seed, steps, guidance, denoise, cfg_normalization, text="", image=None, mask=None, sigmas=None, zit_lora=None):
        
        prompt = text        
        
        # 2. Resolve Input Image and Dimensions (Exact same logic as before)
        input_image_b64 = None
        mask_image_b64 = None
        #active_image = None
        sigma_list = None
        lora_path = ""
        lora_strength = 1.0
                           
        if image is not None:     
            height = image.shape[1] #overrides value of height]
            width = image.shape[2] #overrides value of withd     
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
        
        if sigmas is not None:
            sigma_list = sigmas.cpu().numpy().tolist()

        if zit_lora is not None:
            lora_path = zit_lora.get("path", "")
            lora_strength = zit_lora.get("strength", 1.0)      

        # Payloads updated with cfg_normalization and sigmas
        if input_image_b64 and mask_image_b64: # inpaint pipeline
            payload = Payload_ZIT.inpaint(prompt, input_image_b64, mask_image_b64, denoise, steps, seed, guidance, cfg_normalization, sigma_list, lora_path, lora_strength)        
        elif input_image_b64: # img2img pipeline
            payload = Payload_ZIT.img2img(prompt, denoise, steps, seed, guidance, input_image_b64, cfg_normalization, sigma_list, lora_path,lora_strength)        
        else: # txt2img pipeline            
            payload = Payload_ZIT.txt2img(prompt, width, height, steps, seed, guidance, cfg_normalization, sigma_list, lora_path,lora_strength)   
        
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


class Wan22_Base:
    """Helper to keep node code DRY"""
    def _img_to_b64(self, image):
        if image is None: return None
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.uint8(i))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

class Wan22_Simple(Wan22_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_token": ("STRING", {"default": "Enter RunPod API Key"}),
                "endpoint_id": ("STRING", {"default": "Enter Endpoint ID"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic dragon"}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 1280, "step": 64}),
                "height": ("INT", {"default": 720, "min": 256, "max": 720, "step": 64}),
                "num_frames": ("INT", {"default": 81}),
            },
            "optional": {"image": ("IMAGE",)}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images")
    FUNCTION = "generate"
    CATEGORY = "serverless/Wan22"

    def generate(self, api_token, endpoint_id, prompt, width, height, num_frames, image=None):
        img_b64 = self._img_to_b64(image)
        
        # Cleaner logic: Use the pipeline directly
        if img_b64:
            payload = Payload_Wan.img2vid(prompt, img_b64, 30, 0, 6.0, num_frames)
        else:
            payload = Payload_Wan.txt2vid(prompt, width, height, 30, 0, 6.0, num_frames)

        try:
            result = RunPodClient.send_and_poll(api_token, endpoint_id, payload, timeout=600)
            output_data = result.get("output", {})
            frames_list = output_data.get("frames", [])

            if not frames_list:
                raise ValueError("no frames received")

            image_tensor = RunPodClient.process_output_frames(frames_list)

            return (image_tensor,)

        except Exception as e:
            print(f"❌ Wan Simple Error: {e}")
            return (torch.zeros((1, height, width, 3)), str(e))

class Wan22_Advanced(Wan22_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_token": ("STRING", {"default": "Enter RunPod API Key"}),
                "endpoint_id": ("STRING", {"default": "Enter Endpoint ID"}),
                "positive_prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1280}),
                "height": ("INT", {"default": 720}),
                "num_frames": ("INT", {"default": 81}),
                #"fps": ("INT", {"default": 16}),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 40}),
                "guidance_scale": ("FLOAT", {"default": 6.0}),
                "denoise": ("FLOAT", {"default": 0.7}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "video_input": ("STRING", {"default": ""}),
                "lora": ("serverless_LORA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "serverless/Wan22"

    def generate(self, **kwargs):
        # 1. Process Assets
        img_b64 = self._img_to_b64(kwargs.get("image"))
        vid_b64 = kwargs.get("video_input")
        
        # 2. Select Payload via Pipeline Class
        if vid_b64:
            payload = Payload_Wan.vid2vid(kwargs["positive_prompt"], vid_b64, kwargs["denoise"], kwargs["steps"], kwargs["seed"], kwargs["guidance_scale"], kwargs["num_frames"])
        elif img_b64:
            payload = Payload_Wan.img2vid(kwargs["positive_prompt"], img_b64, kwargs["steps"], kwargs["seed"], kwargs["guidance_scale"], kwargs["num_frames"])
        else:
            payload = Payload_Wan.txt2vid(kwargs["positive_prompt"], kwargs["width"], kwargs["height"], kwargs["steps"], kwargs["seed"], kwargs["guidance_scale"], kwargs["num_frames"])

        # 3. Handle LoRA / Negative Prompt (if your API supports them in pipeline_args)
        if kwargs.get("lora"):
            payload["pipeline_args"]["lora_path"] = kwargs["lora"]["path"]
            payload["pipeline_args"]["lora_strength"] = kwargs["lora"]["strength"]
        
        if kwargs.get("negative_prompt"):
            payload["pipeline_args"]["negative_prompt"] = kwargs["negative_prompt"]

        # 4. Execute
        try:
            res = RunPodClient.send_and_poll(kwargs["api_token"], kwargs["endpoint_id"], payload, timeout=600)
            output_data = res.get("output", {})
            frames_list = output_data.get("frames", [])

            if not frames_list:
                raise ValueError("No frames received from RunPod")

            image_tensor = RunPodClient.process_output_frames(frames_list)
            return (image_tensor,)
        
        except Exception as e:
            print(f"❌ Wan Advanced Error: {e}")
            return (torch.zeros((1, kwargs["height"], kwargs["width"], 3)), str(e))


class serverless_Lora_Loader:
    """
    Combines LoRA path and strength into a single 'serverless_LORA' object.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_path": ("STRING", {"default": "Enter LoRA path or ID"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("serverless_LORA",)
    RETURN_NAMES = ("serverless_LORA",)
    FUNCTION = "get_lora"
    CATEGORY = "serverless/utils"

    def get_lora(self, lora_path, strength):
        # We return a dictionary containing both values
        return ({"path": lora_path, "strength": strength},)

class API_endpoint_tokens:
    """
    A node to plug in the Runpod API tokens and worker id to pass to the nodes
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_token": ("STRING", {"default": "Enter RunPod API Key", "password": True}),
                "endpoint_id": ("STRING", {"default": "Enter Endpoint ID"}),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("api_token_out","endpoint_id_out")
    FUNCTION = "server_info"
    CATEGORY = "serverless/utils"

    def server_info (self, api_token, endpoint_id):
        # We return a dictionary containing both values
        return (api_token, endpoint_id)


NODE_CLASS_MAPPINGS = {
    "ZIT_Simple": ZIT_Simple,    
    "ZIT_Advanced": ZIT_Advanced,
    "Wan22_Simple": Wan22_Simple,
    "Wan22_Advanced": Wan22_Advanced,
    "serverless_Lora_Loader": serverless_Lora_Loader,
    "API_endpoint_tokens": API_endpoint_tokens
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZIT_Simple": "[Simple] Z-Image Turbo API",
    "ZIT_Advanced": "[Advanced] Z-Image Turbo API",    
    "Wan22_Simple": "Wan 2.1 Video [Simple]",
    "Wan22_Advanced": "Wan 2.1 Video [Advanced]",
    "serverless_Lora_Loader": "Lora settings [Advanced nodes] API",
    "API_endpoint_tokens": "API and worker information"
}