import requests
import time
import base64
import torch
import numpy as np
import io
from PIL import Image



class RunPodClient:
    @staticmethod
    def send_and_poll(api_token, endpoint_id, payload, timeout=300):
        url_run = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        
        init_response = requests.post(url_run, headers=headers, json={"input": payload}, timeout=15)
        init_response.raise_for_status()
        job_id = init_response.json().get("id")
        
        url_status = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("RunPod job timed out.")

            status_response = requests.get(url_status, headers=headers, timeout=10)
            status_data = status_response.json()
            
            if status_data.get("status") == "COMPLETED":
                return status_data
            if status_data.get("status") in ["FAILED", "CANCELLED"]:
                raise Exception(f"Job {status_data.get('status')}")
                
            time.sleep(2)

    @staticmethod
    def process_output_image(base64_image):
        """Converts base64 string to a ComfyUI-compatible image tensor."""
        if not base64_image:
            return None
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    @staticmethod
    def process_output_latent(base64_latent):
        """Converts base64 string to a ComfyUI latent dictionary."""
        if not base64_latent:
            return None
        latent_data = base64.b64decode(base64_latent)
        with io.BytesIO(latent_data) as f:
            # map_location='cpu' ensures we don't crash if the user doesn't have a high-end GPU
            raw_latent = torch.load(f, map_location='cpu', weights_only=False)
        return {"samples": raw_latent}

class Payload_ZIT:
    @staticmethod
    def _apply_extras(args, cfg_norm, sigmas=None, lora_path=None, lora_strength=None):
        """Helper to inject optional and new required parameters."""
        args["cfg_normalization"] = cfg_norm
        # Only include lora_path if it's a non-empty string
        if lora_path and isinstance(lora_path, str) and lora_path.strip():
            args["lora_path"] = lora_path
            args["lora_strength"] = lora_strength # Pass the strength to the API
        if sigmas is not None:
            args["sigmas"] = sigmas      
        return args

    @staticmethod
    def txt2img(prompt, width, height, steps, seed, guidance_scale, cfg_norm, sigmas=None, lora_path=None, lora_strength=1.0):
        args = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale
        }
        return {"pipe_type": "base", "pipeline_args": Payload_ZIT._apply_extras(args, cfg_norm, sigmas, lora_path, lora_strength)}

    @staticmethod
    def img2img(prompt, strength, steps, seed, guidance_scale, image_b64, cfg_norm, sigmas=None, lora_path=None, lora_strength=1.0):
        args = {
            "prompt": prompt,
            "image": image_b64,
            "strength": strength,
            "num_inference_steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale            
        }
        return {"pipe_type": "i2i", "pipeline_args": Payload_ZIT._apply_extras(args, cfg_norm, sigmas, lora_path,lora_strength)}

    @staticmethod
    def inpaint(prompt, image_b64, mask_b64, strength, steps, seed, guidance_scale, cfg_norm, sigmas=None, lora_path=None, lora_strength=1.0):
        args = {
            "prompt": prompt,
            "image": image_b64,
            "mask_image": mask_b64,
            "strength": strength,
            "num_inference_steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale
        }
        return {"pipe_type": "inpaint", "pipeline_args": Payload_ZIT._apply_extras(args, cfg_norm, sigmas, lora_path, lora_strength)}


class Payload_Wan:
    @staticmethod
    def _apply_common_args(args, steps, seed, guidance_scale, num_frames, fps):
        """Standardizes video generation parameters."""
        args.update({
            "num_inference_steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "fps": fps
        })
        return args

    @staticmethod
    def txt2vid(prompt, width, height, steps, seed, guidance_scale, num_frames=81, fps=16):
        args = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        payload = Payload_Wan._apply_common_args(args, steps, seed, guidance_scale, num_frames, fps)
        return {"pipe_type": "txt2vid", "pipeline_args": payload}

    @staticmethod
    def img2vid(prompt, image_b64, steps, seed, guidance_scale, num_frames=81, fps=16):
        args = {
            "prompt": prompt,
            "image": image_b64,
        }
        payload = Payload_Wan._apply_common_args(args, steps, seed, guidance_scale, num_frames, fps)
        return {"pipe_type": "img2vid", "pipeline_args": payload}

    @staticmethod
    def vid2vid(prompt, video_b64, strength, steps, seed, guidance_scale, num_frames=81, fps=16):
        args = {
            "prompt": prompt,
            "video": video_b64,
            "strength": strength,
        }
        payload = Payload_Wan._apply_common_args(args, steps, seed, guidance_scale, num_frames, fps)
        return {"pipe_type": "vid2vid", "pipeline_args": payload}

    @staticmethod
    def animate(prompt, image_b64, steps, seed, guidance_scale, num_frames=81, fps=16):
        args = {
            "prompt": prompt,
            "image": image_b64,
        }
        payload = Payload_Wan._apply_common_args(args, steps, seed, guidance_scale, num_frames, fps)
        return {"pipe_type": "animate", "pipeline_args": payload}

    @staticmethod
    def vace(video_b64, steps, seed):
        """Video Autoencoder (VACE) for reconstruction/compression tasks."""
        args = {
            "video": video_b64,
            "num_inference_steps": steps,
            "seed": seed
        }
        return {"pipe_type": "vace", "pipeline_args": args}