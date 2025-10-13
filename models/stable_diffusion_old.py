from diffusers import StableDiffusionPipeline
import torch

def get_stable_diffusion(model_id = "stabilityai/sd-turbo"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    return pipe, make_stable_diffusion_inference

def make_stable_diffusion_inference(pipe, prompt="Hello", shape=(512, 512)):
    image = pipe(
        prompt,
        num_inference_steps=15,
        guidance_scale=7.5,
        height=shape[0],
        width=shape[1]
    ).images[0]
    return image
