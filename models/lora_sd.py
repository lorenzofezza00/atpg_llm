import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

# === FUNZIONE PER APPLICARE LA LORA ===
def apply_lora(pipe, lora_path, alpha=0.8):
    print(f"🔗 Caricamento LoRA da {lora_path} ...")
    lora_state_dict = load_file(lora_path)
    own_state_dict = pipe.unet.state_dict()

    for key, value in lora_state_dict.items():
        if key in own_state_dict:
            own_state_dict[key] += alpha * value
    pipe.unet.load_state_dict(own_state_dict)
    print("✅ LoRA applicata con successo!")

def get_lora_sd(model_id = "stabilityai/stable-diffusion-xl-base-1.0", lora_path = "lora/PixelScenery.safetensors"):
    # MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # modello base
    # LORA_PATH = Path("lora/PixelV3-sd15-v6a-0_p1x3l.safetensors")  # percorso LoRA
    # LORA_PATH = Path("lora/PixelScenery.safetensors")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    
    apply_lora(pipe, lora_path, alpha=0.8)
    
    return pipe, make_lora_ds_inference

def make_lora_ds_inference(pipe, prompt=""):
    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]
    
    return image
