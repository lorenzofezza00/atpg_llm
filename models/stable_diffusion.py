import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import transformers
import torch
import re
from transformers import pipeline, CLIPTokenizer, AutoTokenizer

def _make_run_folder(base_dir: str = "runs") -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(base_dir) / f"run_{ts}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def _save_image(img, path: Path):
    """Salva PIL image su disco."""
    if isinstance(img, Image.Image):
        img.save(path)
    else:
        raise RuntimeError("Formato immagine non riconosciuto per il salvataggio.")


class SDT2I:
    def __init__(
        self,
        system_prompt: str,
        model_id: str = "sd-legacy/stable-diffusion-v1-5",
        dtype: torch.dtype = torch.float16,
        run_base_dir: str = "runs",
        device: str = "cuda"
    ):
        if not system_prompt.strip():
            raise ValueError("system_prompt non può essere vuoto.")
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.run_folder = _make_run_folder(run_base_dir)

        print(f"Loading Stable Diffusion model: {model_id}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16"
        ).to(device)

        self.log = {
            "model_id": model_id,
            "system_prompt": system_prompt,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "steps": []
        }

    def generate(
        self,
        user_prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 15,
        seed: Optional[int] = None,
        shape: tuple = (512, 512),
        step_name: Optional[str] = None
    ) -> Path:
        
        full_prompt = f"{self.system_prompt}\n{user_prompt}"
        generator = torch.Generator(device="cuda")
        if seed is not None:
            generator = generator.manual_seed(seed)

        print(f"Generating image for: {user_prompt[:60]}...")

        result = self.pipe(
            prompt=full_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=shape[0],
            width=shape[1],
            generator=generator
        )

        image = result.images[0]
        step_idx = len(self.log["steps"]) + 1
        step_label = step_name or f"step_{step_idx:02d}"
        fname = f"{self.run_folder.name}_{step_label}.png"
        out_path = self.run_folder / fname
        _save_image(image, out_path)

        entry = {
            "step": step_idx,
            "step_label": step_label,
            "prompt": user_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "shape": shape,
            "seed": seed,
            "file": str(out_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        self.log["steps"].append(entry)
        self._write_log()

        return out_path

    def generate_stepwise(
        self,
        prompts: List[str],
        guidance_scale: float = 7.5,
        num_inference_steps: int = 15,
        seed: Optional[int] = None,
        shape: tuple = (512, 512),
    ) -> List[Path]:
        
        paths = []
        for i, p in enumerate(prompts, start=1):
            out = self.generate(
                user_prompt=p,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                shape=shape,
                step_name=f"step_{i:02d}",
            )
            paths.append(out)
        return paths

    def _write_log(self):
        log_path = self.run_folder / "log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)

class SDT2IChat(SDT2I):
    def __init__(
        self,
        system_prompt: str,
        model_id: str = "sd-legacy/stable-diffusion-v1-5",
        max_context_messages: int = 10,
        dtype: torch.dtype = torch.float16,
        run_base_dir: str = "runs",
        device: str = "cuda"
    ):
        super().__init__(
            system_prompt=system_prompt,
            model_id=model_id,
            dtype=dtype,
            run_base_dir=run_base_dir,
            device=device
        )
        self.max_context_messages = max_context_messages
        self.chat_history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def chat(
        self,
        user_prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 15,
        seed: Optional[int] = None,
        shape: tuple = (512, 512),
    ) -> Path:
        
        self.chat_history.append({"role": "user", "content": user_prompt})
        if len(self.chat_history) > self.max_context_messages:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-self.max_context_messages + 1:]

        ctx_prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in self.chat_history)

        out_path = self.generate(
            user_prompt=ctx_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            shape=shape,
            step_name=f"chat_turn_{len(self.log['steps']) + 1:02d}"
        )

        self.chat_history.append({"role": "assistant", "content": f"[image_saved:{out_path.name}]"})
        return out_path

class SDT2IChatWithLlama(SDT2I):
    def __init__(
        self,
        system_prompt: str,
        max_context_messages: int = 10,
        llama_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        sd_model_id: str = "sd-legacy/stable-diffusion-v1-5",
        dtype: torch.dtype = torch.float16,
        run_base_dir: str = "runs",
        device: str = "cuda"
    ):
        super().__init__(
            system_prompt=system_prompt,
            model_id=sd_model_id,
            dtype=dtype,
            run_base_dir=run_base_dir,
            device=device
        )
        self.max_context_messages = max_context_messages
        self.chat_history = [{"role": "system", "content": system_prompt}]

        self.llama_pipeline = pipeline(
            "text-generation",
            model=llama_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
    def summarize(self):
        # Convert chat history (list of dicts) into readable conversation text
        chat_text = ""
        for msg in self.chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            chat_text += f"{role}: {content}\n"

        # Clear, strict instruction for LLaMA
        instruction = (
            "You are an expert prompt engineer for image generation models such as Stable Diffusion.\n"
            "Analyze the entire conversation below and produce a single final prompt that reflects "
            "all user instructions, corrections, and details mentioned so far.\n"
            "Do NOT include explanations, system messages, or meta text. "
            "Only output the final image prompt itself — nothing else.\n"
            "The prompt must be at most 77 tokens long.\n\n"
            "Conversation:\n"
        )

        # Combine everything
        inputs = instruction + chat_text + "\nFinal image prompt:"

        # Generate summary
        outputs = self.llama_pipeline(
            inputs,
            max_new_tokens=77,
            do_sample=False,
            temperature=0.3,
        )

        # Extract only the generated part (after the colon)
        result = outputs[0]["generated_text"]
        result = result.split("Final image prompt:")[-1].strip()

        return result


    def chat(
        self,
        user_prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 15,
        seed: int = None,
        shape: tuple = (512, 512),
    ):
        self.chat_history.append({"role": "user", "content": user_prompt})

        if len(self.chat_history) > self.max_context_messages:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-self.max_context_messages + 1:]

        sd_prompt = self.summarize()
        print(f"Summarized prompt (<=77 tokens): {sd_prompt}")

        out_path = self.generate(
            user_prompt=sd_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            shape=shape,
            step_name=f"chat_turn_{len(self.log['steps']) + 1:02d}"
        )

        self.chat_history.append({"role": "assistant", "content": f"[image_saved:{out_path.name}]"})
        return out_path
