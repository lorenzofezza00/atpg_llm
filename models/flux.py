
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from PIL import Image
import torch
from diffusers import FluxPipeline


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


# ----------------- FLUX T2I BASE -----------------

class FluxT2I:
    """
    Generatore base Text-to-Image con FluxPipeline.
    Salva automaticamente ogni step e mantiene un log JSON.
    """
    def __init__(
        self,
        system_prompt: str,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        dtype: torch.dtype = torch.bfloat16,
        offload_to_cpu: bool = True,
        run_base_dir: str = "runs",
    ):
        if not system_prompt.strip():
            raise ValueError("system_prompt non può essere vuoto.")
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.run_folder = _make_run_folder(run_base_dir)

        # Caricamento modello Flux
        print(f"Loading Flux model: {model_id}")
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if offload_to_cpu:
            print("Offload to cpu")
            self.pipe.enable_model_cpu_offload()

        # Log di sessione
        self.log = {
            "model_id": model_id,
            "system_prompt": system_prompt,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "steps": []
        }

    def generate(
        self,
        user_prompt: str,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        max_sequence_length: int = 256,
        seed: Optional[int] = None,
        step_name: Optional[str] = None
    ) -> Path:
        """
        Genera immagine singola da prompt.
        """
        full_prompt = f"{self.system_prompt}\n{user_prompt}"
        generator = torch.Generator("cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)

        print(f"Generating image for: {user_prompt[:60]}...")

        result = self.pipe(
            full_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )

        image = result.images[0]
        step_idx = len(self.log["steps"]) + 1
        step_label = step_name or f"step_{step_idx:02d}"
        fname = f"{self.run_folder.name}_{step_label}.png"
        out_path = self.run_folder / fname
        _save_image(image, out_path)

        # aggiorna log
        entry = {
            "step": step_idx,
            "step_label": step_label,
            "prompt": user_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
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
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        max_sequence_length: int = 256,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """
        Esegue più prompt in sequenza e salva ogni immagine.
        """
        paths = []
        for i, p in enumerate(prompts, start=1):
            out = self.generate(
                user_prompt=p,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                seed=seed,
                step_name=f"step_{i:02d}",
            )
            paths.append(out)
        return paths

    def _write_log(self):
        log_path = self.run_folder / "log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)


# ----------------- FLUX CHAT -----------------

class FluxT2IChat(FluxT2I):
    """
    Versione chat con contesto e generazione di immagini
    coerente con la storia della conversazione.
    """
    def __init__(
        self,
        system_prompt: str,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        max_context_messages: int = 10,
        dtype: torch.dtype = torch.bfloat16,
        offload_to_cpu: bool = True,
        run_base_dir: str = "runs",
    ):
        super().__init__(
            system_prompt=system_prompt,
            model_id=model_id,
            dtype=dtype,
            offload_to_cpu=offload_to_cpu,
            run_base_dir=run_base_dir,
        )
        self.max_context_messages = max_context_messages
        self.chat_history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def chat(
        self,
        user_prompt: str,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        max_sequence_length: int = 256,
        seed: Optional[int] = None,
    ) -> Path:
        """
        Aggiunge messaggio utente al contesto, genera immagine e salva.
        """
        self.chat_history.append({"role": "user", "content": user_prompt})
        if len(self.chat_history) > self.max_context_messages:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-self.max_context_messages + 1:]

        # costruisci prompt con contesto
        ctx_prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in self.chat_history)

        out_path = self.generate(
            user_prompt=ctx_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            seed=seed,
            step_name=f"chat_turn_{len(self.log['steps']) + 1:02d}"
        )

        # aggiorna storia
        self.chat_history.append({"role": "assistant", "content": f"[image_saved:{out_path.name}]"})
        return out_path