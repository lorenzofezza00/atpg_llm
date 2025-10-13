import transformers
import torch
import json
import re

class ChatAgent:
    def __init__(self, pipeline: transformers.Pipeline, system_prompt: str, max_context_messages: int = 10):
        if pipeline is None:
            raise ValueError("The pipeline cannot be None.")
        if system_prompt is None or system_prompt.strip() == '':
            raise ValueError("The system_prompt must not be empty or contain only whitespace.")
        
        self.pipeline = pipeline
        self.system_prompt = system_prompt
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        self.max_context_messages = max_context_messages

    def chat(self, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, 
             top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
        
        # Aggiunge messaggio utente
        self.chat_history.append({"role": "user", "content": user_prompt})
        
        # Mantieni solo gli ultimi N messaggi (per non far esplodere il contesto)
        if len(self.chat_history) > self.max_context_messages:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-self.max_context_messages:]
        
        # Genera la risposta
        outputs = self.pipeline(
            self.chat_history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        full_text = outputs[0]["generated_text"][-1]["content"]

        # ✅ Pulisce la risposta per rimuovere eventuali "User:" successivi
        cleaned = re.split(r"\n\s*User:|Assistant:", full_text)[0].strip()

        # Aggiunge la risposta ripulita
        self.chat_history.append({"role": "assistant", "content": cleaned})
        
        return cleaned

    def print_current_history(self):
        print(json.dumps(self.chat_history, indent=2))