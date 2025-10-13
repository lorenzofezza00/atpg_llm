from transformers import pipeline
import transformers
import torch
import json
import re

class Llama:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        # model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
    def make_std_inference(self, messages):
        # messages = [
        #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        #     {"role": "user", "content": "Who are you?"},
        # ]
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        # print(outputs[0]["generated_text"][-1])
        return outputs[0]["generated_text"][-1]
    
class LlamaChat:
    def __init__(self, system_prompt: str, max_context_messages: int = 10, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        if system_prompt is None or system_prompt.strip() == '':
            raise ValueError("The system_prompt must not be empty or contain only whitespace.")
        # model_id = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "Assistant:"
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            dtype=torch.bfloat16,
            # model_kwargs={"dtype": torch.float32},
            # device="cuda",
            device_map="auto",
        )
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


# from transformers import AutoModelForCausalLM, AutoTokenizer
# from diffusers import StableDiffusionPipeline
# import torch

# def get_llama(model_path = "./llama-3.2-1b-instruct"):
#     # model_path = "./llama-3.2-1b-instruct"  # percorso del modello locale
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,  # usa float16 se hai GPU
#         device_map="auto"           # automaticamente su GPU
#     )
#     return (model, tokenizer), make_llama_inference

# def make_llama_inference(llama, history = None, prompt ="", max_new_tokens=256):
#     model, tokenizer = llama
#     if history is not None:
#         # Aggiungi input utente allo storico
#         history.append({"role": "user", "content": prompt})

#         # Costruisci prompt contestualizzato
#         prompt_context = ""
#         for msg in history:
#             if msg["role"] == "user":
#                 prompt_context += f"L'utente dice: {msg['content']}\n"
#             elif msg["role"] == "assistant":
#                 prompt_context += f"Assistente risponde: {msg['content']}\n"

#         # --- LLaMA rielabora il prompt ---
#         # llama_input = (
#         #     prompt_context
#         #     # "Trasforma questo prompt in una descrizione dettagliata "
#         #     # "adatta per generare un'immagine per: \n" + prompt_context
#         # )
#         llama_input = prompt_context

#         inputs = tokenizer(llama_input, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_p=0.9
#             )

#         refined_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         print(f"\n🎨 Prompt rielaborato da LLama-3.2:\n ➡️ {refined_prompt}\n")

#         # Aggiungi risposta di LLaMA allo storico
#         history.append({"role": "assistant", "content": refined_prompt})
#         return refined_prompt
#     else:
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#         # Generazione del testo
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 repetition_penalty=1.1,
#             )

#         # Decodifica e pulizia
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"\n🎨 Prompt rielaborato da LLama-3.2:\n ➡️ {result}\n")
#         return result