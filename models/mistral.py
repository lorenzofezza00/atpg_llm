from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

class Mistral:
    def __init__(self, system_prompt: str, model_id="mistralai/Mistral-7B-v0.1"):
        if not system_prompt or system_prompt.strip() == "":
            raise ValueError("The system_prompt must not be empty or contain only whitespace.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.tokenizer.chat_template = (
            "System: {{ system_prompt }}\n"
            "User: {{ user_prompt }}\n"
            "Assistant:"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

        self.system_prompt = system_prompt

    def make_std_inference(self, user_prompt: str, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        full_text = outputs[0]["generated_text"][-1]["content"]
        cleaned = re.split(r"\n\s*User:|Assistant:", full_text)[0].strip()
        return cleaned


class MistralChat:
    def __init__(self, system_prompt: str, max_context_messages: int = 10, model_id="mistralai/Mistral-7B-v0.1"):
        if not system_prompt or system_prompt.strip() == "":
            raise ValueError("The system_prompt must not be empty or contain only whitespace.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "Assistant:"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        self.system_prompt = system_prompt
        self.chat_history = [{"role": "system", "content": system_prompt}]
        self.max_context_messages = max_context_messages

    def chat(self, user_prompt: str, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1) -> str:
        self.chat_history.append({"role": "user", "content": user_prompt})
        if len(self.chat_history) > self.max_context_messages:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-self.max_context_messages:]
        outputs = self.pipe(
            self.chat_history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        full_text = outputs[0]["generated_text"][-1]["content"]
        cleaned = re.split(r"\n\s*User:|Assistant:", full_text)[0].strip()
        self.chat_history.append({"role": "assistant", "content": cleaned})
        return cleaned
