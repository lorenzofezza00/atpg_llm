
from transformers import pipeline
import torch

# With GPT we have many possibilities:
# * API: multimodal models but expensive
# * Recet open weight models: gpt-oss-20b (+16GB VRAM) and gpt-oss-120b (pesante +60GB VRAM, multi GPU)
#   * Problem: requires python 3.12 (noi abbiamo fino a 3.11)

def get_gpt_oss(model_id = "openai/gpt-oss-20b"):
        pipe = pipeline(
            "text-generation",
            model=model_id,
            dtype="auto",
            device_map="auto",
        )
        
        return pipe

class GPTOSS:
    def __init__(self, model_id= "openai/gpt-oss-20b"):
        # Costruttore: inizializza gli attributi
        self.pipe = get_gpt_oss(model_id)

    def make_std_inference(self, messages):
        # messages = [
        #     {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
        # ]

        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        # print(outputs[0]["generated_text"][-1])
        return outputs[0]["generated_text"][-1]


# from openai import OpenAI

# def get_gpt_client(api_key=""):
#     client = OpenAI(
#     api_key=api_key
#     )
#     return client, make_gpt_inference

# def make_gpt_inference(client, history=None, prompt=""):
#     if history is not None:
#         history.append(prompt)
#         prompt_context = " ".join(history)

#         # --- ChatGPT rielabora il prompt ---
#         chat_completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "Sei un assistente che trasforma prompt generici in descrizioni dettagliate e artistiche per generare immagini."
#                 },
#                 {"role": "user", "content": prompt_context}
#             ],
#             temperature=0.7,
#         )

#         refined_prompt = chat_completion.choices[0].message.content.strip()
#         print(f"\n🎨 Prompt rielaborato da ChatGPT:\n ➡️ {refined_prompt}\n")
#         return refined_prompt
#     else:
#         resp = client.responses.create(
#             model="gpt-4o",
#             instructions="You are a coding assistant that speaks like a pirate.",
#             input=prompt,
#         )
        
#         print(f"\n🎨 Prompt rielaborato da ChatGPT:\n ➡️ {resp.output_text}\n")
#         return resp.output_text