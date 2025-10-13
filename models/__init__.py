from .llama import Llama , LlamaChat
from .mistral import Mistral, MistralChat
from .flux import FluxT2I, FluxT2IChat
from .stable_diffusion import SDT2I, SDT2IChat, SDT2IChatWithLlama

models = {
    "llama": Llama,
    "llama_chat": LlamaChat,
    "mistral": Mistral,
    "mistral_chat": MistralChat,
    "flux": FluxT2I,
    "flux_chat": FluxT2IChat,
    "sd": SDT2I,
    "sd_chat": SDT2IChat,
    "sd_chat_llama": SDT2IChatWithLlama,
}

def get_model(name, **kwargs):
    """Model"""
    return models[name.lower()](**kwargs)
