from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"  # base text-only model

def load_model_from_hf(model_name: str = MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",      # fp16/bf16 on GPU if available
        device_map="auto"        # put on GPU(s) if present
    )
    return tokenizer, model