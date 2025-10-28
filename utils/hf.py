from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"  # base text-only model
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",      # fp16/bf16 on GPU if available
    device_map="auto"        # put on GPU(s) if present
)