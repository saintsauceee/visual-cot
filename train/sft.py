from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils.hf import load_model_from_hf

def sft(
    model_name: str, 
    raw_data: list[dict[str, str]], 
    train_args: TrainingArguments,
    output_dir: str = "sft_out",
    max_length: int = 2048
) -> None:
    """ SFT Pipeline """
    
    tokenizer, model = load_model_from_hf(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    data = Dataset.from_list(raw_data)

    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=max_length
        )

    ds = data.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = TrainingArguments(
        output_dir="sft_out",
        per_device_train_batch_size=4,      # fits 4-bit + LoRA
        gradient_accumulation_steps=8,
        max_steps=5000,
        learning_rate=2e-4,                 # for LoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,                   # LoRA: usually 0
        bf16=True,                          # A100 supports bf16
        gradient_checkpointing=True,        # saves VRAM
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        group_by_length=True,               # faster packing by length
    )
    
    sft("Qwen/Qwen3-8B", [{"text": "Include the question, board, answer, ..."}], train_args=args)