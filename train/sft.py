import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, 
    AutoTokenizer, AutoModelForCausalLM
)

import rh_data
from puzzle import RushHourSample
from hf import BASE_MODEL_NAME

# Sample data format:
# {
#     "id": 1,
#     "exit": (3, 6),
#     "min_num_moves": 2,
#     "board": [
#       [".", ".", ".", ".", ".", "."],
#       [".", ".", "B", ".", ".", "."],
#       ["R", "R", "B", ".", ".", "."],
#       [".", ".", ".", ".", ".", "."],
#       [".", ".", ".", ".", ".", "."],
#       [".", ".", ".", ".", ".", "."]
#     ]
# }

# Sample train/test ids format:
# "3": [
#     31,
#     22,
#     11,
#     32,
#     13,
#     33,
#     16,
#     17,
#     9,
#     12,
#     29
# ]

output_example = [
    {"name": "B", "direction": "left", "distance": 1},
    {"name": "C", "direction": "down", "distance": 3},
    {"name": "R", "direction": "right", "distance": 4},
]

def board_to_str(board: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in board)

def build_prompt(
    board: str, 
    exit: str | tuple[int, int], 
    output_example: list[dict[str, str | int]] = output_example, 
    few_shot_examples: list[tuple[str, list[dict[str, str | int]]]] | None = None
) -> str:
    """ Build prompt for SFT sample """
    
    board_str = board
    example_str = repr(output_example)

    prompt = (
        "You have to solve the following 6x6 Rush Hour puzzle.\n"
        "Your goal is to move the red car out.\n"
        "On the board, 'R' designates the Red car.\n"
        f"The exit is located at {exit}.\n\n"
        "Rules:\n"
        "- The board is a 6x6 grid.\n"
        "- Each car is a horizontal or vertical line of identical letters.\n"
        "- Horizontal cars can only move left or right.\n"
        "- Vertical cars can only move up or down.\n"
        "- Cars cannot move outside the board.\n"
        "- Cars cannot pass through or overlap other cars.\n"
        "- Each move must be legal under these rules.\n\n"
        "Your output must be a Python list of moves.\n"
        "Each move is a dict with keys 'name', 'direction', 'distance'.\n"
        "Here is an example of the correct format:\n"
        f"{example_str}\n\n"
        "Return only a Python list of moves, no explanation.\n"
        "Provide only the text response with no bolding or formatting.\n"
    )

    if few_shot_examples:
        for board, sln in few_shot_examples:
            prompt += "\nExample Puzzle:\n"
            prompt += f"{board}\n"
            prompt += "\nSolution:\n"
            prompt += f"{repr(sln)}\n"
    
    prompt += (
        "\nNow, solve the following board:\n"
        f"{board_str}\n"
    )
            
    return prompt

def format_sample(puzzle: RushHourSample,) -> str:
    prompt = build_prompt(
        board=board_to_str(puzzle.board),
        exit=puzzle.exit,
        output_example=output_example,
    )

    target_list = repr(puzzle.solution_moves)

    return prompt + '\nSolution:\n' + target_list

def sft(
    model_name: str,
    raw_data: list[dict[str, str]], 
    train_args: TrainingArguments,
    output_dir: str = "sft_out",
    max_length: int = 2048
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

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

    model.push_to_hub("saintsauce/Qwen2.5-7B-RushHour-SFT")
    tokenizer.push_to_hub("saintsauce/Qwen2.5-7B-RushHour-SFT")

if __name__ == "__main__":
    _, train_puzzles, _ = rh_data.create_dataset(
        include_fsp=False,
        include_train=True,
        include_test=False,
    )
     
    raw_data = [
        {"text": format_sample(puzzle)} 
        for puzzle in train_puzzles
    ]

    args = TrainingArguments(
        output_dir="sft_out",

        num_train_epochs=8,
        learning_rate=5e-5,
        warmup_ratio = 0.03,
        lr_scheduler_type = "cosine",
        per_device_train_batch_size=4,      # fits 4-bit + LoRA
        gradient_checkpointing=False,        # saves VRAM
        
        weight_decay=0.0,
        bf16=True,                          # A100 supports bf16

        group_by_length=False,
        push_to_hub=False,
    )
    
    sft(
        BASE_MODEL_NAME,
        raw_data=raw_data,
        train_args=args
    )