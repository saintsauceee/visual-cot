import torch
from hf import load_model_from_hf
from sft import create_dataset, build_prompt

def board_to_str(board: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in board)

MODEL_NAME_STR = "Qwen/Qwen3-8B"  # base text-only model

tokenizer, model = load_model_from_hf(MODEL_NAME_STR)
model.eval()

train_puzzles, test_puzzles = create_dataset()

output_example = [
    {"name": "B", "direction": "left", "distance": 1},
    {"name": "C", "direction": "down", "distance": 3},
    {"name": "R", "direction": "right", "distance": 4},
]

prompt_example = build_prompt(
    board_to_str(train_puzzles[0].board), 
    train_puzzles[0].exit, 
    output_example
) + '\nSolution:\n'

print("\n\n")

print("Example Prompt:\n")
print(prompt_example)

inputs = tokenizer(
    prompt_example, 
    return_tensors="pt",
    truncation=True,
    max_length=2048,
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,      # greedy for now
        temperature=0.0,
    )

gen_ids = generated[0][inputs["input_ids"].shape[1]:]
gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

print("Generated Solution:\n")
print(gen_text)

print("\n\n")

print("Ground-truth Solution:\n")
print(test_puzzles[0].solution_moves)