import time
import torch
from hf import load_model_from_hf
from sft import create_dataset, build_prompt

def board_to_str(board: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in board)

MODEL_NAME_STR = "Qwen/Qwen2.5-7B-Instruct"  # base text-only model

tokenizer, model = load_model_from_hf(MODEL_NAME_STR)
model.eval()

train_puzzles, test_puzzles = create_dataset()

def evaluate_sample(idx: int, message: str | None = None):
    sample = test_puzzles[idx]

    prompt_example = build_prompt(
        board_to_str(sample.board), 
        sample.exit
    ) + '\nSolution:\n'

    print("\n")

    print("Example Prompt:\n")
    print(prompt_example)

    inputs = tokenizer(
        prompt_example, 
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,      # greedy for now
            temperature=0.0,
        )
    end = time.perf_counter()

    gen_ids = generated[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print("\n\n")

    print("Generated Solution:\n")
    print(gen_text)

    print("\n\n")

    print("Ground-truth Solution:\n")
    print(sample.solution_moves)

    infer_time = end - start
    print(f"\nInference time: {infer_time:.3f} seconds")

if __name__ == "__main__":
    evaluate_sample(5)