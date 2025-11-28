import ast
import time
import torch
from copy import deepcopy
from hf import load_model_from_hf
from puzzle import RushHourPuzzle
from sft import create_dataset, build_prompt

def board_to_str(board: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in board)

def validate_solution(puzzle, moves, check_optimal=False):
    sim = RushHourPuzzle(
        id=puzzle.id,
        exit=puzzle.exit,
        min_num_moves=puzzle.min_num_moves,
        board=deepcopy(puzzle.board),
    )

    for i, m in enumerate(moves, 1):
        if not isinstance(m, dict):
            return False, f"bad move {i}"
        if not all(k in m for k in ("name", "direction", "distance")):
            return False, f"bad move {i}"
        try:
            sim.move(m["name"], m["direction"], int(m["distance"]))
        except Exception as e:
            return False, f"invalid {i}: {e}"

    if not sim.solved():
        return False, "not solved"

    if check_optimal and len(moves) != puzzle.min_num_moves:
        return False, "not optimal"

    return True, "ok"

def evaluate_sample(idx: int, verbose: bool = False):
    sample = test_puzzles[idx]

    prompt_example = build_prompt(
        board_to_str(sample.board), 
        sample.exit
    ) + '\nSolution:\n'

    if verbose:
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
        )
    end = time.perf_counter()

    gen_ids = generated[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if verbose:
        print("\n")

        print("Generated Solution:\n")
        print(gen_text)

        print("\n")

        print("Ground-truth Solution:\n")
        print(sample.solution_moves)

        infer_time = end - start
        print(f"\nInference time: {infer_time:.3f} seconds")
    
    if verbose:
        print("Validating solution...")

    try:
        gen_moves = ast.literal_eval(gen_text)
    except Exception as e:
        if verbose:
            print(f"failed to parse generated solution: {e}")
        return False, 0.0
    
    valid, msg = validate_solution(sample, gen_moves, check_optimal=True)

    if verbose:
        if valid:
            print("Solution is valid and optimal!")
        else:
            print(f"Solution is invalid: {msg}")
    
    return valid, infer_time

if __name__ == "__main__":
    tokenizer, model = load_model_from_hf()
    model.eval()

    train_puzzles, test_puzzles = create_dataset()

    valid, infer_time = evaluate_sample(5, verbose=True)