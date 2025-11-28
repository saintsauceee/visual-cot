import ast
import time
import torch
from copy import deepcopy
from hf import load_model_from_hf
from puzzle import CarNotFound, InvalidMove, RushHourPuzzle
from sft import create_dataset, build_prompt
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

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
            print(f"move {i} is not a dict: {m}")
            return False, "TYPE_ERROR"
        if not all(k in m for k in ("name", "direction", "distance")):
            print(f"move {i} is missing keys: {m}")
            return False, "MISSING_KEYS"
        try:
            sim.move(m["name"], m["direction"], int(m["distance"]))
        except CarNotFound as e:
            print(f"move {i} car not found: {e}")
            return False, "CAR_NOT_FOUND"
        except InvalidMove as e:
            print(f"move {i} invalid: {e}")
            return False, "INVALID_MOVE"
        except Exception as e:
            print(f"move {i} unknown error: {e}")
            return False, "UNKNOWN_ERROR"

    if not sim.solved():
        return False, "UNSOLVED"

    if check_optimal and len(moves) != puzzle.min_num_moves:
        return True, "NOT_OPTIMAL"

    return True, "OPTIMAL"

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

    infer_time = end - start

    if verbose:
        print("\n")

        print("Generated Solution:\n")
        print(gen_text)
        print("\n")

        print("Ground-truth Solution:\n")
        print(sample.solution_moves)
        print("\n")

        print(f"Inference time: {infer_time:.3f} seconds\n")

    try:
        gen_moves = ast.literal_eval(gen_text)
    except Exception as e:
        if verbose:
            print(f"failed to parse generated solution: {e}")
        return False, "PARSE_ERROR", 0.0
    
    valid, label = validate_solution(sample, gen_moves, check_optimal=True)

    if verbose:
        if valid:
            print("Solution is valid and optimal!")
        else:
            print(f"Solution is invalid: {label}")
    
    return valid, label, infer_time

def model_evaluate():
    results = []
    for idx, puzzle in enumerate(test_puzzles):
        valid, label, infer_time = evaluate_sample(idx, verbose=False)
        level = getattr(puzzle, "min_num_moves", None)
        results.append({
            "idx": idx,
            "level": level,
            "valid": valid,
            "label": label,
            "time": infer_time,
        })
    return results

def aggregate(results):
    by_level = defaultdict(list)
    label_counts = Counter()

    total_time = 0.0
    sample_count = 0

    for r in results:
        if r["label"] != "PARSE_ERROR":
            total_time += r["time"]
            not_parse_err_count += 1
        if r["level"] is not None:
            by_level[r["level"]].append(r["valid"])
        label_counts[r["label"]] += 1

        sample_count += 1

    levels = sorted(by_level.keys())
    success = [sum(v) / len(v) for v in (by_level[l] for l in levels)]

    print(f"Average inference time: {total_time / sample_count:.3f} seconds")

    return levels, success, label_counts

if __name__ == "__main__":
    tokenizer, model = load_model_from_hf()
    model.eval()

    _, test_puzzles = create_dataset()

    results = model_evaluate()
    levels, success, label_counts = aggregate(results)

    # SR per level
    plt.figure()
    plt.bar(levels, success)
    plt.xlabel("Level")
    plt.ylabel("Success Rate")
    plt.title("Qwen2.5 (0SP) - Success Rate by Level")
    plt.show()

    # Label dist.
    plt.figure()    
    labels = list(label_counts.keys())
    counts = [label_counts[l] for l in labels]
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Qwen2.5 (0SP) - Label Distribution")
    plt.tight_layout()
    plt.show()