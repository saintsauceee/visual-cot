import argparse
import ast
import time
import torch
from copy import deepcopy
from hf import load_model_from_hf
from puzzle import CarNotFound, InvalidMove, RushHourPuzzle
from sft import create_dataset, build_prompt
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime
from rh import RushHourSample

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

def evaluate_sample(
    sample,
    verbose: bool = False, 
    few_shot_examples: list[tuple[str, list[dict[str, str | int]]]] | None = None,
    print_output: bool = False,
):
    prompt_example = build_prompt(
        board_to_str(sample.board), 
        sample.exit,
        few_shot_examples=few_shot_examples
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

    if print_output:
        print(f"{80 * '='}")
        print("GENERATED OUTPUT:")
        print(f"{80 * '='}")

        print("\nGenerated Text:\n")
        print(gen_text)
        print("\n")

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
        print(f"Parsed generated moves for puzzle {sample.id}: {gen_moves}")
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

def model_evaluate(
    include_fsp: bool,
    fsp_puzzles: list[RushHourSample],
    test_puzzles: list[RushHourSample]
):
    results = []

    fsp_by_level = defaultdict(list)
    if include_fsp:
        for puzzle in fsp_puzzles:
            level = getattr(puzzle, "min_num_moves", None)
            if level is not None:
                fsp_by_level[level].append((board_to_str(puzzle.board), puzzle.solution_moves))

    for level in range(3, 21):
        if include_fsp:
            curr_level_examples = fsp_by_level.get(level, None)
            if not curr_level_examples:
                print(f"No few-shot examples found for level {level}, using zero-shot for this level.")
                curr_level_examples = None
        else:
            curr_level_examples = None

        for idx, puzzle in enumerate(tqdm(test_puzzles, desc=f"Evaluating Level {level}", unit="puzzle"), start=0):
            puzzle_level = getattr(puzzle, "min_num_moves", None)
            if puzzle_level != level:
                continue
            
            valid, label, infer_time = evaluate_sample(
                puzzle, 
                verbose=False, 
                few_shot_examples=curr_level_examples
            )

            results.append({
                "idx": idx,
                "level": puzzle_level,
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
        total_time += r.get("time", 0.0)
        if r["level"] is not None:
            by_level[r["level"]].append(r["valid"])
        label_counts[r["label"]] += 1

        sample_count += 1

    levels = sorted(by_level.keys())
    success = [sum(v) / len(v) for v in (by_level[l] for l in levels)] if levels else []

    avg_time = total_time / sample_count if sample_count > 0 else 0.0
    print(f"Average inference time: {avg_time:.3f} seconds over {sample_count} samples")

    return levels, success, label_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shots",
        type=int,
        default=3,
        help="Number of few-shot examples to include in the prompt (0 or 3) or -1 for sample",
    )
    args = parser.parse_args()

    if args.shots not in [-1, 0, 3]:
        raise ValueError("--shots must be -1, 0 or 3")

    model_name_str = "saintsauce/Qwen2.5-7B-RushHour-SFT"
    tokenizer, model = load_model_from_hf(model_name=model_name_str)
    model.eval()

    fsp_puzzles, _, test_puzzles = create_dataset(True, False, True)

    if args.shots == -1:
        sample_idx = 5
        if sample_idx < len(test_puzzles):
            puzzle = test_puzzles[sample_idx]
            print(f"\n{'=' * 80}")
            print(f"SAMPLE MODE - Puzzle Index: {sample_idx}")
            print(f"{'=' * 80}\n")

            fsp_formatted = [(board_to_str(puzzle.board), puzzle.solution_moves) for puzzle in fsp_puzzles[0:3]]
            
            prompt = build_prompt(
                board_to_str(puzzle.board), 
                puzzle.exit,
                few_shot_examples=fsp_formatted
            )
            
            print(f"\n{"=" * 80}")
            print("PROMPT (Zero-shot):")
            print(f"{"=" * 80}\n")
            print(prompt)
            
            valid, label, infer_time = evaluate_sample(
                puzzle, 
                verbose=False, 
                few_shot_examples=fsp_formatted,
                print_output=True
            )
            
            print(f"\n{"=" * 80}")
            print("GROUND TRUTH SOLUTION:")
            print(f"{"=" * 80}\n")
            print(puzzle.solution_moves)

            print(f"\nValidation Result: {label}")
            print(f"Inference time: {infer_time:.3f} seconds")
        else:
            print(f"Error: Sample index {sample_idx} out of range (test set has {len(test_puzzles)} puzzles)")
    else:
        results = model_evaluate(
            include_fsp=True if args.shots > 0 else False,
            fsp_puzzles=fsp_puzzles,
            test_puzzles=test_puzzles
        )
        
        levels, success, label_counts = aggregate(results)

        # SR per level
        plt.figure()
        plt.bar(levels, success)
        plt.xlabel("Puzzle Level (Minimum Moves)")
        plt.ylabel("Average Success Rate")
        shot_str = f"{args.shots}-shot"
        plt.title(f"Average Success Rate by Puzzle Level - {model_name_str} ({shot_str})")
        plt.ylim(0, 1)
        plt.xticks(range(3, 21))
        fname1 = f"success_rate_by_level_{shot_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fname1, bbox_inches='tight')
        print(f"Saved plot: {os.path.abspath(fname1)}")
        plt.show()

        # Label distribution
        plt.figure()    
        labels = list(label_counts.keys())
        counts = [label_counts[l] for l in labels]
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title(f"Label Distribution - {model_name_str} ({shot_str})")
        plt.tight_layout()
        fname2 = f"label_distribution_{shot_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fname2, bbox_inches='tight')
        print(f"Saved plot: {os.path.abspath(fname2)}")
        plt.show()