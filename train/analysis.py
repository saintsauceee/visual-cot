import os
import ast
import time
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

import rh_data
from puzzle import RushHourSample, validate_solution
from sft import board_to_str, build_prompt

MAX_WORKERS = 10

API_KEYS = [
    "AIzaSyDkUUoVBNHqMQWquADrZ4v4VOUwQuvYCJY",
    "AIzaSyAfIriooSuGAxuLz-le-k5wxfQ4PFWgz8I"
]

GOOGLE_API_KEY = "AIzaSyAfIriooSuGAxuLz-le-k5wxfQ4PFWgz8I"
MODEL_NAME = "gemini-2.5-pro"

client = genai.Client(
  api_key=GOOGLE_API_KEY
)

def generate_output(prompt: str):
    max_retries = 3
    delay_seconds = 5

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True
                    )
                ),
            )
            end_time = time.time()
            duration = end_time - start_time

            thoughts, answer = "", ""
            for part in response.candidates[0].content.parts:  # type: ignore
                text = getattr(part, "text", None)
                if not text:
                    continue

                if getattr(part, "thought", False):
                    thoughts += text + "\n"
                else:
                    answer += text + "\n"

            return thoughts, answer, duration

        except Exception as e:
            if "409" not in str(e):
                raise

            if attempt == max_retries - 1:
                raise

            print(f"[WARN] Gemini 409 on attempt {attempt+1}/{max_retries}, retrying in {delay_seconds}s: {e}")
            time.sleep(delay_seconds)
    
    raise RuntimeError("generate_output: exhausted retries without success")

def evaluate_sample(
    sample: RushHourSample,
    verbose: bool = False, 
    few_shot_examples: list[tuple[str, list[dict[str, str | int]]]] | None = None,
):
    prompt = build_prompt(
        board_to_str(sample.board), 
        sample.exit,
        few_shot_examples=few_shot_examples
    ) + '\nSolution:'

    if verbose:
        print("=" * 80 + "\n")
        print("Prompt:\n")
        print(prompt + "\n")
        print("=" * 80 + "\n")

    thoughts, answer, duration = generate_output(prompt)

    if verbose:
        print("Generated Solution:\n")
        print(answer)

        print(f"Inference time: {duration:.3f} seconds\n")

    try:
        gen_moves = ast.literal_eval(answer)
    except Exception as e:
        if verbose:
            print(f"failed to parse generated solution: {e}")
        return False, "PARSE_ERROR", "", "", 0.0
    
    valid, label = validate_solution(sample, gen_moves, check_optimal=True)

    if verbose:
        if valid:
            print("Solution is valid and optimal!")
        else:
            print(f"Solution is invalid: {label}")
    
    return valid, label, thoughts, answer, duration

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
                fsp_by_level[level].append(
                    (board_to_str(puzzle.board), puzzle.solution_moves)
                )

    for level in range(3, 21):

        # choose few-shot examples for this level
        if include_fsp:
            curr_level_examples = fsp_by_level.get(level, None)
            if not curr_level_examples:
                print(f"No few-shot examples found for level {level}, using zero-shot.")
                curr_level_examples = None
        else:
            curr_level_examples = None
        
        level_puzzles = [
            (idx, p) for idx, p in enumerate(test_puzzles)
            if getattr(p, "min_num_moves", None) == level
        ]
        if not level_puzzles:
            continue

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    evaluate_sample,
                    puzzle,
                    False,
                    curr_level_examples,
                ): (idx, puzzle)
                for idx, puzzle in level_puzzles
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating Level {level}", unit="puzzle"):
                idx, puzzle = futures[future]
                try:
                    valid, label, thoughts, answer, duration = future.result()
                except Exception as e:
                    valid, label, duration = False, f"EXCEPTION: {e}", 0.0

                results.append({
                    "idx": idx,
                    "level": level,
                    "valid": valid,
                    "label": label,
                    "time": duration,
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

def plot_results(levels, success, label_counts, shots):
    plt.figure() # Success rate per level (styled like label distribution plot)

    success_map = dict(zip(levels, success)) if levels else {}
    full_levels = list(range(3, 21))
    success_for_all_levels = [success_map.get(l, 0.0) for l in full_levels]

    plt.bar(full_levels, success_for_all_levels)
    plt.xlabel("Puzzle Level (Minimum Moves)")
    plt.ylabel("Average Success Rate")
    
    shot_str = f"{shots}-shot"
    plt.title(f"Average Success Rate by Puzzle Level - Gemini 2.5 Pro ({shot_str})")
    plt.ylim(0, 1)
    plt.xticks(full_levels)
    plt.tight_layout()
    
    fname1 = f"success_rate_by_level_{shot_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fname1, bbox_inches='tight')
    print(f"Saved plot: {os.path.abspath(fname1)}")
    
    plt.show()

    plt.figure() # Label distribution plot

    labels = list(label_counts.keys())
    counts = [label_counts[l] for l in labels]

    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Count")
    
    plt.title(f"Label Distribution - Gemini 2.5 Pro ({shot_str})")
    plt.tight_layout()

    fname2 = f"label_distribution_{shot_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fname2, bbox_inches='tight')
    print(f"Saved plot: {os.path.abspath(fname2)}")
    
    plt.show()

if __name__ == "__main__":
    shots = 0

    if shots > 0:
        fsp_puzzles, _, test_puzzles = rh_data.create_dataset(True, False, True)
    else:
        _, _, test_puzzles = rh_data.create_dataset(False, False, True)

    results = model_evaluate(
        include_fsp = True if shots > 0 else False,
        fsp_puzzles = fsp_puzzles if shots > 0 else [],
        test_puzzles = test_puzzles
    )
    
    levels, success, label_counts = aggregate(results)
    plot_results(levels, success, label_counts, shots)