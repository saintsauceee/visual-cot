import re
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
from sft import board_to_str

MODEL_NAME = "gemini-2.5-pro"

AUGMENTED_INSTRUCTIONS = """You are solving a Rush Hour puzzle on a 6x6 grid.

Output ONLY a Python literal dict with exactly these keys:
- "steps": a list of step dicts
- "final_moves": a list of move dicts

Each step dict MUST have:
- "move": a move dict with keys {"name","direction","distance"}
- "board": the ASCII board AFTER applying that move

Constraints:
- "final_moves" MUST equal [step["move"] for step in steps].
- Every move must be legal from the previous state.
- The puzzle is solved only if the red car 'R' reaches the exit.
- Do NOT output any explanation, markdown, or backticks. Only the Python literal dict.

Legality details:
- Rows 1..6 top->bottom, cols 1..6 left->right.
- Cars are contiguous; orientation is fixed.
- A move is legal only if all traversed destination cells are '.' and remain inside the 6x6 grid.
- The provided "board" must be exactly the result of applying the move to the previous board.

Move format:
{"name": <single-character car id>, "direction": "up"|"down"|"left"|"right", "distance": <int>}
"""

KEYS = [
    "AIzaSyA5tVNSBvsWxM3ElgL7yGm6J2nDDhXWOQA",
    "AIzaSyDtfoAtCbsX_XqlZfwDcQAi1NYxrVu_6hY",
    "AIzaSyAx4GymDhNoZKPJn3Fd3Na8Z75a1FmuPVA"
]

# map each level to an index into KEYS
LEVEL_TO_KEY_IDX = {
    3: 0, 4: 1, 5: 2,
    6: 0, 7: 1, 8: 2,
    9: 0, 10: 1, 11: 2,
    12: 0, 13: 1, 14: 2,
    15: 0, 16: 1, 17: 2,
    18: 0, 19: 1, 20: 2,
}

clients = [
    genai.Client(api_key=k)
    for k in KEYS
]

def strip_code_fences(s: str) -> str:
    s = s.strip()
    # Remove leading ```python / ``` and trailing ```
    s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()

def build_augmented_prompt(sample: RushHourSample) -> str:
    board = board_to_str(sample.board)
    exit_ = sample.exit

    prompt = (
        AUGMENTED_INSTRUCTIONS
        + "\n\nInitial board:\n"
        + board
        + "\n\nExit:\n"
        + str(exit_)
        + "\nExit is to the RIGHT of column 6 on the red car's row.\n"
        + "\n\nReturn the Python literal dict now."
    )
    return prompt

def generate_output(prompt: str, client: genai.Client):
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
                        include_thoughts=True,
                        # thinking_budget=THINKING_BUDGET,
                    )
                ),
            )
            end_time = time.time()
            duration = end_time - start_time

            # thinking token count
            ttc = response.usage_metadata.thoughts_token_count if response.usage_metadata else 0
            # print(f"[INFO] Gemini response received in {duration:.3f}s (TTC: {ttc})")

            thoughts, answer = "", ""
            for part in response.candidates[0].content.parts:  # type: ignore
                text = getattr(part, "text", None)
                if not text:
                    continue

                if getattr(part, "thought", False):
                    thoughts += text + "\n"
                else:
                    answer += text + "\n"

            return thoughts, answer, duration, ttc

        except Exception as e:
            if "409" not in str(e) and "429" not in str(e):
                raise

            if attempt == max_retries - 1:
                raise

            print(f"[WARN] Gemini error on attempt {attempt+1}/{max_retries}, retrying in {delay_seconds}s: {e}")
            time.sleep(delay_seconds)
    
    raise RuntimeError("generate_output: exhausted retries without success")

def evaluate_sample(
    sample: RushHourSample,
    client: genai.Client,
    verbose: bool = False, 
    few_shot_examples = None,
):
    prompt = build_augmented_prompt(sample)

    if verbose:
        print("=" * 80 + "\n")
        print("Prompt:\n")
        print(prompt + "\n")
        print("=" * 80 + "\n")

    thoughts, answer, duration, ttc = generate_output(prompt, client)

    if verbose:
        print("Generated Solution:\n")
        print(answer)
        print(f"Inference time: {duration:.3f} seconds\n")

    try:
        clean = strip_code_fences(answer)
        obj = ast.literal_eval(clean)
    except Exception as e:
        if verbose:
            print(f"failed to parse generated output: {e}")
        return False, "PARSE_ERROR", thoughts, answer, duration, ttc, prompt
    
    if not isinstance(obj, dict):
        return False, "BAD_TOPLEVEL_TYPE", thoughts, answer, duration, ttc, prompt

    if "steps" not in obj or not isinstance(obj["steps"], list):
        return False, "MISSING_STEPS", thoughts, answer, duration, ttc, prompt

    if "final_moves" not in obj:
        return False, "MISSING_FINAL_MOVES", thoughts, answer, duration, ttc, prompt


    gen_moves = obj["final_moves"]
    if not isinstance(gen_moves, list):
        return False, "BAD_FINAL_MOVES_TYPE", thoughts, answer, duration, ttc, prompt
    
    valid, label = validate_solution(sample, gen_moves, check_optimal=True)

    if verbose:
        if valid:
            print("Solution is valid and optimal!")
        else:
            print(f"Solution is invalid: {label}")
    
    return valid, label, thoughts, answer, duration, ttc, prompt

def model_evaluate(
    include_fsp: bool, # don't use
    fsp_puzzles: list[RushHourSample], # don't use
    test_puzzles: list[RushHourSample]
):
    results: list[dict] = []

    def evaluate_level(level: int) -> list[dict]:
        level_puzzles = [
            (p.id, p) for p in test_puzzles
            if getattr(p, "min_num_moves", None) == level
        ]
        if not level_puzzles:
            return []

        # pick client/key for this level
        key_idx = LEVEL_TO_KEY_IDX[level]
        client_for_level = clients[key_idx]

        level_results: list[dict] = []

        # inner pool: parallel over puzzles in this level
        with ThreadPoolExecutor(max_workers=len(level_puzzles)) as executor:
            futures = {
                executor.submit(
                    evaluate_sample,
                    puzzle,
                    client_for_level,
                    False,
                    None,
                ): (idx, puzzle)
                for idx, puzzle in level_puzzles
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating Level {level}",
                unit="puzzle"
            ):
                idx, puzzle = futures[future]
                try:
                    valid, label, thoughts, answer, duration, ttc, prompt = future.result()
                except Exception as e:
                    valid, label, duration, thoughts, answer, ttc, prompt = False, f"EXCEPTION: {e}", 0.0, "", "", 0, ""

                level_results.append({
                    "idx": idx,
                    "level": level,
                    "valid": valid,
                    "label": label,
                    "time": duration,
                    "ttc": ttc,
                    "prompt": prompt,
                    "thoughts": thoughts,
                    "answer": answer,
                })

        return level_results

    levels = list(range(3, 21))

    # outer pool: parallel over levels
    with ThreadPoolExecutor(max_workers=len(levels)) as level_executor:
        level_futures = {
            level_executor.submit(evaluate_level, level): level
            for level in levels
        }

        for future in as_completed(level_futures):
            level_results = future.result()
            results.extend(level_results)

    return results

def aggregate(results):
    by_level = defaultdict(list)
    ttc_by_level = defaultdict(list)
    label_counts = Counter()
    total_time = 0.0
    sample_count = 0

    for r in results:
        total_time += r.get("time", 0.0)
        if r["level"] is not None:
            by_level[r["level"]].append(r["valid"])
            ttc_by_level[r["level"]].append(r.get("ttc", 0))
        label_counts[r["label"]] += 1

        sample_count += 1

    levels = sorted(by_level.keys())
    success = [sum(v) / len(v) for v in (by_level[l] for l in levels)] if levels else []
    avg_ttc = [sum(v) / len(v) for v in (ttc_by_level[l] for l in levels)] if levels else []

    avg_time = total_time / sample_count if sample_count > 0 else 0.0
    print(f"Average inference time: {avg_time:.3f} seconds over {sample_count} samples")

    return levels, success, label_counts, avg_ttc

def plot_results(levels, success, label_counts, avg_ttc, shots):
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

    plt.figure() # Average TTC per level

    ttc_map = dict(zip(levels, avg_ttc)) if levels else {}
    ttc_for_all_levels = [ttc_map.get(l, 0.0) for l in full_levels]

    plt.bar(full_levels, ttc_for_all_levels)
    plt.xlabel("Puzzle Level (Minimum Moves)")
    plt.ylabel("Average Thinking Token Count")
    
    plt.title(f"Average Thinking Token Count by Puzzle Level - Gemini 2.5 Pro ({shot_str})")
    plt.xticks(full_levels)
    plt.tight_layout()
    
    fname3 = f"avg_ttc_by_level_{shot_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fname3, bbox_inches='tight')
    print(f"Saved plot: {os.path.abspath(fname3)}")  
    plt.show()

def run_experiment():
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

    import json
    output_file = f"results_{shots}shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_file}")
    
    levels, success, label_counts, avg_ttc = aggregate(results)
    plot_results(levels, success, label_counts, avg_ttc, shots)

if __name__ == "__main__":
    run_experiment()

    # Example verbose evaluation of a single puzzle
    # _, _, test_puzzles = rh_data.create_dataset(False, False, True)

    # sample = test_puzzles[10]

    # evaluate_sample(
    #     sample,
    #     clients[0],
    #     verbose=True,
    #     few_shot_examples=None,
    # )