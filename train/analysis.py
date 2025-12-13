import os
import ast
import argparse
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

MODEL_NAME = "gemini-2.5-pro"

KEYS = [
    "AIzaSyD-n1tOAJDleD1kZn7tneGinHgPZ1fZKcw",
    "AIzaSyCfLo1kXCmRLJqbvyU-r02GmWfiZ28SwGY",
    "AIzaSyDJ2fWvzbdHKN7T63-oa9553FCNLKe7Zys",
    "AIzaSyAuuHz4S4ESFdFYdt9BNWRlQpAglKns4YY",
    "AIzaSyA5qM4pCfr0iHFPIQWJljy67W2Ov-r5niY",
]

# map each level to an index into KEYS
LEVEL_TO_KEY_IDX = {
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 
    8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 
    13: 0, 14: 1, 15: 2, 16: 3, 17: 4, 
    18: 0, 19: 1, 20: 2,
}

clients = [
    genai.Client(api_key=k)
    for k in KEYS
]

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
                        include_thoughts=True
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

    thoughts, answer, duration, ttc = generate_output(
        prompt, 
        client
    )

    if verbose:
        print("Generated Solution:\n")
        print(answer)

        print(f"Inference time: {duration:.3f} seconds\n")

    try:
        gen_moves = ast.literal_eval(answer)
    except Exception as e:
        if verbose:
            print(f"failed to parse generated solution: {e}")
        return False, "PARSE_ERROR", thoughts, answer, duration, ttc, prompt
    
    valid, label = validate_solution(sample, gen_moves, check_optimal=True)

    if verbose:
        if valid:
            print("Solution is valid and optimal!")
        else:
            print(f"Solution is invalid: {label}")
    
    return valid, label, thoughts, answer, duration, ttc, prompt

def model_evaluate(
    include_fsp: bool,
    fsp_puzzles: list[RushHourSample],
    test_puzzles: list[RushHourSample]
):
    results: list[dict] = []

    # build few-shot examples grouped by level
    fsp_by_level: dict[int, list[tuple[str, list[dict[str, str | int]]]]] = defaultdict(list)
    if include_fsp:
        for puzzle in fsp_puzzles:
            level = getattr(puzzle, "min_num_moves", None)
            if level is not None:
                fsp_by_level[level].append(
                    (board_to_str(puzzle.board), puzzle.solution_moves)
                )

    def evaluate_level(level: int) -> list[dict]:
        # choose few-shot examples for this level
        if include_fsp:
            curr_level_examples = fsp_by_level.get(level, None)
            if not curr_level_examples:
                print(f"No few-shot examples found for level {level}, using zero-shot.")
                curr_level_examples = None
        else:
            curr_level_examples = None

        # collect puzzles for this level
        '''
        level_puzzles = [
            (idx, p) for idx, p in enumerate(test_puzzles)
            if getattr(p, "min_num_moves", None) == level
        ]
        '''
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
                    curr_level_examples,
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

if __name__ == "__main__":
    shots = 3

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

    # Example verbose evaluation of a single puzzle
    # _, _, test_puzzles = rh_data.create_dataset(False, False, True)

    # sample = test_puzzles[0]

    # evaluate_sample(
    #     sample,
    #     clients[0],
    #     verbose=True,
    #     few_shot_examples=None,
    # )