import ast
import os
import copy
import json
import time
import tqdm
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

key = "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"
client = OpenAI(api_key=key)

results = []
nbr_threads = 20

with open("./results_3shot_20251212_191639.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract puzzles that weren't solved correctly
failed_puzzles = [puzzle for puzzle in data if not puzzle.get("valid", False)]


# PROMPT FOR EVALUATION
def create_cot_eval_prompt(puzzle):
    prompt = f'''You are evaluating an LLM's solution to a Rush Hour puzzle.

    You must evaluate the response using the following criteria:

    1. Factuality:
    Factuality evaluates if the factual information can be grounded in reliable sources.
    (e.g. board size, car orientation, occupied squares, correct exit)?

    2. Validity:
        Validity evaluates if a reasoning step contains no logical errors.
        (e.g. no illegal moves, no colisions, cars stay within bounds).

    3. Coherence:
    Coherence measures if a reasoning step’s precon- ditions are satisfied by the previous steps 
    (e.g. moving a car but not explaining why it had to be moved, a car changes position with no corresponding move).

    4. Utility
    How useful is the response for actually solving the puzzle?
    A solution with illegal moves or incorrect outcomes has low utility.
    (e.g. making unnecessary moves, failing to solve the puzzle).

    Scores must be on a 1–5 scale, where:
    - 5 = excellent
    - 3 = partially correct / fair
    - 1 = very poor 

    Below is an example showing how these criteria should be applied.

    --- BEGIN EXAMPLE ---
    Puzzle Prompt:
    {failed_puzzles[0]["prompt"]}

    LLM Reasoning and Final Answer:
    {failed_puzzles[0]["thoughts"]}

    Human Evaluation

    Factuality Score (1–l5): 5
    Explanation: All cars in the thought process are existent on the baord and the described positions are accurate.

    Validity Score (1–5): 2
    Explanation: While the red car is moved correctly, car 'D' is moved illegally, as it collides with car 'C'. One invalid move out of a total of 3 minimal moves is bad.

    Coherence Score (1–5): 5
    Explanation: All the steps logically follow from one another, with no unexplained position changes.

    Utility Score (1–5): 4
    Explanation: The solution cannot be used to solve the puzzle due to invalid moves, but the overall strategy is sound and the red car is moved towards the exit.

    --- END EXAMPLE ---

    
    Now evaluate the following LLM response using the same criteria, definitions, and scoring standards.

    Puzzle Prompt: {puzzle["prompt"]}
    LLM Reasoning and Final Answer: {puzzle["thoughts"]}

    IMPORTANT:
    - Use the definitions above.
    - Be consistent with the given example.
    - Do not provide explanations for the scores.
    - The output must strictly follow the format below.

    RESPONSE FORMAT EXAMPLE IN JSON:
    {{
    "Factuality": 5,
    "Validity": 2,
    "Coherence": 5,
    "Utility": 4
    }}


    '''
    return prompt


# TESTING WITH ONE SAMPLE
# ---------- SINGLE-SAMPLE SANITY CHECK ----------
test_puzzle = failed_puzzles[0]
test_prompt = create_cot_eval_prompt(test_puzzle)

response = client.responses.create(
    model="gpt-5",
    input=test_prompt,
)

print("=== RAW RESPONSE OBJECT ===")
print(response)

print("\n=== output_text ===")
print(repr(response.output_text))


with ThreadPoolExecutor(max_workers=nbr_threads) as executor:
    future_to_puzzle = {}

    for puzzle in failed_puzzles:
        prompt = create_cot_eval_prompt(puzzle)
        future = executor.submit(
            client.responses.create,
            model="gpt-5",
            input=prompt
        )
        future_to_puzzle[future] = puzzle

    for future in tqdm(as_completed(future_to_puzzle), total=len(future_to_puzzle)):
        try:
            response = future.result()
            puzzle = future_to_puzzle[future]

            results.append({
                "puzzle_id": puzzle.get("idx"),
                "level": puzzle.get("level"),
                "raw_scores": response.output_text
            })
        except Exception as e:
            print(f"Error processing a puzzle: {e}")



# ---------- SAVE RAW RESULTS ----------
date = time.strftime("_%Y%m%d_%H%M%S", time.localtime())
with open(f"cot_evaluation_results{date}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)


# ---------- PARSE + AVERAGES ----------
rows = []

for puzzle in results:
    try:
        scores = json.loads(puzzle["raw_scores"])
        rows.append({
            "level": puzzle["level"],
            "Factuality": scores["Factuality"],
            "Validity": scores["Validity"],
            "Coherence": scores["Coherence"],
            "Utility": scores["Utility"],
        })
    except Exception as e:
        print("Failed to parse:", puzzle["raw_scores"])

df = pd.DataFrame(rows)
df = df.dropna(subset=["level"])
df["level"] = df["level"].astype(int)

plt.figure(figsize=(8, 5))

level_means = df.groupby("level").mean()

for col in ["Factuality", "Validity", "Coherence", "Utility"]:
    plt.plot(level_means.index, level_means[col], marker="o", label=col)

plt.xlabel("Puzzle Level")
plt.ylabel("Average Score")
plt.title("CoT Evaluation Scores vs Puzzle Level")
plt.ylim(1, 5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cot_scores_vs_level.png", dpi=300)
plt.show()
