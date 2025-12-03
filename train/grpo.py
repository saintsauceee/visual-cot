import ast
from typing import cast
from copy import deepcopy

from transformers import PreTrainedModel

import rh_data
from puzzle import RushHourSample, validate_solution
from hf import load_sft_full_precision
from sft import board_to_str, build_prompt

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

def rh_reward(puzzle: RushHourSample, gen_text: str) -> float:
    try:
        moves = ast.literal_eval(gen_text)
        if not isinstance(moves, list):
            return -1.0
    except Exception:
        return -1.0
    
    _, label = validate_solution(puzzle, moves, check_optimal=True)

    if label in ("TYPE_ERROR", "MISSING_KEYS"):
        return -1.0
    if label in ("CAR_NOT_FOUND", "INVALID_MOVE", "UNKNOWN_ERROR"):
        return -0.5
    if label == "UNSOLVED":
        return 0.1
    if label == "NOT_OPTIMAL":
        return 0.7
    if label == "OPTIMAL":
        return 1.0

    return 0.0

def build_rl_puzzles(
    min_level: int = 3, 
    max_level: int = 20
) -> list[RushHourSample]:
    _, train_puzzles, _ = rh_data.create_dataset(
        include_fsp=False,
        include_train=True,
        include_test=False,
    )
    return [
        puzzle for puzzle in train_puzzles
        if min_level <= getattr(puzzle, "min_num_moves", 999) <= max_level
    ]

def build_rl_dataset(
    min_level: int = 3,
    max_level: int = 20
) -> Dataset:
    puzzles = build_rl_puzzles(
        min_level=min_level,
        max_level=max_level,
    )

    rows = []
    for p in puzzles:
        board_str = board_to_str(p.board)
        prompt = build_prompt(board=board_str, exit=p.exit)

        prompt = prompt + "\nSolution:\n"

        rows.append(
            {
                "prompt": prompt,
                "id": p.id,
                "board": p.board,
                "exit": p.exit,
                "min_num_moves": p.min_num_moves,
            }
        )

    return Dataset.from_list(rows)

def grpo_rh_reward(prompts, completions, **kwargs):
    ids = kwargs.get("id")
    boards = kwargs.get("board")
    exits = kwargs.get("exit")
    min_moves = kwargs.get("min_num_moves")

    if ids is None or boards is None or exits is None or min_moves is None:
        return [0.0 for _ in completions]

    rewards: list[float] = []

    for comp, pid, b, ex, mn in zip(completions, ids, boards, exits, min_moves):
        exit_coord = tuple(ex) if isinstance(ex, (list, tuple)) else ex

        puzzle = RushHourSample(
            id=pid,
            board=b,
            exit=exit_coord,
            min_num_moves=mn,
            solution_moves=[],
        )

        rewards.append(float(rh_reward(puzzle, comp)))

    return rewards


def train_grpo(
    output_dir: str = "rl_out",
    seed: int = 42,
    min_level: int = 3,
    max_level: int = 20,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
):
    # no quantization for GRPO
    tokenizer, model = load_sft_full_precision()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_rl_dataset(min_level=min_level, max_level=max_level)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=4,
        generation_batch_size=4,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=cast(PreTrainedModel, model),
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=grpo_rh_reward,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.push_to_hub("saintsauce/Qwen2.5-7B-RushHour-RL")
    tokenizer.push_to_hub("saintsauce/Qwen2.5-7B-RushHour-RL")

if __name__ == "__main__":
    train_grpo()