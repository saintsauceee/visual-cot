# The Limits of Text-Only Spatial Reasoning in LLMs

## Overview

Large Language Models (LLMs) frequently struggle with multi-step spatial planning, yet distinguishing visual perception failures from intrinsic reasoning deficits remains challenging. This project investigates the strict limits of text-only spatial reasoning using the **Rush Hour** sliding-block puzzle.

We evaluate **Gemini 2.5 Pro** on procedurally generated puzzles ranging from trivial (3-step) to complex (20-step) optimal solutions. By stripping away visual perception and providing text-only inputs, we isolate the reasoning engine to test the hypothesis that LLMs exhibit a distinct "planning horizon"—a complexity threshold beyond which state-tracking degrades.

## Methodology

### Environment: Rush Hour
We use the Rush Hour puzzle as a deterministic, fully verifiable benchmark.
-   **Goal**: Maneuver a specific "Red Car" to the exit on a 6x6 grid.
-   **Constraint**: Vehicles move only along their major axis; no overlapping allowed.
-   **Dataset**: 180 puzzles spanning 18 levels of difficulty (optimal solution lengths 3–20).

### Prompting Strategies
We compare three inference strategies:
1.  **Standard Zero-Shot**: Direct problem solving.
2.  **Few-Shot Chain-of-Thought (CoT)**: Providing in-context examples.
3.  **Augmented Visual CoT**: A structured diagnostic format we introduced. This requires the model to **explicitly textually render the board state** after every move. This enforces "in-context simulation" and allows us to verify the model's internal state tracking step-by-step.

## Experiments & Analysis Code

The core experimental scripts are located in the `experiments/` directory:

### `experiments/analysis.py`
This script drives the primary evaluation pipeline for standard prompting strategies (Zero-Shot and Few-Shot CoT).
-   **Model**: Evaluates **Gemini 2.5 Pro** across all 18 difficulty levels.
-   **Functionality**: Manages concurrent API requests, handles rate limiting across multiple API keys, and processes model responses.
-   **Output**: Generates raw solution data which is later processed to calculate success rates and validity metrics.

### `experiments/augmented.py`
This script implements the **Augmented Visual CoT** diagnostic experiment.
-   **Method**: Uses a specialized system prompt (`AUGMENTED_INSTRUCTIONS`) that enforces a strict output format: a dictionary containing both the move sequence and the **explicit ASCII board state** after every single move.
-   **Purpose**: This "forced simulation" allows us to verify the model's internal state tracking at every step. By comparing the model's predicted board state against the actual game engine state, we can detect exactly when and how the model's spatial reasoning drifts, even if the final answer happens to be correct (or incorrect).