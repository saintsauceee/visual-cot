import ast
from train.eval import validate_solution

def rushhour_reward(puzzle, moves):
    ok, label = validate_solution(puzzle, moves, check_optimal=True)

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

# wrap validate solution to take text input directly
def reward_from_text(puzzle, gen_text: str) -> float:
    try:
        moves = ast.literal_eval(gen_text)
        if not isinstance(moves, list):
            return -1.0
    except Exception:
        return -1.0
    return rushhour_reward(puzzle, moves)