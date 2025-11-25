import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.rh_exceptions import CarNotFound, InvalidMove
from data.rh_puzzle import RushHourPuzzle

# Regex to capture simple natural-language move descriptions.
# Examples it matches:
# - "Red left 2"
# - "Move R to the right by 1"
# - "slide car B up 3 spaces"
MOVE_PATTERN = re.compile(
    r"""
    (?:(?:move|slide|shift)\s+)?                # optional leading verb
    (?:the\s+)?                                 # optional 'the'
    (?:car\s+)?                                 # optional 'car' before the id
    (?P<car>[A-Za-z]+)                          # car identifier or color name
    (?:\s+car)?                                 # optional 'car' after the id
    \s+(?:to\s+the\s+)?                         # connector before direction
    (?P<direction>left|right|up|down|           # direction keywords
        leftward|rightward|leftwards|rightwards)
    \b
    (?:\s+(?:by|for|to)?\s*)?                   # optional filler before distance
    (?P<distance>-?\d+)                         # move distance
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DIRECTION_ALIASES = {
    "leftward": "left",
    "leftwards": "left",
    "rightward": "right",
    "rightwards": "right",
}


@dataclass
class ParseResult:
    moves: List[Dict[str, object]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    valid_moves: int = 0
    solved: bool = False
    error: Optional[str] = None


@dataclass
class ScoreResult:
    label: str
    score: float


def parse_cot_to_moves(cot_text: str) -> ParseResult:
    """
    Parse natural-language CoT text into structured moves.

    Args:
        cot_text: Free-form chain-of-thought description.

    Returns:
        ParseResult containing extracted moves and any parse errors.
    """

    moves: List[Dict[str, object]] = []
    errors: List[str] = []

    for match in MOVE_PATTERN.finditer(cot_text):
        car_raw = match.group("car").strip()
        direction_raw = match.group("direction").lower()
        distance_raw = match.group("distance")

        try:
            distance = int(distance_raw)
        except ValueError:
            errors.append(f"Could not parse distance '{distance_raw}' in segment: {match.group(0)}")
            continue

        direction = _DIRECTION_ALIASES.get(direction_raw, direction_raw)
        car_name = car_raw[0].upper()

        moves.append({"name": car_name, "direction": direction, "distance": distance})

    if not moves:
        errors.append("No moves could be parsed from the provided text.")

    return ParseResult(moves=moves, errors=errors)


def simulate_moves(puzzle_dict: Dict[str, object], move_list: List[Dict[str, object]]) -> SimulationResult:
    """
    Simulate the parsed moves on a RushHourPuzzle instance.

    Args:
        puzzle_dict: Raw puzzle data loaded from the dataset.
        move_list: List of move dicts with keys: name, direction, distance.

    Returns:
        SimulationResult capturing progress, solved status, and any errors.
    """

    puzzle_obj = RushHourPuzzle(
        id=puzzle_dict["id"],
        exit=puzzle_dict["exit"],
        min_moves=puzzle_dict.get("min_moves"),
        board=copy.deepcopy(puzzle_dict["board"]),
    )

    valid_moves = 0
    error: Optional[str] = None

    for move in move_list:
        try:
            puzzle_obj.move(move["name"], move["direction"].lower(), int(move["distance"]))
            valid_moves += 1
            if puzzle_obj.solved():
                break
        except InvalidMove as exc:
            error = f"Invalid move: {move}. {exc}"
            break
        except CarNotFound as exc:
            error = f"Car not found: {move.get('name')}. {exc}"
            break

    solved = puzzle_obj.solved()
    return SimulationResult(valid_moves=valid_moves, solved=solved, error=error)


def score_cot(sim_result: SimulationResult, parse_errors: List[str]) -> ScoreResult:
    """
    Assign a heuristic label/score to the CoT attempt based on simulation.

    Args:
        sim_result: Outcome from simulate_moves.
        parse_errors: Errors from parse_cot_to_moves.

    Returns:
        ScoreResult with label ('good'|'okay'|'bad') and score (1.0|0.5|0.0).
    """

    if sim_result.solved:
        return ScoreResult(label="good", score=1.0)

    if sim_result.error is not None:
        return ScoreResult(label="bad", score=0.0)

    if sim_result.valid_moves == 0 or (parse_errors and not sim_result.valid_moves):
        return ScoreResult(label="bad", score=0.0)

    return ScoreResult(label="okay", score=0.5)