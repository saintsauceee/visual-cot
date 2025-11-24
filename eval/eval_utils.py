import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from data.data_loader import *
from data.rh_puzzle import *

def eval_sol(puzzle, solution):

    puzzle_obj = RushHourPuzzle(
        id=puzzle["id"],
        exit=puzzle["exit"],
        min_moves=puzzle.get("min_moves", None),
        board=puzzle["board"]
    )

    completed_moves = 0 
    success = False
    error = None

    for move in solution:
        try:
            puzzle_obj.move(move["name"], move["direction"].lower(), move["distance"])
            completed_moves += 1
        except InvalidMove:
            error = f"Invalid move: {move}"
            break
        except CarNotFound:
            error = f"Car not found: {move['name']}"
            break
    
    if puzzle_obj.solved():
        success = True
    
    return completed_moves, success, error

def get_eval_ids():
    
    full_dataset = data_loader()

    grouped = {}

    # Group puzzles by minimal required number of moves
    for pid, puzzle in full_dataset.items():
        min_moves = puzzle["min_moves"]
        grouped.setdefault(min_moves, []).append(pid)

    # Create a set for testing samples and another for few shots prompting
    test_ids = {}
    fsp_ids = {}

    # Split dataset into test set and few (3) shots prompting set
    for min_moves, ids in grouped.items():
        if len(ids) >= 8:
            random_samples = random.sample(ids, 8)
            test_ids[min_moves] = random_samples[0:5]
            fsp_ids[min_moves] = random_samples[5:]
        else:
            test_ids[min_moves] = ids
            fsp_ids[min_moves] = []

    return test_ids, fsp_ids


if __name__ == "__main__":
    test_set, fsp_set = get_eval_ids()
    print(fsp_set)
    '''
    # Example usage
    puzzle = data_loader("./dataset/rush_no_wall_1000_balanced.json")[33]

    solution = [
        {"name": "B", "direction": "left", "distance": 1},
        {"name": "C", "direction": "down", "distance": 3},
        {"name": "R", "direction": "right", "distance": 4},
    ]

    moves, success, error = evaluate_solution(puzzle, solution)
    print(f"Moves made: {moves}, Success: {success}, Error: {error}")
    '''