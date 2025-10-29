import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import *
from data.rh_puzzle import *

def evaluate_solution(puzzle, solution):

    puzzle_obj = RushHourPuzzle(
        id=puzzle["id"],
        exit=puzzle["exit"],
        min_moves=puzzle.get("minimum_number_moves", None),
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

if __name__ == "__main__":
    # Example usage
    
    puzzle = data_loader("./dataset/rush_no_wall_1000_balanced.json")[33]

    solution = [
        {"name": "B", "direction": "left", "distance": 1},
        {"name": "C", "direction": "down", "distance": 3},
        {"name": "R", "direction": "right", "distance": 4},
    ]

    moves, success, error = evaluate_solution(puzzle, solution)
    print(f"Moves made: {moves}, Success: {success}, Error: {error}")