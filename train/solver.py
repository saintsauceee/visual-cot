from __future__ import annotations
import copy
from collections import deque

from puzzle import RushHourPuzzle, InvalidMove, CarNotFound

def board_for_prompt(board: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in board)

def board_key(puzzle: RushHourPuzzle):
    return tuple(tuple(row) for row in puzzle.board)

def generate_moves(puzzle: RushHourPuzzle):
    moves = []
    size = puzzle.size
    seen = set()

    for r in range(size):
        for c in range(size):
            car = puzzle.board[r][c]
            if car is None or car in seen:
                continue
            seen.add(car)

            res = puzzle.find_car_position(car)
            if res is None:
                continue
            positions, orient = res

            if orient == "H":
                # move left
                row, left_c = positions[0]
                dist = 0
                cc = left_c - 1
                while cc >= 0 and puzzle.board[row][cc] is None:
                    dist += 1
                    moves.append({"name": car, "direction": "left", "distance": dist})
                    cc -= 1
                # move right
                row, right_c = positions[-1]
                dist = 0
                cc = right_c + 1
                while cc < size and puzzle.board[row][cc] is None:
                    dist += 1
                    moves.append({"name": car, "direction": "right", "distance": dist})
                    cc += 1

            elif orient == "V":
                # move up
                top_r, col = positions[0]
                dist = 0
                rr = top_r - 1
                while rr >= 0 and puzzle.board[rr][col] is None:
                    dist += 1
                    moves.append({"name": car, "direction": "up", "distance": dist})
                    rr -= 1
                # move down
                bottom_r, col = positions[-1]
                dist = 0
                rr = bottom_r + 1
                while rr < size and puzzle.board[rr][col] is None:
                    dist += 1
                    moves.append({"name": car, "direction": "down", "distance": dist})
                    rr += 1

    return moves

def solve_puzzle(puzzle: RushHourPuzzle):
    start = RushHourPuzzle(
        id=puzzle.id,
        exit=puzzle.exit,
        min_num_moves=puzzle.min_num_moves,
        board=copy.deepcopy(puzzle.board),
    )

    q = deque([(start, [])])
    visited = {board_key(start)}

    while q:
        state, path = q.popleft()

        if state.solved():
            return path

        for move in generate_moves(state):
            new_state = RushHourPuzzle(
                id=state.id,
                exit=state.exit,
                min_num_moves=state.min_num_moves,
                board=copy.deepcopy(state.board),
            )
            try:
                new_state.move(move["name"], move["direction"], move["distance"])
            except (InvalidMove, CarNotFound):
                continue

            key = board_key(new_state)
            if key in visited:
                continue
            visited.add(key)

            q.append((new_state, path + [move]))

    return None
