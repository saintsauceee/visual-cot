import copy

def solve(puzzle):
    """
    Solve the Rush Hour puzzle using Iterative Deepening Search (IDS).

    Returns a list of moves:
        [{"name": "A", "direction": "UP", "move": 1}, ...]
    """
    board = puzzle["board"]
    exit_pos = (puzzle["exit"][0] - 1, puzzle["exit"][1] - 1)  # 0-indexed

    def is_goal(state):
        """Goal: red car 'X' reaches the exit."""
        r, c = exit_pos
        return state[r][c] == "X"

    def possible_moves(state):
        """
        Generate all next states from the current board.
        Each move includes car name, direction, and number of squares moved.
        """
        moves = []
        seen_cars = set()

        for r in range(6):
            for c in range(6):
                car = state[r][c]
                if not car or car == "none" or car in seen_cars:
                    continue
                seen_cars.add(car)

                # Collect all positions of this car
                positions = [(i, j) for i in range(6) for j in range(6) if state[i][j] == car]

                rows = {i for i, _ in positions}
                cols = {j for _, j in positions}
                horiz = len(rows) == 1

                if horiz:
                    row = next(iter(rows))
                    min_c = min(j for _, j in positions)
                    max_c = max(j for _, j in positions)

                    # move LEFT
                    steps = 1
                    while min_c - steps >= 0 and state[row][min_c - steps] is None:
                        new_state = copy.deepcopy(state)
                        # clear old positions
                        for _, j in positions:
                            new_state[row][j] = None
                        # new positions
                        for _, j in positions:
                            new_state[row][j - steps] = car
                        moves.append((car, "LEFT", steps, new_state))
                        steps += 1

                    # move RIGHT
                    steps = 1
                    while max_c + steps < 6 and state[row][max_c + steps] is None:
                        new_state = copy.deepcopy(state)
                        for _, j in positions:
                            new_state[row][j] = None
                        for _, j in positions:
                            new_state[row][j + steps] = car
                        moves.append((car, "RIGHT", steps, new_state))
                        steps += 1

                else:  # vertical car
                    col = next(iter(cols))
                    min_r = min(i for i, _ in positions)
                    max_r = max(i for i, _ in positions)

                    # move UP
                    steps = 1
                    while min_r - steps >= 0 and state[min_r - steps][col] is None:
                        new_state = copy.deepcopy(state)
                        for i, _ in positions:
                            new_state[i][col] = None
                        for i, _ in positions:
                            new_state[i - steps][col] = car
                        moves.append((car, "UP", steps, new_state))
                        steps += 1

                    # move DOWN
                    steps = 1
                    while max_r + steps < 6 and state[max_r + steps][col] is None:
                        new_state = copy.deepcopy(state)
                        for i, _ in positions:
                            new_state[i][col] = None
                        for i, _ in positions:
                            new_state[i + steps][col] = car
                        moves.append((car, "DOWN", steps, new_state))
                        steps += 1

        return moves

    def dls(state, depth, visited):
        """Depth-limited DFS."""
        if is_goal(state):
            return []

        if depth == 0:
            return None

        visited.add(tuple(tuple(row) for row in state))

        for car, direction, move, next_state in possible_moves(state):
            state_key = tuple(tuple(row) for row in next_state)
            if state_key not in visited:
                result = dls(next_state, depth - 1, visited)
                if result is not None:
                    return [{"name": car, "direction": direction, "move": move}] + result

        return None

    # Iterative deepening
    for limit in range(1, 50):
        visited = set()
        result = dls(board, limit, visited)
        if result is not None:
            return {
                "minimum_number_moves": len(result),
                "moves": result
            }

    return None

if __name__ == "__main__":
    puzzle = {
        "name": 1,
        "exit": [6, 4],
        "board": [
            ["B", "B", "P" , "P", "P", "A"],
            [None, None , None, "X", None, "A"],
            ["R", None, None, "X", None, None],
            ["R", None, "Q", "Q", "Q", None],
            ["R", "C", None, None, None, None],
            [None, "C", None, "O", "O", "O"]
        ]
    }

    solution = solve(puzzle)
    print(solution)