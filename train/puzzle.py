from copy import deepcopy
from dataclasses import dataclass

@dataclass
class RushHourSample:
    id: int
    board: list[list[str]]
    exit: str | tuple[int, int]
    min_num_moves: int
    solution_moves: list[dict[str, str | int]]

    @classmethod
    def from_dict(cls, d: dict) -> "RushHourSample":
        return cls(
            id=d["id"],
            board=d["board"],
            exit=d["exit"],
            min_num_moves=d["min_num_moves"],
            solution_moves=d["solution_moves"],
        )

class RushHourException(Exception):
    """Base exception class for Rush Hour errors."""
    pass

class InvalidMove(RushHourException):
    """Raised when an invalid move is attempted."""
    pass

class CarNotFound(RushHourException):
    """Raised when a specified car is not found on the board."""
    pass

class RushHourPuzzle:
    """
    Implementation of the Rush Hour Puzzle.
    - Board size N x N (default 6)
    
    Here is a snapshot of what a grid should look like (conceptually):

    board = [
        [   'A',   'A',    None,   'B',    None,   None    ],
        [   None,  None,   None,   'B',    None,   'F'     ],
        [   None,  'X',    'X',    None,   None,   'F'     ],
        [   'C',   None,   None,   None,   None,   None    ],
        [   'C',   None,   'D',    'D',    None,   None    ],
        [   None,  None,   None,   None,   'E',    'E'     ],
    ]

    There is no justified spacing in the actual snapshots, it's just for my personal OCD.
    """

    def __init__(self, id, exit, min_num_moves, board):
        self.id =id # puzzle id
        self.exit = exit # [row, col]
        self.min_num_moves = min_num_moves # minimum moves to solve
        self.board = board # list of lists representing the board
        self.size = len(self.board)  # board size N x N
       
        # unused attributes for compatibility
        self.N = self.size
        self.vehicles = {}  # vehicle_id: (row, col, length, direction)
    
    def find_car_position(self, car):
        car_positions = []
        orientation = None
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == car: #found the car
                    car_positions.append((r, c))

                    if c + 1 < self.size and self.board[r][c+1] == car: # horizontal car
                        orientation = 'H'
                        car_positions.append((r, c+1))
                        if c+2 < self.size and self.board[r][c+2] == car:
                            car_positions.append((r, c+2))

                    if  r + 1 < self.size and self.board[r+1][c] == car: # vertical car
                        orientation = 'V'
                        car_positions.append((r+1, c))
                        if r + 2 < self.size and self.board[r+2][c] == car:
                            car_positions.append((r+2, c))

                    return car_positions, orientation
        return None

    def move(self, car, direction, distance):
        car_data = self.find_car_position(car)
        if car_data is None:
            raise CarNotFound(f"Car {car} not found on the board.")
        car_positions, orientation = car_data
        
        if direction == 'up':
            if orientation != 'V':
                raise InvalidMove(f"Car {car} cannot move up; it is not vertical.")
            r, c = car_positions[0]
            if r - distance < 0:
                raise InvalidMove(f"Car {car} cannot move up by {distance}; out of bounds.")
            for step in range(1, distance + 1):
                if self.board[r - step][c] is not None:
                    raise InvalidMove(f"Car {car} cannot move up by {distance}; path blocked.")
            # Move the car
            for (r, c) in car_positions:
                self.board[r][c] = None
            for (r, c) in car_positions:
                self.board[r - distance][c] = car

        elif direction == 'down':
            if orientation != 'V':
                raise InvalidMove(f"Car {car} cannot move down; it is not vertical.")
            r, c = car_positions[-1]
            if r + distance >= self.size:
                raise InvalidMove(f"Car {car} cannot move down by {distance}; out of bounds.")
            for step in range(1, distance + 1):
                if self.board[r + step][c] is not None:
                    raise InvalidMove(f"Car {car} cannot move down by {distance}; path blocked.")
            # Move the car
            for (r, c) in car_positions:
                self.board[r][c] = None
            for (r, c) in car_positions:
                self.board[r + distance][c] = car

        elif direction == 'left':
            if orientation != 'H':
                raise InvalidMove(f"Car {car} cannot move left; it is not horizontal.")
            r, c = car_positions[0]
            if c - distance < 0:
                raise InvalidMove(f"Car {car} cannot move left by {distance}; out of bounds.")
            for step in range(1, distance + 1):
                if self.board[r][c - step] is not None:
                    raise InvalidMove(f"Car {car} cannot move left by {distance}; path blocked.")
            # Move the car
            for (r, c) in car_positions:
                self.board[r][c] = None
            for (r, c) in car_positions:
                self.board[r][c - distance] = car

        elif direction == 'right':
            if orientation != 'H':
                raise InvalidMove(f"Car {car} cannot move right; it is not horizontal.")
            r, c = car_positions[-1]
            if c + distance >= self.size:
                raise InvalidMove(f"Car {car} cannot move right by {distance}; out of bounds.")
            for step in range(1, distance + 1):
                if self.board[r][c + step] is not None:
                    raise InvalidMove(f"Car {car} cannot move right by {distance}; path blocked.")
            # Move the car
            for (r, c) in car_positions:
                self.board[r][c] = None
            for (r, c) in car_positions:
                self.board[r][c + distance] = car

                
        else:
            raise InvalidMove(f"Invalid direction {direction} for car {car}.")
    
    def solved(self):
        exit_row = self.exit[0]
        exit_col = self.exit[1]
        return self.board[exit_row-1][exit_col-1] == 'R'

def validate_solution(puzzle, moves, check_optimal=False):
    sim = RushHourPuzzle(
        id=puzzle.id,
        exit=puzzle.exit,
        min_num_moves=puzzle.min_num_moves,
        board=deepcopy(puzzle.board),
    )

    for i, m in enumerate(moves, 1):
        if not isinstance(m, dict):
            print(f"move {i} is not a dict: {m}")
            return False, "TYPE_ERROR"
        if not all(k in m for k in ("name", "direction", "distance")):
            print(f"move {i} is missing keys: {m}")
            return False, "MISSING_KEYS"
        try:
            sim.move(m["name"], m["direction"], int(m["distance"]))
        except CarNotFound as e:
            print(f"move {i} car not found: {e}")
            return False, "CAR_NOT_FOUND"
        except InvalidMove as e:
            print(f"move {i} invalid: {e}")
            return False, "INVALID_MOVE"
        except Exception as e:
            print(f"move {i} unknown error: {e}")
            return False, "UNKNOWN_ERROR"

    if not sim.solved():
        return False, "UNSOLVED"

    if check_optimal and len(moves) != puzzle.min_num_moves:
        return True, "NOT_OPTIMAL"

    return True, "OPTIMAL"