from data.rh_exceptions import *
from data.data_loader import *

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

    def __init__(self, id, exit, min_moves, board):
        self.id =id # puzzle id
        self.exit = exit # [row, col]
        self.min_moves = min_moves # minimum moves to solve
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
    
    '''
    def set_vehicles(self, vehicles):
        self.vehicles = dict(vehicles)
        self._validate()

    def grid(self):
        N = self.N
        g = [ [None] * N for _ in range(N) ]

        # (row, col, len, dir)
        for vid, (r, c, L, d) in self.vehicles.items():
            for k in range(L):
                rr = r + (k if d == 'V' else 0)     # vertical
                cc = c + (k if d == 'H' else 0)     # horizontal

                if not (0 <= rr < N and 0 <= cc < N):
                    raise ValueError("Vehicle out of bounds: %s" % vid)
                
                if g[rr][cc] is not None:
                    raise ValueError("Overlap at (%d,%d) between %s %s" % (rr, cc, g[rr][cc], vid))
                
                g[rr][cc] = vid
        
        return g
    
    def snapshot(self):
        """ ASCII snapshot to visualize the board. """

        g = self.grid()
        rows = []
        for r in range(self.N):
            row = []
            for c in range(self.N):
                row.append((g[r][c] or '.'))
            rows.append(' '.join(row))
        return '\n'.join(rows)

    def _validate(self):
        for vid, (r, c, L, d) in self.vehicles.items():
            if d not in ('H', 'V'):
                raise ValueError("direction must be 'H' or 'V' for %s" % vid)
            if L not in (2, 3):
                raise ValueError("vehicle length must be 2 or 3 for %s" % vid)
            if not (0 <= r < self.N and 0 <= c < self.N):
                raise ValueError("anchor out of bounds for %s" % vid)
            
            # ensure full length fits on board from its anchor
            rr = r + (L - 1 if d == 'V' else 0)
            cc = c + (L - 1 if d == 'H' else 0)
            if not (0 <= rr < self.N and 0 <= cc < self.N):
                raise ValueError("vehicle does not fit: %s" % vid)
        
        # enforce red-car convention: horizontal only
        if 'X' in self.vehicles and self.vehicles['X'][3] != 'H':
            raise ValueError("red car 'X' must be horizontal")
    '''

if __name__ == "__main__":

    '''
    import data_loader as dl

    board = RushHourPuzzle()
    board.set_vehicles({
        'X': (2, 1, 2, 'H'),
        'A': (0, 0, 2, 'H'),
        'B': (0, 3, 3, 'V'),
        'C': (3, 0, 3, 'V'),
        'D': (4, 2, 2, 'H'),
        'E': (5, 4, 2, 'H'),
        'F': (1, 5, 2, 'V'),
    })
    print(board.snapshot())

    puzzles = dl.data_loader("./dataset/rush_hour_puzzles.json")
    print(puzzles[1])
    '''
    test_puzzle = data_loader("./dataset/rush_no_wall_1000_balanced.json")[1]
    puzzle1 = RushHourPuzzle(id= test_puzzle['id'], exit= test_puzzle['exit'], min_moves= test_puzzle['min_moves'], board= test_puzzle['board'])
    puzzle1.move('B', 'left', 2)
    print(puzzle1.board)
    pass
