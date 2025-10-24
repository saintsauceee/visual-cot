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

    def __init__(self, N: int = 6):
        self.N = N
        self.vehicles = {}
        self.exit = ('right', 2)
    
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

if __name__ == "__main__":
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