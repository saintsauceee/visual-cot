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