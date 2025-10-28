import json

def data_loader(file_path):
    """
    Load all puzzles from a single JSON file into a dictionary.

    Args:
        file_path (str): Path to the JSON file containing all puzzles.

    Returns:
        dict: {puzzle_name (int): board (list of lists)}
    """
    with open(file_path, "r", encoding="utf-8") as file:
        puzzles_list = json.load(file)

    # Convert list of puzzles to dictionary {name: board}
    puzzles = {
        puzzle["name"]: {
            "id": puzzle["name"],
            "exit": puzzle.get("exit"),
            "min_moves": puzzle.get("min_moves"),
            "board": puzzle["board"]
        }
        for puzzle in puzzles_list
    }

    return puzzles


if __name__ == "__main__":
    puzzles = data_loader("./dataset/rush_no_wall_1000_balanced.json")
    print(puzzles[1])  # Example to print the first puzzle