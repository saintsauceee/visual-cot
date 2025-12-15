import json

def data_loader(file_path = './dataset/rush_no_wall_1000_balanced.json'):
    
    with open(file_path, "r", encoding="utf-8") as file:
        puzzles_list = json.load(file)

    # Convert list of puzzles to dictionary {name: board}
    puzzles = {
        puzzle["name"]: {
            "id": puzzle["name"],
            "exit": puzzle.get("exit"),
            "min_moves": puzzle.get("min_num_moves"),
            "board": puzzle["board"]
        }
        for puzzle in puzzles_list
    }

    return puzzles


if __name__ == "__main__":
    puzzles = data_loader()
    print(puzzles[1])  # Example to print the first puzzle