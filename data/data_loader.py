import os
import json 

# Loads all the puzzles in the given folder into a dictionnary

def data_loader(folder_path):
    puzzles = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            # Extract puzzle number (e.g. "3.json" -> 3)
            puzzle_number = int(filename.split(".")[0])

            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                puzzles[puzzle_number] = json.load(file)

    return puzzles


if __name__ == "__main__":

    puzzles = data_loader("./dataset")
    print(puzzles)
    