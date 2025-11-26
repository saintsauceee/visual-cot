import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.data_loader import *


full_dataset = data_loader()
train_ids = {}

# Group puzzles by minimal required number of moves
for pid, puzzle in full_dataset.items():
    min_moves = puzzle["min_moves"]
    train_ids.setdefault(min_moves, []).append(pid)


with open("eval/eval_datasets/ids_testing.json", "r") as f:
        raw = json.load(f)
        test_ids = {int(k): v for k, v in raw.items()}

with open("eval/eval_datasets/ids_fs_prompting.json", "r") as f:
    raw = json.load(f)
    fsp_ids = {int(k): v for k, v in raw.items()}

for level in train_ids:
    for puzzle in train_ids[level][:]:
        if puzzle in fsp_ids[level] or puzzle in test_ids[level]:
            train_ids[level].remove(puzzle)

with open("eval/eval_datasets/ids_training.json", "w") as f:
        json.dump(train_ids, f, indent=4)
