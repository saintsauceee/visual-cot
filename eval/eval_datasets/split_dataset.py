import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.data_loader import *
import random 

random.seed(42)
TEST_SIZE = 10
FSP_SIZE = 3

full_dataset = data_loader()

grouped_dataset = {}

for pid in full_dataset:
    puzzle = full_dataset[pid]
    level = puzzle['min_moves']
    grouped_dataset.setdefault(level, []).append(pid)

# Create empty test ids dict, fsp ids dict and trainind ids dict
ids_test = {}
ids_fsp = {}
ids_train = {}

for i in range(3, 21):
    random.shuffle(grouped_dataset[i])
    ids_test[i] = grouped_dataset[i][:TEST_SIZE]
    ids_fsp[i] = grouped_dataset[i][TEST_SIZE: TEST_SIZE+FSP_SIZE]
    ids_train[i] = grouped_dataset[i][TEST_SIZE+FSP_SIZE+1:]

print(ids_test[3])
print(ids_fsp[3])
print(ids_train[3])

print(ids_test)

IDS_TEST_PATH = 'eval/eval_datasets/ids_test.json'
IDS_FSP_PATH = 'eval/eval_datasets/ids_fsp.json'
IDS_TRAIN_PATH = 'eval/eval_datasets/ids_training.json'

with open(IDS_TRAIN_PATH, "w") as f:
        json.dump(ids_train, f, indent=4)

with open(IDS_TEST_PATH, "w") as f:
        json.dump(ids_test, f, indent=4)

with open(IDS_FSP_PATH, "w") as f:
        json.dump(ids_fsp, f, indent=4)