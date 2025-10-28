import pandas as pd

def count_walls(board):
    count = 0
    for cell in board:
        if cell == 'x':
            count += 1
    return count

with open('dataset/rush.txt', 'r') as file:
    lines = file.readlines()

df = pd.DataFrame(columns=['0 wall', '1 wall', '2 wall'])

for line in lines:
    parts = line.split()
    diff = int(parts[0])
    board = parts[1]

    if diff not in df.index:
        df.loc[diff] = [0, 0, 0]
    
    walls_count = count_walls(board)

    if walls_count == 0:
        df.at[diff, '0 wall'] += 1
    elif walls_count == 1:
        df.at[diff, '1 wall'] += 1
    elif walls_count == 2:
        df.at[diff, '2 wall'] += 1

df.to_csv("dataset/full_dataset_summary.txt", sep="\t", index=True)
