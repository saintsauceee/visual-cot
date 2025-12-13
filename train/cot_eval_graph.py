import json
import ast
import matplotlib.pyplot as plt
from collections import defaultdict

with open("./cot_evaluation_results_20251213_073258.json", "r", encoding="utf-8") as f:
    data = json.load(f)

scores = defaultdict(lambda: defaultdict(list))

for item in data:
    level = item["level"]
    s = ast.literal_eval(item["raw_scores"])
    for k, v in s.items():
        scores[level][k].append(v)

levels = sorted(scores.keys())
criteria = ["Factuality", "Validity", "Coherence", "Utility"]

avg = {
    c: [
        sum(scores[lvl][c]) / len(scores[lvl][c])
        for lvl in levels
    ]
    for c in criteria
}

plt.figure()
for c in criteria:
    plt.plot(levels, avg[c], marker="o", label=c)

plt.xlabel("Level")
plt.ylabel("Average score")
plt.title("Average evaluation scores per level")
plt.legend()
plt.tight_layout()
plt.savefig("avg_scores_per_level_by_criterion.png")
plt.show()
