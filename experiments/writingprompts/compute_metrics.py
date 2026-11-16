import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate_story_metrics

MIN_VALID_WORDS = 100

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parents[2]

story_files = [
    project_root / "data" / "generated" / "stories.csv",
    project_root / "data" / "generated" / "stories_llama.csv",
]

frames = [pd.read_csv(path) for path in story_files if path.exists()]
stories = pd.concat(frames, ignore_index=True)

rows = []
for _, row in stories.iterrows():
    metrics = evaluate_story_metrics(row["story"])
    rows.append({
        "prompt_id": row["prompt_id"],
        "model": row["model"],
        "strategy": row["strategy"],
        **metrics,
        "valid": metrics["word_count"] >= MIN_VALID_WORDS,
    })

results = pd.DataFrame(rows)

output_path = project_root / "data" / "results" / "metrics_lexical.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False, encoding="utf-8")

valid = results[results["valid"]]

print("Stories:", len(results), "| valid:", len(valid), "| excluded:", len(results) - len(valid))
print()
print(
    valid.groupby(["model", "strategy"])[
        ["word_count", "mattr", "distinct_1", "distinct_2", "repeated_trigram_rate"]
    ].mean().round(3)
)
print("\nSaved to:", output_path)