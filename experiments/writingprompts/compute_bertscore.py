from pathlib import Path

import pandas as pd
from bert_score import score

MIN_VALID_WORDS = 100

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parents[2]

story_files = [
    project_root / "data" / "generated" / "stories.csv",
    project_root / "data" / "generated" / "stories_llama.csv",
]

references = pd.read_csv(project_root / "data" / "processed" / "writingprompts_sample.csv")
reference_by_id = dict(zip(references["prompt_id"], references["story"]))

frames = [pd.read_csv(path) for path in story_files if path.exists()]
stories = pd.concat(frames, ignore_index=True)

stories["story"] = stories["story"].astype(str)
stories["is_valid"] = stories["story"].str.split().str.len() >= MIN_VALID_WORDS

valid = stories[stories["is_valid"]].copy()
candidates = valid["story"].tolist()
reference_texts = [reference_by_id[prompt_id] for prompt_id in valid["prompt_id"]]

print("Scoring", len(candidates), "stories. The first run downloads the model (about 1.4 GB).")

precision, recall, f1 = score(
    candidates,
    reference_texts,
    lang="en",
    batch_size=16,
    rescale_with_baseline=True,
    verbose=True,
)

valid["bertscore_f1"] = [round(value, 4) for value in f1.tolist()]

results = stories[["prompt_id", "model", "strategy"]].merge(
    valid[["prompt_id", "model", "strategy", "bertscore_f1"]],
    on=["prompt_id", "model", "strategy"],
    how="left",
)

output_path = project_root / "data" / "results" / "metrics_bertscore.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False, encoding="utf-8")

print()
print(results.groupby(["model", "strategy"])["bertscore_f1"].mean().round(4))
print("\nMissing (excluded short stories):", int(results["bertscore_f1"].isna().sum()))
print("Saved to:", output_path)