from pathlib import Path

import pandas as pd

TARGET_WORDS = 400
KEYS = ["prompt_id", "model", "strategy"]
CRITERIA = ["coherence", "creativity", "relevance", "consistency", "engagement", "language_quality"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

results_dir = Path(__file__).resolve().parents[2] / "data" / "results"

lexical = pd.read_csv(results_dir / "metrics_lexical.csv")
reference = pd.read_csv(results_dir / "metrics_reference.csv")
bertscore = pd.read_csv(results_dir / "metrics_bertscore.csv")
ai = pd.read_csv(results_dir / "ai_scores.csv")

for name, frame in [("lexical", lexical), ("reference", reference), ("bertscore", bertscore), ("ai", ai)]:
    duplicates = frame.duplicated(KEYS).sum()
    print(f"{name}: {len(frame)} rows, duplicates: {duplicates}")

data = lexical.merge(reference, on=KEYS, how="left")
data = data.merge(bertscore, on=KEYS, how="left")
data = data.merge(ai.drop(columns=["reason", "tokens"]), on=KEYS, how="left")

data["length_deviation"] = (data["word_count"] - TARGET_WORDS).abs()
data["ai_mean"] = data[CRITERIA].mean(axis=1)

output_path = results_dir / "all_results.csv"
data.to_csv(output_path, index=False, encoding="utf-8")

valid = data[data["valid"]]

print("\nTotal rows:", len(data), "| valid:", len(valid))
print("Valid stories without AI scores:", int(valid["ai_mean"].isna().sum()))

print("\nAI score distribution (number of stories per score):")
distribution = valid[CRITERIA].apply(lambda column: column.value_counts()).fillna(0).astype(int).sort_index()
print(distribution)

columns = CRITERIA + ["ai_mean", "bleu", "rouge_l", "bertscore_f1", "word_count", "length_deviation"]


def summarize(frame, by):
    grouped = frame.groupby(by)[columns]
    return grouped.mean().round(2).astype(str) + " (" + grouped.std().round(2).astype(str) + ")"


print("\nBy model - mean (std):")
print(summarize(valid, "model").T)

print("\nBy strategy - mean (std):")
print(summarize(valid, "strategy").T)

print("\nSaved to:", output_path)