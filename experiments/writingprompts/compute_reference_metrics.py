from pathlib import Path

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

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

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rows = []
for index, row in stories.iterrows():
    story = str(row["story"])
    reference = reference_by_id[row["prompt_id"]]

    if len(story.split()) < MIN_VALID_WORDS:
        bleu = None
        rouge_l = None
    else:
        bleu = round(sacrebleu.sentence_bleu(story, [reference]).score, 3)
        rouge_l = round(scorer.score(reference, story)["rougeL"].fmeasure, 4)

    rows.append({
        "prompt_id": row["prompt_id"],
        "model": row["model"],
        "strategy": row["strategy"],
        "bleu": bleu,
        "rouge_l": rouge_l,
    })

    if (index + 1) % 100 == 0:
        print(f"{index + 1}/{len(stories)} done")

results = pd.DataFrame(rows)

output_path = project_root / "data" / "results" / "metrics_reference.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False, encoding="utf-8")

print()
print(results.groupby(["model", "strategy"])[["bleu", "rouge_l"]].mean().round(3))
print("\nMissing (excluded short stories):", int(results["bleu"].isna().sum()))
print("Saved to:", output_path)