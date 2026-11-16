import re
from pathlib import Path

import pandas as pd

MIN_STORY_WORDS = 350
MAX_STORY_WORDS = 450
NUM_EXAMPLES = 3
RANDOM_STATE = 42

project_root = Path(__file__).resolve().parents[2]

input_path = project_root / "data" / "raw" / "writingprompts" / "train-00000-of-00002.parquet"
test_path = project_root / "data" / "processed" / "writingprompts_sample.csv"
output_path = project_root / "data" / "processed" / "fewshot_examples.csv"


def clean_text(text):
    text = str(text)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\w)[ \t]+([’'])[ \t]*(s|t|re|ve|ll|d|m)\b", r"\1\2\3", text)
    text = re.sub(r"(\w)[ \t]+(n[’']t)\b", r"\1\2", text)
    text = re.sub(r"[ \t]+([.,!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[])[ \t]+", r"\1", text)
    text = re.sub(r"([“‘])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]+([”])", r"\1", text)
    text = re.sub(r'"[ \t]*([^"\n]*?)[ \t]*"', r'"\1"', text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_key(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def looks_like_poem(story):
    lines = [line for line in story.split("\n") if line.strip()]
    if len(lines) <= 10:
        return False
    average_words = sum(len(line.split()) for line in lines) / len(lines)
    return average_words < 9


test_df = pd.read_csv(test_path)
test_keys = set(test_df["prompt"].apply(normalize_key))

df = pd.read_parquet(input_path)

df = df[
    df["prompt"].str.match(r"^\s*\[\s*WP\s*\]", case=False, na=False)
].copy()

df["prompt"] = df["prompt"].fillna("").astype(str).str.strip()
df["story"] = df["story"].fillna("").astype(str).str.strip()

df = df[
    (df["prompt"] != "") &
    (df["story"] != "")
].copy()

df["prompt"] = df["prompt"].str.replace(
    r"^\s*\[\s*WP\s*\]\s*",
    "",
    regex=True,
    case=False
).str.strip()

df["story_words"] = df["story"].str.split().str.len()

df = df[
    (df["story_words"] >= MIN_STORY_WORDS) &
    (df["story_words"] <= MAX_STORY_WORDS)
].copy()

df = df[~df["prompt"].str.match(r"^\s*\[")].copy()
df = df[~df["story"].apply(looks_like_poem)].copy()

df["prompt"] = df["prompt"].apply(clean_text)
df["prompt_key"] = df["prompt"].apply(normalize_key)

df = df[~df["prompt_key"].isin(test_keys)].copy()

unique_prompts = (
    df.groupby("prompt_key", group_keys=False)
    .sample(n=1, random_state=RANDOM_STATE)
)

examples = unique_prompts.sample(n=NUM_EXAMPLES, random_state=RANDOM_STATE).copy()
examples["story"] = examples["story"].apply(clean_text)

examples = examples[["prompt", "story"]].reset_index(drop=True)
examples.insert(0, "example_id", range(1, len(examples) + 1))

examples.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Candidates after filters (excluding test prompts):", len(unique_prompts))
print("Overlap with test prompts:", examples["prompt"].apply(normalize_key).isin(test_keys).sum())
print()

for _, row in examples.iterrows():
    print(f"--- Example {row['example_id']} ({len(row['story'].split())} words) ---")
    print("PROMPT:", row["prompt"])
    print(row["story"][:500], "...\n")

print("Saved to:", output_path)