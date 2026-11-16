import re
from pathlib import Path

import pandas as pd

MIN_STORY_WORDS = 200
MAX_STORY_WORDS = 800
SAMPLE_SIZE = 100
RANDOM_STATE = 42

project_root = Path(__file__).resolve().parents[2]

input_path = project_root / "data" / "raw" / "writingprompts" / "train-00000-of-00002.parquet"
output_path = project_root / "data" / "processed" / "writingprompts_sample.csv"


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


def looks_like_poem(story):
    lines = [line for line in story.split("\n") if line.strip()]
    if len(lines) <= 10:
        return False
    average_words = sum(len(line.split()) for line in lines) / len(lines)
    return average_words < 9


df = pd.read_parquet(input_path)

print("Original dataset rows:", len(df))

df = df[
    df["prompt"].str.match(r"^\s*\[\s*WP\s*\]", case=False, na=False)
].copy()

print("WP rows:", len(df))

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

print("Rows with", MIN_STORY_WORDS, "-", MAX_STORY_WORDS, "story words:", len(df))

df = df[~df["prompt"].str.match(r"^\s*\[")].copy()

df = df[~df["story"].apply(looks_like_poem)].copy()

df["prompt_key"] = (
    df["prompt"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

unique_prompts = (
    df.groupby("prompt_key", group_keys=False)
    .sample(n=1, random_state=RANDOM_STATE)
)

print("Unique prompts available:", len(unique_prompts))

sample = unique_prompts.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()

sample["prompt"] = sample["prompt"].apply(clean_text)
sample["story"] = sample["story"].apply(clean_text)

sample = sample[["prompt", "story"]].reset_index(drop=True)
sample.insert(0, "prompt_id", range(1, len(sample) + 1))

output_path.parent.mkdir(parents=True, exist_ok=True)

sample.to_csv(output_path, index=False, encoding="utf-8-sig")

word_counts = sample["story"].str.split().str.len()

print("\n--- DATASET PREPARATION SUMMARY ---")
print("Selected prompts:", len(sample))
print("Unique prompts:", sample["prompt"].nunique())
print("Missing values:")
print(sample.isnull().sum())
print("Story words - min/median/mean/max:",
      word_counts.min(), word_counts.median(), round(word_counts.mean()), word_counts.max())
print("\nSaved to:", output_path)