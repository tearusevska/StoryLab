import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.prompting import build_writing_prompt
from src.gemini_client import generate_text as generate_gemini
from src.groq_client import generate_text as generate_gpt_oss

TEMPERATURE = 0.7
MAX_TOKENS = 3000
TARGET_WORDS = 400
DELAY_SECONDS = 2
MAX_CONSECUTIVE_ERRORS = 5

STRATEGIES = ["Zero-shot", "Few-shot", "Structured"]

MODELS = {
    "Gemini 3.5 Flash-Lite": generate_gemini,
    "GPT OSS 20B": generate_gpt_oss,
}

COLUMNS = ["prompt_id", "model", "strategy", "story", "words", "seconds"]

project_root = Path(__file__).resolve().parents[2]

prompts_path = project_root / "data" / "processed" / "writingprompts_sample.csv"
examples_path = project_root / "data" / "processed" / "fewshot_examples.csv"
output_path = project_root / "data" / "generated" / "stories.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

prompts_df = pd.read_csv(prompts_path)
examples = pd.read_csv(examples_path).to_dict("records")

done = set()
if output_path.exists():
    existing = pd.read_csv(output_path)
    done = set(zip(existing["prompt_id"], existing["model"], existing["strategy"]))
else:
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(COLUMNS)

errors_in_a_row = {name: 0 for name in MODELS}
total = len(prompts_df) * len(STRATEGIES) * len(MODELS)
counter = 0

for _, row in prompts_df.iterrows():
    for strategy in STRATEGIES:
        prompt = build_writing_prompt(
            row["prompt"],
            strategy,
            examples=examples,
            target_words=TARGET_WORDS,
        )

        for model_name, generate in MODELS.items():
            counter += 1
            key = (row["prompt_id"], model_name, strategy)

            if key in done:
                continue

            if errors_in_a_row[model_name] >= MAX_CONSECUTIVE_ERRORS:
                continue

            start = time.time()
            try:
                story = generate(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            except Exception as error:
                errors_in_a_row[model_name] += 1
                print(f"[{counter}/{total}] FAILED {model_name} | {strategy} | prompt {row['prompt_id']}: {error}")
                if errors_in_a_row[model_name] >= MAX_CONSECUTIVE_ERRORS:
                    print(f"{model_name} skipped for the rest of this run after {MAX_CONSECUTIVE_ERRORS} errors in a row.")
                time.sleep(DELAY_SECONDS)
                continue

            errors_in_a_row[model_name] = 0
            seconds = round(time.time() - start, 1)

            with open(output_path, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    row["prompt_id"],
                    model_name,
                    strategy,
                    story,
                    len(story.split()),
                    seconds,
                ])

            print(f"[{counter}/{total}] OK {model_name} | {strategy} | prompt {row['prompt_id']} | {len(story.split())} words | {seconds}s")
            time.sleep(DELAY_SECONDS)

print("Saved to:", output_path)