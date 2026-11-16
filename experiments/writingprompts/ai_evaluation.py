import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

JUDGE_MODEL = "qwen/qwen3.8-27b"
MIN_VALID_WORDS = 100
DELAY_SECONDS = 1
MAX_CONSECUTIVE_ERRORS = 5
RANDOM_STATE = 42

CRITERIA = ["coherence", "creativity", "relevance", "consistency", "engagement", "language_quality"]
COLUMNS = ["prompt_id", "model", "strategy"] + CRITERIA + ["reason", "tokens"]

RUBRIC = """You are an impartial literary evaluator. You will see a writing prompt and a short story written for it. Rate the story on six criteria, each from 1 to 10.

Criteria:
- coherence: events are logically connected, with a clear beginning, development and ending.
- creativity: originality of ideas and imaginative treatment of the prompt.
- relevance: how well the story follows and fits the writing prompt.
- consistency: characters, setting, tone and facts stay consistent, without contradictions.
- engagement: the story holds the reader's interest and creates emotional pull.
- language_quality: grammar, fluency, vocabulary and style.

Scale: 1-2 very poor, 3-4 weak, 5-6 adequate, 7-8 good, 9-10 excellent. Use the full scale and reserve 9-10 for truly exceptional work. Do not reward or penalize the length of the story.

Reply only with JSON in exactly this format:
{"coherence": <1-10>, "creativity": <1-10>, "relevance": <1-10>, "consistency": <1-10>, "engagement": <1-10>, "language_quality": <1-10>, "reason": "<one short sentence summarizing your judgment>"}"""

project_root = Path(__file__).resolve().parents[2]

story_files = [
    project_root / "data" / "generated" / "stories.csv",
    project_root / "data" / "generated" / "stories_llama.csv",
]

prompts = pd.read_csv(project_root / "data" / "processed" / "writingprompts_sample.csv")
prompt_by_id = dict(zip(prompts["prompt_id"], prompts["prompt"]))

frames = [pd.read_csv(path) for path in story_files if path.exists()]
stories = pd.concat(frames, ignore_index=True)
stories["story"] = stories["story"].astype(str)
stories = stories[stories["story"].str.split().str.len() >= MIN_VALID_WORDS]
stories = stories.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

output_path = project_root / "data" / "results" / "ai_scores.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

done = set()
if output_path.exists():
    existing = pd.read_csv(output_path)
    done = set(zip(existing["prompt_id"], existing["model"], existing["strategy"]))
else:
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(COLUMNS)

client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=120.0, max_retries=2)


def parse_scores(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in the reply.")

    data = json.loads(match.group(0))

    for criterion in CRITERIA:
        value = int(data[criterion])
        if not 1 <= value <= 10:
            raise ValueError(f"{criterion} out of range: {value}")
        data[criterion] = value

    data["reason"] = str(data.get("reason", "")).strip()
    return data


def judge(writing_prompt, story):
    user_message = f"Writing prompt:\n{writing_prompt}\n\nStory:\n{story}"
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_completion_tokens=800,
        reasoning_effort="none",
    )
    scores = parse_scores(response.choices[0].message.content)
    scores["tokens"] = response.usage.total_tokens
    return scores


total = len(stories)
errors_in_a_row = 0

for index, row in stories.iterrows():
    key = (row["prompt_id"], row["model"], row["strategy"])

    if key in done:
        continue

    scores = None
    last_error = None

    for attempt in range(2):
        try:
            scores = judge(prompt_by_id[row["prompt_id"]], row["story"])
            break
        except Exception as error:
            last_error = error
            time.sleep(DELAY_SECONDS)

    if scores is None:
        errors_in_a_row += 1
        print(f"[{index + 1}/{total}] FAILED prompt {row['prompt_id']} | {row['model']} | {row['strategy']}: {last_error}")
        if errors_in_a_row >= MAX_CONSECUTIVE_ERRORS:
            print(f"Stopped after {MAX_CONSECUTIVE_ERRORS} errors in a row. Run the script again later to continue.")
            break
        continue

    errors_in_a_row = 0

    with open(output_path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [row["prompt_id"], row["model"], row["strategy"]]
            + [scores[criterion] for criterion in CRITERIA]
            + [scores["reason"], scores["tokens"]]
        )

    print(f"[{index + 1}/{total}] OK prompt {row['prompt_id']} | {row['model']} | {row['strategy']}")
    time.sleep(DELAY_SECONDS)

print("Saved to:", output_path)