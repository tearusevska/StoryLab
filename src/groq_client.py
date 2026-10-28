import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GPT_OSS_MODEL = "openai/gpt-oss-20b"

def generate_text(prompt, temperature=0.7):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GPT OSS 20B API key is missing.")

    client = Groq(
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )

    try:
        response = client.chat.completions.create(
            model=GPT_OSS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_completion_tokens=3000,
            reasoning_effort="low",
            include_reasoning=False,
        )

        message = response.choices[0].message
        story = message.content

        if not story or not story.strip():
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(
                "GPT OSS 20B returned no story text. "
                f"Finish reason: {finish_reason}."
            )

        return story.strip()

    except Exception as error:
        raise RuntimeError(
            f"GPT OSS 20B generation failed: {error}"
        ) from error
