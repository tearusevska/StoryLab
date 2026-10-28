import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"

def generate_text(prompt, temperature=0.7):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the .env file.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120000,
            retry_options=types.HttpRetryOptions(attempts=2),
        ),
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=800,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()

    except Exception as error:
        raise RuntimeError(
            f"Gemini generation failed: {error}"
        ) from error
