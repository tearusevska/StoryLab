import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv

from src.prompting import (
    build_story_prompt,
    build_opening_prompt,
    build_choices_prompt,
    build_ending_prompt,
)
from src.evaluation import evaluate_text
from src.gemini_client import generate_text as generate_gemini
from src.groq_client import generate_text as generate_gpt_oss

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "storylab-local-development-key",
)

MODELS = {
    "Gemini 3.5 Flash-Lite": generate_gemini,
    "GPT OSS 20B": generate_gpt_oss,
}

DEFAULTS = {
    "language": "English",
    "genre": "Fantasy",
    "topic": "Friendship",
    "style": "Descriptive",
    "characters": "A student, a magician, and a mysterious book",
    "strategy": "Zero-shot",
    "model": "Gemini 3.5 Flash-Lite",
    "word_limit": "150",
    "temperature": "0.7",
}

USER_DATA = {}


def get_data():
    if "storylab_id" not in session:
        session["storylab_id"] = secrets.token_hex(16)

    user_id = session["storylab_id"]

    if user_id not in USER_DATA:
        USER_DATA[user_id] = {
            "history": [],
            "last_result": None,
            "last_prompt": "",
            "last_comparison": [],
            "last_comparison_prompt": "",
            "interactive": {},
        }

    return USER_DATA[user_id]


def get_settings():
    settings = DEFAULTS.copy()
    settings.update(session.get("settings", {}))
    return settings


def read_settings():
    previous_settings = get_settings()
    settings = {}

    for key, default in DEFAULTS.items():
        settings[key] = request.form.get(
            key,
            previous_settings.get(key, default),
        )

    if not settings["topic"].strip():
        raise ValueError("Внеси тема за приказната.")

    if not settings["characters"].strip():
        raise ValueError("Внеси барем еден лик.")

    if settings["model"] not in MODELS:
        raise ValueError("Избери валиден модел.")

    if settings["strategy"] not in [
        "Zero-shot",
        "Few-shot",
        "Structured",
    ]:
        raise ValueError("Избери валидна prompting-стратегија.")

    try:
        temperature = float(settings["temperature"])
        word_limit = int(settings["word_limit"])
    except (TypeError, ValueError):
        raise ValueError("Провери ги должината и креативноста.")

    settings["temperature"] = str(max(0.0, min(1.0, temperature)))
    settings["word_limit"] = str(max(50, min(500, word_limit)))

    session["settings"] = settings
    return settings


def make_prompt(settings, previous_story=""):
    prompt = build_story_prompt(
        genre=settings["genre"],
        topic=settings["topic"],
        style=settings["style"],
        characters=settings["characters"],
        language=settings["language"],
        strategy=settings["strategy"],
        previous_story=previous_story,
    )

    prompt += (
        f"\nWrite approximately {settings['word_limit']} words "
        "for this story segment."
    )

    return prompt


def generate(model_name, prompt, settings):
    if model_name not in MODELS:
        raise ValueError("Избраниот модел не е валиден.")

    return MODELS[model_name](
        prompt,
        temperature=float(settings["temperature"]),
    )


def format_error(exception):
    return str(exception)


def save_history(model_name, story, settings):
    data = get_data()

    data["history"].append({
        "model": model_name,
        "story": story,
        "metrics": evaluate_text(story),
        "genre": settings["genre"],
        "strategy": settings["strategy"],
        "language": settings["language"],
    })

    data["history"] = data["history"][-20:]


def fallback_choices(language):
    if language == "Macedonian":
        return [
            "Открива тајна порака.",
            "Се соочува со неочекуван избор.",
            "Запознава мистериозен сојузник.",
        ]

    return [
        "Discover a hidden message.",
        "Face an unexpected choice.",
        "Meet a mysterious ally.",
    ]


def parse_choices(text, language):
    choices = []

    for line in text.splitlines():
        cleaned = re.sub(
            r"^\s*(?:(?:option\s*)?\d+[.):]|[-*•])\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = cleaned.strip("\"' ")

        if not cleaned:
            continue

        cleaned = re.split(r"(?<=[.!?])\s+", cleaned)[0]
        words = cleaned.split()

        if len(words) > 12:
            cleaned = " ".join(words[:12]).rstrip(",;:") + "…"

        if cleaned and cleaned not in choices:
            choices.append(cleaned)

        if len(choices) == 3:
            break

    if len(choices) < 3:
        for choice in fallback_choices(language):
            if choice not in choices:
                choices.append(choice)

            if len(choices) == 3:
                break

    return choices


def make_interactive_choices(state, settings):
    try:
        text = generate(
            state["model"],
            build_choices_prompt(
                state["story"],
                settings["language"],
            ),
            settings,
        )

        return parse_choices(text, settings["language"])

    except Exception:
        return fallback_choices(settings["language"])


@app.route("/")
def index():
    return redirect(url_for("generate_page"))


@app.route("/generate", methods=["GET", "POST"])
def generate_page():
    error = None
    data = get_data()
    result = data["last_result"]
    settings = get_settings()

    if request.method == "POST":
        try:
            settings = read_settings()
            prompt = make_prompt(settings)
            story = generate(settings["model"], prompt, settings)

            result = {
                "model": settings["model"],
                "story": story,
                "metrics": evaluate_text(story),
            }

            data["last_result"] = result
            data["last_prompt"] = prompt

            save_history(
                settings["model"],
                story,
                settings,
            )

        except Exception as exception:
            error = format_error(exception)

    return render_template(
        "generate.html",
        settings=settings,
        result=result,
        prompt=data["last_prompt"],
        error=error,
    )


@app.route("/compare", methods=["GET", "POST"])
def compare_page():
    error = None
    data = get_data()
    comparison = data["last_comparison"]
    settings = get_settings()

    if request.method == "POST":
        try:
            settings = read_settings()
            prompt = make_prompt(settings)

            model_names = [
                "Gemini 3.5 Flash-Lite",
                "GPT OSS 20B",
            ]

            results = {}

            def run_comparison_model(model_name):
                try:
                    story = generate(
                        model_name,
                        prompt,
                        settings,
                    )

                    return {
                        "model": model_name,
                        "story": story,
                        "metrics": evaluate_text(story),
                    }

                except Exception as exception:
                    return {
                        "model": model_name,
                        "error": format_error(exception),
                    }

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        run_comparison_model,
                        model_name,
                    ): model_name
                    for model_name in model_names
                }

                for future in as_completed(futures):
                    model_name = futures[future]
                    results[model_name] = future.result()

            comparison = [
                results[model_name]
                for model_name in model_names
            ]

            for item in comparison:
                if "story" in item:
                    save_history(
                        item["model"],
                        item["story"],
                        settings,
                    )

            data["last_comparison"] = comparison
            data["last_comparison_prompt"] = prompt

        except Exception as exception:
            error = format_error(exception)

    return render_template(
        "compare.html",
        settings=settings,
        comparison=comparison,
        prompt=data["last_comparison_prompt"],
        error=error,
    )


@app.route("/interactive", methods=["GET", "POST"])
def interactive_page():
    error = None
    data = get_data()
    settings = get_settings()
    state = data["interactive"]

    if request.method == "POST":
        action = request.form.get("action")

        try:
            if action == "start":
                settings = read_settings()

                prompt = build_opening_prompt(
                    genre=settings["genre"],
                    topic=settings["topic"],
                    style=settings["style"],
                    characters=settings["characters"],
                    language=settings["language"],
                    strategy=settings["strategy"],
                    word_limit=int(settings["word_limit"]),
                )

                story = generate(
                    settings["model"],
                    prompt,
                    settings,
                )

                state = {
                    "story": story,
                    "choices": [],
                    "finished": False,
                    "ai_continuations": 0,
                    "user_choices": 0,
                    "model": settings["model"],
                    "settings": settings.copy(),
                }

                state["choices"] = make_interactive_choices(
                    state,
                    settings,
                )

                data["interactive"] = state

            elif action == "continue":
                state = data["interactive"]

                if not state or not state.get("story"):
                    raise ValueError(
                        "Нема активна приказна. Прво започни нова."
                    )

                if state.get("finished"):
                    raise ValueError(
                        "Оваа приказна е завршена. Започни нова."
                    )

                settings = state["settings"]
                choice = request.form.get("choice", "").strip()

                if not choice:
                    raise ValueError(
                        "Избери една опција или внеси сопствен избор."
                    )

                prompt = build_story_prompt(
                    genre=settings["genre"],
                    topic=settings["topic"],
                    style=settings["style"],
                    characters=settings["characters"],
                    language=settings["language"],
                    strategy=settings["strategy"],
                    previous_story=state["story"],
                    user_choice=choice,
                )

                prompt += (
                    f"\nWrite the next segment in approximately "
                    f"{settings['word_limit']} words."
                )

                continuation = generate(
                    state["model"],
                    prompt,
                    settings,
                )

                state["story"] += "\n\n" + continuation
                state["ai_continuations"] += 1
                state["user_choices"] += 1

                state["choices"] = make_interactive_choices(
                    state,
                    settings,
                )

                data["interactive"] = state

            elif action == "end":
                state = data["interactive"]

                if not state or not state.get("story"):
                    raise ValueError(
                        "Нема активна приказна. Прво започни нова."
                    )

                if state.get("finished"):
                    raise ValueError(
                        "Приказната веќе е завршена."
                    )

                settings = state["settings"]
                direction = request.form.get("ending", "").strip()

                if not direction:
                    direction = (
                        "Write a satisfying ending that resolves "
                        "the main conflict."
                    )

                ending_prompt = build_ending_prompt(
                    state["story"],
                    settings["language"],
                    direction,
                )

                ending = generate(
                    state["model"],
                    ending_prompt,
                    settings,
                )

                state["story"] += "\n\n" + ending
                state["finished"] = True
                state["ending"] = direction
                state["choices"] = []
                state["metrics"] = evaluate_text(state["story"])

                data["interactive"] = state

                save_history(
                    state["model"],
                    state["story"],
                    settings,
                )

            else:
                raise ValueError("Непозната акција.")

        except Exception as exception:
            error = format_error(exception)

    state = data["interactive"]

    return render_template(
        "interactive.html",
        settings=settings,
        state=state,
        error=error,
    )


@app.route("/results")
def results_page():
    return render_template(
        "results.html",
        history=get_data()["history"],
    )


if __name__ == "__main__":
    app.run(debug=True)
