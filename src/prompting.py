def build_story_prompt(
    genre,
    topic,
    style,
    characters,
    language,
    strategy,
    previous_story="",
    user_choice="",
):
    task = f"""
Write in {language}.

Genre: {genre}
Topic: {topic}
Writing style: {style}
Characters: {characters}

Requirements:
- Write an engaging and coherent story.
- Keep characters and events consistent.
- Follow the selected genre and writing style.
- Write natural prose, not a list.
"""

    if previous_story:
        task += f"""
Continue the existing story. Do not restart or repeat it.

Previous story:
{previous_story}
"""

    if user_choice:
        task += f"""
The user's choice for the next event is:
{user_choice}

Make this choice clearly affect what happens next.
"""

    if strategy == "Zero-shot":
        instruction = task

    elif strategy == "Few-shot":
        instruction = f"""
Example:
Previous scene: A traveler finds a mysterious key.
User choice: Open the old wooden door.
Continuation: Behind the door is a forgotten library.

Use this only as an example of a continuation.
Do not reuse its characters or events.

{task}
"""

    elif strategy == "Structured":
        instruction = f"""
Follow these requirements:
1. Continue existing events when a previous story is provided.
2. Incorporate the user's choice when provided.
3. Maintain consistency.
4. Return only the story, without explanations.

{task}
"""

    else:
        raise ValueError(f"Unknown prompting strategy: {strategy}")

    instruction += "\nReturn only the requested story text."
    return instruction.strip()


def build_opening_prompt(
    genre, topic, style, characters, language, strategy, word_limit=150
):
    prompt = build_story_prompt(
        genre, topic, style, characters, language, strategy
    )
    return (
        prompt
        + f"\nWrite the opening segment in approximately {word_limit} words."
    )


def build_choices_prompt(previous_story, language):
    return f"""
Based on the story below, suggest exactly three possible next events.

Story:
{previous_story}

Language: {language}

Strict requirements:
- Return exactly three options, numbered 1., 2., and 3.
- Each option must be one short sentence of no more than 12 words.
- Describe one specific thing that could happen next.
- Make the three options clearly different.
- Do not write paragraphs, explanations, dialogue, or story continuations.
- Do not include an introduction or conclusion.
"""


def build_ending_prompt(previous_story, language, user_choice="End the story"):
    return f"""
Write a coherent ending in {language} for the existing story.

Resolve the main conflict naturally and respect the established characters
and events. Do not restart or repeat the story.

Story so far:
{previous_story}

Ending direction:
{user_choice}

Return only the ending text.
"""
