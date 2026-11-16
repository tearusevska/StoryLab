def build_writing_prompt(writing_prompt, strategy, examples=None, target_words=400):
    base = (
        f"Write a short story of about {target_words} words "
        f"based on this writing prompt:\n\n{writing_prompt}"
    )

    if strategy == "Zero-shot":
        instruction = base

    elif strategy == "Few-shot":
        if not examples:
            raise ValueError("Few-shot strategy requires examples.")

        example_blocks = []
        for number, example in enumerate(examples, start=1):
            example_blocks.append(
                f"Example {number}\n"
                f"Writing prompt: {example['prompt']}\n"
                f"Story:\n{example['story']}"
            )

        instruction = (
            "Below are examples of writing prompts and the stories written for them.\n\n"
            + "\n\n".join(example_blocks)
            + "\n\nNow write a new story in the same way. "
            "Do not reuse the characters or events from the examples.\n\n"
            + base
        )

    elif strategy == "Structured":
        instruction = (
            f"{base}\n\n"
            "Requirements:\n"
            "1. Stay faithful to the writing prompt.\n"
            "2. Give the story a clear beginning, a development, and an ending.\n"
            "3. Keep characters, setting, and events consistent.\n"
            "4. Make the story engaging and original.\n"
            "5. Use natural, fluent prose."
        )

    else:
        raise ValueError(f"Unknown prompting strategy: {strategy}")

    return instruction + "\n\nReturn only the story text."