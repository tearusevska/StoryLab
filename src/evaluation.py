import re

def tokenize_text(text):
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)

def evaluate_text(text):
    words = tokenize_text(text)
    word_count = len(words)
    unique_words = len(set(words))

    if word_count == 0:
        return {
            "word_count": 0,
            "lexical_diversity": 0.0,
            "repetition_rate": 0.0
        }

    lexical_diversity = unique_words / word_count
    repetition_rate = 1 - lexical_diversity

    return {
        "word_count": word_count,
        "lexical_diversity": round(lexical_diversity, 3),
        "repetition_rate": round(repetition_rate, 3)
    }
