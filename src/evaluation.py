import re

def tokenize_text(text):
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)

def moving_average_ttr(words, window=50):
    if not words:
        return 0.0

    if len(words) < window:
        return len(set(words)) / len(words)

    ratios = [
        len(set(words[start:start + window])) / window
        for start in range(len(words) - window + 1)
    ]
    return sum(ratios) / len(ratios)

def evaluate_story_metrics(text, window=50, prefix_words=300):
    words = tokenize_text(text)
    word_count = len(words)

    if word_count < 3:
        return {
            "word_count": word_count,
            "mattr": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repeated_trigram_rate": 0.0,
        }

    prefix = words[:prefix_words]
    bigrams = list(zip(prefix, prefix[1:]))
    trigrams = list(zip(prefix, prefix[1:], prefix[2:]))

    return {
        "word_count": word_count,
        "mattr": round(moving_average_ttr(words, window), 3),
        "distinct_1": round(len(set(prefix)) / len(prefix), 3),
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 3),
        "repeated_trigram_rate": round(1 - len(set(trigrams)) / len(trigrams), 3),
    }