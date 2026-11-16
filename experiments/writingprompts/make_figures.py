from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = Path(__file__).resolve().parents[2] / "data" / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(results_dir / "all_results.csv")
valid = data[data["valid"]]

MODELS = [model for model in ["Gemini 3.5 Flash-Lite", "GPT OSS 20B", "Llama 3.2 3B"] if model in set(valid["model"])]
STRATEGIES = ["Zero-shot", "Few-shot", "Structured"]
CRITERIA = ["coherence", "creativity", "relevance", "consistency", "engagement", "language_quality"]
CRITERIA_LABELS = ["Coherence", "Creativity", "Relevance", "Consistency", "Engagement", "Language quality"]

MODEL_COLORS = {"Gemini 3.5 Flash-Lite": "#4C78A8", "GPT OSS 20B": "#F58518", "Llama 3.2 3B": "#54A24B"}
STRATEGY_COLORS = ["#9ecae1", "#4292c6", "#08519c"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def mean_ci(series):
    series = series.dropna()
    standard_error = series.std(ddof=1) / np.sqrt(len(series))
    return series.mean(), 1.96 * standard_error


def grouped_bars(ax, groups, bars, get_series, colors, group_labels=None, width=0.8):
    positions = np.arange(len(groups))
    bar_width = width / len(bars)
    for index, bar in enumerate(bars):
        values = [mean_ci(get_series(group, bar)) for group in groups]
        ax.bar(
            positions - width / 2 + bar_width * (index + 0.5),
            [value[0] for value in values],
            bar_width,
            yerr=[value[1] for value in values],
            capsize=3,
            label=bar,
            color=colors[index],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels or groups)


fig, ax = plt.subplots(figsize=(8, 5))
grouped_bars(
    ax, MODELS, STRATEGIES,
    lambda model, strategy: valid[(valid["model"] == model) & (valid["strategy"] == strategy)]["ai_mean"],
    STRATEGY_COLORS,
)
ax.set_ylim(0, 10)
ax.set_ylabel("AI score (mean of 6 criteria, 1-10)")
ax.set_title("AI evaluation score by model and prompting strategy")
ax.legend(title="Strategy", frameon=False)
fig.text(0.01, 0.01, "Error bars: 95% confidence interval", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig(figures_dir / "fig1_ai_score_by_strategy.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
grouped_bars(
    ax, CRITERIA, MODELS,
    lambda criterion, model: valid[valid["model"] == model][criterion],
    [MODEL_COLORS[model] for model in MODELS],
    group_labels=CRITERIA_LABELS,
)
ax.set_ylim(0, 10.5)
ax.set_ylabel("AI score (1-10)")
ax.set_title("AI evaluation scores by criterion and model")
ax.legend(frameon=False, ncol=len(MODELS), loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
fig.savefig(figures_dir / "fig2_criteria_by_model.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
box = ax.boxplot(
    [valid[valid["model"] == model]["word_count"] for model in MODELS],
    patch_artist=True,
    tick_labels=MODELS,
)
for patch, model in zip(box["boxes"], MODELS):
    patch.set_facecolor(MODEL_COLORS[model])
    patch.set_alpha(0.8)
ax.axhline(400, color="red", linestyle="--", label="Requested length (400 words)")
ax.set_ylabel("Words per story")
ax.set_title("Story length by model")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(figures_dir / "fig3_word_count_by_model.png", dpi=200)
plt.close(fig)

panels = [
    ("mattr", "Lexical diversity (MATTR)\nhigher = richer vocabulary"),
    ("repeated_trigram_rate", "Repeated trigram rate\nlower = less repetition"),
    ("rouge_l", "ROUGE-L vs human story"),
    ("bertscore_f1", "BERTScore F1 vs human story"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (metric, title) in zip(axes.flat, panels):
    values = [mean_ci(valid[valid["model"] == model][metric]) for model in MODELS]
    ax.bar(
        np.arange(len(MODELS)),
        [value[0] for value in values],
        yerr=[value[1] for value in values],
        capsize=3,
        color=[MODEL_COLORS[model] for model in MODELS],
    )
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_title(title, fontsize=10)
fig.suptitle("Automatic text metrics by model (95% CI)")
fig.tight_layout()
fig.savefig(figures_dir / "fig4_automatic_metrics.png", dpi=200)
plt.close(fig)

print("Saved 4 figures to", figures_dir)