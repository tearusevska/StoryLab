import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

results_dir = Path(__file__).resolve().parents[2] / "data" / "results"

data = pd.read_csv(results_dir / "all_results.csv")
valid = data[data["valid"]]

METRICS = [
    "ai_mean", "coherence", "creativity", "relevance", "consistency", "engagement", "language_quality",
    "length_deviation", "mattr", "distinct_2", "repeated_trigram_rate",
    "rouge_l", "bertscore_f1", "bleu",
]

CORRELATION_METRICS = ["word_count", "mattr", "distinct_2", "rouge_l", "bertscore_f1", "bleu"]

MODELS = sorted(valid["model"].unique())
STRATEGIES = ["Zero-shot", "Few-shot", "Structured"]


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        value = (len(p_values) - rank) * p_values[index]
        running_max = max(running_max, value)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def format_p(value):
    if pd.isna(value):
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def compare(wide, factor, within, metric):
    wide = wide.dropna()
    if len(wide) < 5:
        return []

    try:
        friedman_p = stats.friedmanchisquare(*[wide[column] for column in wide.columns]).pvalue
    except ValueError:
        friedman_p = np.nan

    rows = []
    for first, second in itertools.combinations(wide.columns, 2):
        difference = wide[first] - wide[second]
        if difference.abs().sum() == 0:
            p_value = 1.0
        else:
            p_value = stats.wilcoxon(wide[first], wide[second]).pvalue
        spread = difference.std(ddof=1)
        effect = difference.mean() / spread if spread > 0 else np.nan
        rows.append({
            "factor": factor,
            "within": within,
            "metric": metric,
            "comparison": f"{first} vs {second}",
            "n_prompts": len(wide),
            "mean_a": round(wide[first].mean(), 4),
            "mean_b": round(wide[second].mean(), 4),
            "mean_diff": round(difference.mean(), 4),
            "cohens_dz": round(effect, 3) if not np.isnan(effect) else np.nan,
            "p_raw": p_value,
            "friedman_p": friedman_p,
        })

    adjusted = holm(np.array([row["p_raw"] for row in rows]))
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    return rows


all_rows = []

for metric in METRICS:
    wide = valid.groupby(["prompt_id", "model"])[metric].mean().unstack("model")
    all_rows += compare(wide[MODELS], "model", "All strategies", metric)

    wide = valid.groupby(["prompt_id", "strategy"])[metric].mean().unstack("strategy")
    all_rows += compare(wide[STRATEGIES], "strategy", "All models", metric)

    for model in MODELS:
        subset = valid[valid["model"] == model]
        wide = subset.pivot(index="prompt_id", columns="strategy", values=metric)
        all_rows += compare(wide[STRATEGIES], "strategy", model, metric)

pairwise = pd.DataFrame(all_rows)
pairwise.to_csv(results_dir / "stats_pairwise.csv", index=False, encoding="utf-8")

print("=== MODEL COMPARISON (paired by prompt, Wilcoxon + Holm) ===")
model_rows = pairwise[(pairwise["factor"] == "model") & (pairwise["metric"].isin(["ai_mean", "length_deviation"]))]
for _, row in model_rows.iterrows():
    print(
        f"{row['metric']:<17} {row['comparison']:<45} diff={row['mean_diff']:>8}  "
        f"d_z={row['cohens_dz']:>6}  p_holm={format_p(row['p_holm'])}"
    )

print("\n=== FRIEDMAN p-values: do models differ? ===")
friedman_models = pairwise[pairwise["factor"] == "model"].drop_duplicates("metric").set_index("metric")["friedman_p"]
print(friedman_models.map(format_p).to_string())

print("\n=== FRIEDMAN p-values: do strategies differ? ===")
friedman_strategies = (
    pairwise[pairwise["factor"] == "strategy"]
    .drop_duplicates(["within", "metric"])
    .pivot(index="metric", columns="within", values="friedman_p")
    .reindex(METRICS)
)
print(friedman_strategies.map(format_p))

correlation_rows = []
groups = [("All models", valid)] + [(model, valid[valid["model"] == model]) for model in MODELS]
for name, subset in groups:
    for metric in CORRELATION_METRICS:
        pair = subset[["ai_mean", metric]].dropna()
        rho, p_value = stats.spearmanr(pair["ai_mean"], pair[metric])
        correlation_rows.append({"group": name, "metric": metric, "spearman_rho": round(rho, 3), "p_value": p_value, "n": len(pair)})

correlations = pd.DataFrame(correlation_rows)
correlations.to_csv(results_dir / "stats_correlations.csv", index=False, encoding="utf-8")

print("\n=== SPEARMAN correlation of ai_mean with other metrics ===")
print(correlations.pivot(index="metric", columns="group", values="spearman_rho").reindex(CORRELATION_METRICS))

print("\nSaved: stats_pairwise.csv, stats_correlations.csv in", results_dir)