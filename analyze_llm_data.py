# /// script
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
#     "scipy",
# ]
# ///

import json
import os
from collections import namedtuple
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix

from detect_foreign_words import LANG_NAMES

# Configuration
TEXT_RESULTS_FILE = "llm_evaluation_results.jsonl"
WORD_RESULTS_FILE = "first_llm_word_pure_results.jsonl"
INJECTED_RESULTS_FILE = "first_llm_word_injected_results.jsonl"
PLOTS_DIR = "plots_llm"

# Model display names
MODEL_NAMES = {
    "llama3.1:8b": "Llama 3.1 8B",
    "qwen3:8b": "Qwen 3 8B",
    "ministral-3:8b": "Ministral 3 8B",
}

# Method display names
METHOD_NAMES = {
    "baseline": "Baseline",
    "schema": "Schema",
    "constrained": "Constrained",
}

# ============================================================================
# Helper Functions
# ============================================================================


def model_name(value):
    return MODEL_NAMES.get(value, value)


def method_name(value):
    return METHOD_NAMES.get(value, value)


def lang_name(value):
    return LANG_NAMES.get(value, value)


def to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)

    as_str = str(value).strip().lower()
    if as_str in {"true", "1", "yes", "y"}:
        return True
    if as_str in {"false", "0", "no", "n"}:
        return False
    return default


def to_number(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mcnemar_test(contingency_table):
    """
    Perform McNemar's test on a 2x2 contingency table.

    contingency_table format: [[both_correct, group1_only],
                               [group2_only, both_wrong]]
    """
    b = contingency_table[0][1]
    c = contingency_table[1][0]
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    pvalue = 1 - chi2.cdf(statistic, df=1)
    Result = namedtuple("McNemarResult", ["statistic", "pvalue"])
    return Result(statistic, pvalue)


def parse_jsonl_file(file_path):
    """Parse JSONL line-by-line and skip malformed or non-object rows."""
    records = []
    malformed_rows = 0
    scalar_rows = 0

    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue

            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            if not isinstance(obj, dict):
                scalar_rows += 1
                continue

            obj["_line_number"] = line_number
            records.append(obj)

    frame = pd.DataFrame(records)
    diagnostics = {
        "rows": len(records),
        "malformed_rows": malformed_rows,
        "scalar_rows": scalar_rows,
    }
    return frame, diagnostics


def normalize_common_columns(df):
    """Normalize shared fields across task datasets."""
    normalized = df.copy()

    for col in ["model", "method", "true_lang", "detected_lang", "source"]:
        if col not in normalized.columns:
            normalized[col] = "unknown"
        normalized[col] = normalized[col].fillna("unknown").astype(str)

    if "correct" not in normalized.columns:
        normalized["correct"] = normalized["true_lang"] == normalized["detected_lang"]
    normalized["correct"] = normalized["correct"].apply(to_bool)

    if "llm_valid_json" not in normalized.columns:
        normalized["llm_valid_json"] = True
    normalized["llm_valid_json"] = normalized["llm_valid_json"].apply(
        lambda x: to_bool(x, default=True)
    )

    if "llm_retry_count" not in normalized.columns:
        normalized["llm_retry_count"] = 0
    normalized["llm_retry_count"] = (
        normalized["llm_retry_count"]
        .apply(lambda x: int(to_number(x, default=0)))
        .astype(int)
    )

    if "confidence" not in normalized.columns:
        normalized["confidence"] = np.nan
    normalized["confidence"] = normalized["confidence"].apply(to_number)

    if "llm_latency_ms" not in normalized.columns:
        normalized["llm_latency_ms"] = np.nan
    normalized["llm_latency_ms"] = normalized["llm_latency_ms"].apply(to_number)

    return normalized


# ============================================================================
# Data Loading
# ============================================================================


def load_data():
    """Load and normalize all LLM evaluation datasets."""
    print("Loading LLM evaluation data...")

    text_df, text_diag = parse_jsonl_file(TEXT_RESULTS_FILE)
    word_df, word_diag = parse_jsonl_file(WORD_RESULTS_FILE)
    injected_df, injected_diag = parse_jsonl_file(INJECTED_RESULTS_FILE)

    text_df = normalize_common_columns(text_df)
    word_df = normalize_common_columns(word_df)
    injected_df = normalize_common_columns(injected_df)

    if "main_correct" in injected_df.columns:
        injected_df["main_correct"] = injected_df["main_correct"].apply(to_bool)
    else:
        injected_df["main_correct"] = injected_df["correct"]

    for col in ["foreign_precision", "foreign_recall", "foreign_f1"]:
        if col not in injected_df.columns:
            injected_df[col] = np.nan
        injected_df[col] = injected_df[col].apply(to_number)

    print(f"Loaded {len(text_df)} text-level records from {TEXT_RESULTS_FILE}")
    print(f"Loaded {len(word_df)} pure-word records from {WORD_RESULTS_FILE}")
    print(
        f"Loaded {len(injected_df)} injected-context records from {INJECTED_RESULTS_FILE}"
    )
    print()

    print("Input data quality diagnostics:")
    print(
        f"  Text: malformed={text_diag['malformed_rows']}, scalar={text_diag['scalar_rows']}"
    )
    print(
        f"  Word: malformed={word_diag['malformed_rows']}, scalar={word_diag['scalar_rows']}"
    )
    print(
        f"  Injected: malformed={injected_diag['malformed_rows']}, scalar={injected_diag['scalar_rows']}"
    )
    print()

    return text_df, word_df, injected_df


# ============================================================================
# Analysis Functions
# ============================================================================


def print_accuracy_by_model_method(df, metric_col, title):
    print(title)
    grouped = (
        df.groupby(["model", "method"])[metric_col]
        .mean()
        .sort_index(
            key=lambda idx: idx.map(
                lambda x: model_name(x) if x in MODEL_NAMES else method_name(x)
            )
        )
    )

    for (model, method), value in grouped.items():
        print(
            f"  {model_name(model):20s} | {method_name(method):12s}: {value * 100:6.2f}%"
        )
    print()


def print_per_language_accuracy(df, metric_col, title):
    print(title)
    lang_accuracy = (
        df.pivot_table(
            values=metric_col,
            index="true_lang",
            columns=["model", "method"],
            aggfunc="mean",
        )
        * 100
    )

    lang_order = sorted(lang_accuracy.index, key=lang_name)
    lang_accuracy = lang_accuracy.loc[lang_order]

    for lang in lang_accuracy.index:
        line_parts = [f"  {lang_name(lang):15s}"]
        for model, method in lang_accuracy.columns:
            line_parts.append(
                f"{model_name(model)}[{method_name(method)}]={lang_accuracy.loc[lang, (model, method)]:5.1f}%"
            )
        print(" | ".join(line_parts))
    print()

    return lang_accuracy


def print_top_misclassifications(df, title, max_rows=10):
    print(title)
    misclassified = (
        df[~df["correct"]]
        .groupby(["true_lang", "detected_lang"])
        .size()
        .sort_values(ascending=False)
        .head(max_rows)
    )

    if misclassified.empty:
        print("  No misclassifications found.")
    else:
        for (true_lang, detected_lang), count in misclassified.items():
            print(
                f"  {lang_name(true_lang):15s} -> {lang_name(detected_lang):15s}: {count:4d}"
            )
    print()


def print_quality_metrics(df, title):
    print(title)
    grouped = df.groupby(["model", "method"]).agg(
        invalid_json_rate=("llm_valid_json", lambda s: (1 - s.mean()) * 100),
        avg_confidence=("confidence", "mean"),
        avg_latency_ms=("llm_latency_ms", "mean"),
        median_latency_ms=("llm_latency_ms", "median"),
        avg_retries=("llm_retry_count", "mean"),
    )

    for (model, method), row in grouped.iterrows():
        print(
            f"  {model_name(model):20s} | {method_name(method):12s}: "
            f"invalid={row['invalid_json_rate']:5.2f}%, "
            f"conf={row['avg_confidence']:.3f}, "
            f"lat(ms)={row['avg_latency_ms']:.1f}, "
            f"retry={row['avg_retries']:.2f}"
        )
    print()


def analyze_text_level(text_df):
    print("=" * 80)
    print("TEXT-LEVEL LLM EVALUATION")
    print("=" * 80)
    print()

    print_accuracy_by_model_method(
        text_df, "correct", "Overall Accuracy by Model/Method:"
    )
    text_lang_accuracy = print_per_language_accuracy(
        text_df, "correct", "Per-Language Accuracy (Text-Level):"
    )
    print_top_misclassifications(text_df, "Top Misclassifications (Text-Level):")
    print_quality_metrics(text_df, "Quality Metrics (Text-Level):")

    return text_lang_accuracy


def analyze_word_level(word_df):
    print("=" * 80)
    print("PURE-WORD LLM EVALUATION")
    print("=" * 80)
    print()

    print_accuracy_by_model_method(
        word_df, "correct", "Overall Accuracy by Model/Method:"
    )
    word_lang_accuracy = print_per_language_accuracy(
        word_df, "correct", "Per-Language Accuracy (Pure-Word):"
    )
    print_top_misclassifications(word_df, "Top Misclassifications (Pure-Word):")
    print_quality_metrics(word_df, "Quality Metrics (Pure-Word):")

    return word_lang_accuracy


def analyze_injected_level(injected_df):
    print("=" * 80)
    print("INJECTED-CONTEXT LLM EVALUATION")
    print("=" * 80)
    print()

    print_accuracy_by_model_method(
        injected_df, "main_correct", "Main-Language Accuracy by Model/Method:"
    )

    print("Foreign-word Metrics by Model/Method:")
    grouped = injected_df.groupby(["model", "method"]).agg(
        precision=("foreign_precision", "mean"),
        recall=("foreign_recall", "mean"),
        f1=("foreign_f1", "mean"),
        injected_count=("injected_count", "mean"),
        predicted_count=("predicted_count", "mean"),
    )

    for (model, method), row in grouped.iterrows():
        print(
            f"  {model_name(model):20s} | {method_name(method):12s}: "
            f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, "
            f"inj={row['injected_count']:.2f}, pred={row['predicted_count']:.2f}"
        )
    print()

    print_quality_metrics(injected_df, "Quality Metrics (Injected-Context):")


# ============================================================================
# Statistical Significance Testing
# ============================================================================


def perform_mcnemar_tests(df, metric_col, id_columns, level_name):
    """Perform pairwise McNemar tests between model/method groups."""
    print(f"Statistical Significance Tests ({level_name}):")
    print("(McNemar's test, alpha = 0.05)")
    print()

    groups = sorted(
        df[["model", "method"]].drop_duplicates().itertuples(index=False, name=None)
    )

    if len(groups) < 2:
        print("  Not enough groups for pairwise testing.")
        print()
        return

    for (model_1, method_1), (model_2, method_2) in combinations(groups, 2):
        g1 = df[(df["model"] == model_1) & (df["method"] == method_1)]
        g2 = df[(df["model"] == model_2) & (df["method"] == method_2)]

        left = g1[id_columns + [metric_col]].rename(columns={metric_col: "g1_correct"})
        right = g2[id_columns + [metric_col]].rename(columns={metric_col: "g2_correct"})
        merged = left.merge(right, on=id_columns, how="inner")

        if merged.empty:
            continue

        g1_correct = merged["g1_correct"].apply(to_bool)
        g2_correct = merged["g2_correct"].apply(to_bool)

        both_correct = ((g1_correct) & (g2_correct)).sum()
        g1_only = ((g1_correct) & (~g2_correct)).sum()
        g2_only = ((~g1_correct) & (g2_correct)).sum()
        both_wrong = ((~g1_correct) & (~g2_correct)).sum()

        contingency = [[both_correct, g1_only], [g2_only, both_wrong]]
        result = mcnemar_test(contingency)

        significance = (
            "***"
            if result.pvalue < 0.001
            else "**"
            if result.pvalue < 0.01
            else "*"
            if result.pvalue < 0.05
            else "ns"
        )
        label_1 = f"{model_name(model_1)}[{method_name(method_1)}]"
        label_2 = f"{model_name(model_2)}[{method_name(method_2)}]"
        print(
            f"  {label_1:35s} vs {label_2:35s}: p = {result.pvalue:.4f} {significance}"
        )

    print()
    print("  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print()


# ============================================================================
# Visualization Functions
# ============================================================================


def setup_plot_style():
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def _plot_grouped_accuracy(df, metric_col, title, output_file):
    grouped = df.groupby(["model", "method"])[metric_col].mean().reset_index()
    grouped["score_percent"] = grouped[metric_col] * 100
    grouped["model_name"] = grouped["model"].map(model_name)
    grouped["method_name"] = grouped["method"].map(method_name)

    model_order = sorted(grouped["model_name"].unique())
    method_order = sorted(grouped["method_name"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=grouped,
        x="model_name",
        y="score_percent",
        hue="method_name",
        order=model_order,
        hue_order=method_order,
        palette="Set2",
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_injected_metrics(injected_df, output_file):
    grouped = (
        injected_df.groupby(["model", "method"])
        .agg(
            main_accuracy=("main_correct", "mean"),
            foreign_f1=("foreign_f1", "mean"),
        )
        .reset_index()
    )

    grouped["model_name"] = grouped["model"].map(model_name)
    grouped["method_name"] = grouped["method"].map(method_name)
    grouped["main_accuracy"] = grouped["main_accuracy"] * 100
    grouped["foreign_f1"] = grouped["foreign_f1"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(
        data=grouped,
        x="model_name",
        y="main_accuracy",
        hue="method_name",
        palette="Set2",
        ax=ax1,
    )
    ax1.set_title("Injected: Main-Language Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Model", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=15)

    sns.barplot(
        data=grouped,
        x="model_name",
        y="foreign_f1",
        hue="method_name",
        palette="Set2",
        ax=ax2,
    )
    ax2.set_title("Injected: Foreign-Word F1", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Model", fontsize=10, fontweight="bold")
    ax2.set_ylabel("F1 (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=15)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title="Method", fontsize=8)
    ax2.legend().remove()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_confidence_distributions(text_df, word_df, output_file):
    text_view = text_df[["model", "method", "confidence"]].copy()
    text_view["task"] = "text"

    word_view = word_df[["model", "method", "confidence"]].copy()
    word_view["task"] = "pure_word"

    combined = pd.concat([text_view, word_view], ignore_index=True)
    combined["method_name"] = combined["method"].map(method_name)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=combined,
        x="method_name",
        y="confidence",
        hue="task",
        palette="Set2",
        ax=ax,
    )

    ax.set_title("Confidence Distribution by Method", fontsize=14, fontweight="bold")
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Confidence", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_latency_comparisons(text_df, word_df, injected_df, output_file):
    rows = []
    for task_name, frame in [
        ("text", text_df),
        ("pure_word", word_df),
        ("injected", injected_df),
    ]:
        subset = frame[["method", "llm_latency_ms"]].copy()
        subset["task"] = task_name
        rows.append(subset)

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.dropna(subset=["llm_latency_ms"])
    combined["method_name"] = combined["method"].map(method_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=combined,
        x="task",
        y="llm_latency_ms",
        hue="method_name",
        palette="Set2",
        ax=ax,
    )

    ax.set_title("Latency Comparison by Task/Method", fontsize=14, fontweight="bold")
    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_invalid_json_rate_heatmap(text_df, word_df, injected_df, output_file):
    stacked = pd.concat(
        [
            text_df.assign(task="text"),
            word_df.assign(task="pure_word"),
            injected_df.assign(task="injected"),
        ],
        ignore_index=True,
    )

    grouped = (
        stacked.groupby(["task", "model", "method"])["llm_valid_json"]
        .mean()
        .reset_index()
    )
    grouped["invalid_rate"] = (1 - grouped["llm_valid_json"]) * 100
    grouped["group"] = (
        grouped["model"].map(model_name) + " | " + grouped["method"].map(method_name)
    )

    pivot = grouped.pivot(index="task", columns="group", values="invalid_rate")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=max(5.0, float(np.nanmax(pivot.values)) if not pivot.empty else 5.0),
        cbar_kws={"label": "Invalid JSON (%)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_title(
        "Invalid JSON Rate by Task/Model/Method", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Model | Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Task", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_text_language_heatmap(text_df, output_file):
    lang_accuracy = (
        text_df.pivot_table(
            values="correct",
            index="true_lang",
            columns=["model", "method"],
            aggfunc="mean",
        )
        * 100
    )

    if lang_accuracy.empty:
        print("Skipped text language heatmap: no data available")
        return

    column_order = sorted(
        lang_accuracy.columns,
        key=lambda pair: (model_name(pair[0]), method_name(pair[1])),
    )
    lang_accuracy = lang_accuracy[column_order]
    lang_accuracy.columns = [
        f"{model_name(m)}|{method_name(t)}" for m, t in lang_accuracy.columns
    ]

    row_order = sorted(lang_accuracy.index, key=lang_name)
    lang_accuracy = lang_accuracy.loc[row_order]
    lang_accuracy.index = [lang_name(lang) for lang in lang_accuracy.index]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        lang_accuracy,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_title("Text-Level Per-Language Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model | Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Language", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_text_confusion_matrices(text_df):
    """Create confusion matrices for each model/method pair for text-level data."""
    pairs = sorted(
        text_df[["model", "method"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for idx, (model, method) in enumerate(pairs, start=8):
        subset = text_df[(text_df["model"] == model) & (text_df["method"] == method)]
        languages = sorted(subset["true_lang"].dropna().unique(), key=lang_name)

        if not languages:
            continue

        cm = confusion_matrix(
            subset["true_lang"], subset["detected_lang"], labels=languages
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm.astype(float) / row_sums * 100

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=[lang_name(v) for v in languages],
            yticklabels=[lang_name(v) for v in languages],
            cbar_kws={"label": "Percentage (%)"},
            linewidths=0.5,
            linecolor="gray",
            ax=ax,
        )

        ax.set_title(
            f"Confusion Matrix: {model_name(model)} [{method_name(method)}]",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Detected Language", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Language", fontsize=12, fontweight="bold")

        output_file = os.path.join(
            PLOTS_DIR,
            f"{idx:02d}_confusion_matrix_{model.replace('.', '_').replace(':', '_')}_{method}.png",
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Created/verified plots directory: {PLOTS_DIR}")
    print()

    setup_plot_style()
    text_df, word_df, injected_df = load_data()

    analyze_text_level(text_df)
    perform_mcnemar_tests(
        text_df,
        metric_col="correct",
        id_columns=["text", "true_lang"],
        level_name="Text-Level",
    )

    analyze_word_level(word_df)
    perform_mcnemar_tests(
        word_df,
        metric_col="correct",
        id_columns=["sample_id", "source_index", "true_lang", "normalized_word"],
        level_name="Pure-Word",
    )

    analyze_injected_level(injected_df)
    perform_mcnemar_tests(
        injected_df,
        metric_col="main_correct",
        id_columns=["sample_id", "base_lang"],
        level_name="Injected-Context (Main Language)",
    )

    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()

    _plot_grouped_accuracy(
        text_df,
        metric_col="correct",
        title="Text-Level Accuracy by Model/Method",
        output_file=os.path.join(PLOTS_DIR, "01_text_accuracy_by_model_method.png"),
    )
    _plot_grouped_accuracy(
        word_df,
        metric_col="correct",
        title="Pure-Word Accuracy by Model/Method",
        output_file=os.path.join(
            PLOTS_DIR, "02_pure_word_accuracy_by_model_method.png"
        ),
    )
    plot_injected_metrics(
        injected_df,
        output_file=os.path.join(PLOTS_DIR, "03_injected_context_metrics.png"),
    )
    plot_confidence_distributions(
        text_df,
        word_df,
        output_file=os.path.join(PLOTS_DIR, "04_confidence_distributions.png"),
    )
    plot_latency_comparisons(
        text_df,
        word_df,
        injected_df,
        output_file=os.path.join(PLOTS_DIR, "05_latency_comparisons.png"),
    )
    plot_invalid_json_rate_heatmap(
        text_df,
        word_df,
        injected_df,
        output_file=os.path.join(PLOTS_DIR, "06_invalid_json_rate_heatmap.png"),
    )
    plot_text_language_heatmap(
        text_df,
        output_file=os.path.join(PLOTS_DIR, "07_text_language_heatmap.png"),
    )
    plot_text_confusion_matrices(text_df)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {PLOTS_DIR}/")
    print()


if __name__ == "__main__":
    main()
