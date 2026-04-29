import argparse
import csv
import html
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emse-thesis-mpl-config")
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RUN_DIR = Path("evaluation_results/flores_foreign_words_run_all")
PLOTS_SUBDIR = "plots"
MODEL_DISPLAY_NAMES = {
    "facebook-fasttext-language-identification": "fasttext",
    "glotlid": "glotlid",
    "lingua": "lingua",
    "lingua-spanish-only": "lingua-es",
    "spanish-binary-baseline": "binary-es",
}
TASK_COLORS = {
    "Injected": "#3A86FF",
    "Phrase": "#FB5607",
}
MODEL_COLORS = {
    "fasttext": "#2A9D8F",
    "glotlid": "#F4A261",
    "lingua": "#6C8CC3",
    "binary-es": "#D16BA5",
    "lingua-es": "#A7C957",
}
FALLBACK_MODEL_COLORS = ["#2A9D8F", "#F4A261", "#6C8CC3", "#D16BA5", "#A7C957", "#577590"]

STANDARD_TASKS = {
    "injected": "injected_detection_metrics.csv",
    "phrase": "phrase_detection_metrics.csv",
}

WINDOW_TASKS = {
    "injected": "injected_window_detection_metrics.csv",
    "phrase": "phrase_window_detection_metrics.csv",
}

PURE_FILES = {
    "pure": "pure_foreign_detection_metrics.csv",
    "pure_window": "pure_window_detection_metrics.csv",
}

WINDOW_FIELDS = [
    "window_decision_rule",
    "window_size",
    "window_foreign_threshold",
    "window_shared_foreign_threshold",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze one evaluate_methods.py run for foreign-word detection."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports and plots. Defaults to <run-dir>/analysis_methods.",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip HTML report generation.",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def display_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, str(model_name))


def display_scope_name(scope):
    return "window" if scope == "window" else "standard"


def read_metric_csv(run_dir, filename, task_name, scope):
    path = Path(run_dir) / filename
    if not path.exists():
        return pd.DataFrame(), {"artifact": filename, "status": "missing"}

    frame = pd.read_csv(path)
    frame["task_name"] = task_name
    frame["scope"] = scope
    frame["source_file"] = filename
    frame["run_name"] = Path(run_dir).name
    for field in WINDOW_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "foreign_false_positive_rate",
        "window_size",
        "window_foreign_threshold",
        "window_shared_foreign_threshold",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, {"artifact": filename, "status": "loaded", "rows": len(frame)}


def load_run_data(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    metadata = load_json(run_dir / "run_metadata.json")

    diagnostics = []
    standard_frames = []
    window_frames = []
    pure_frames = []

    for task_name, filename in STANDARD_TASKS.items():
        frame, diagnostic = read_metric_csv(run_dir, filename, task_name, "standard")
        diagnostics.append(diagnostic)
        if not frame.empty:
            standard_frames.append(frame)

    for task_name, filename in WINDOW_TASKS.items():
        frame, diagnostic = read_metric_csv(run_dir, filename, task_name, "window")
        diagnostics.append(diagnostic)
        if not frame.empty:
            window_frames.append(frame)

    for task_name, filename in PURE_FILES.items():
        frame, diagnostic = read_metric_csv(run_dir, filename, task_name, task_name)
        diagnostics.append(diagnostic)
        if not frame.empty:
            pure_frames.append(frame)

    standard_df = pd.concat(standard_frames, ignore_index=True) if standard_frames else pd.DataFrame()
    window_df = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
    pure_df = pd.concat(pure_frames, ignore_index=True) if pure_frames else pd.DataFrame()
    diagnostics_df = pd.DataFrame(diagnostics)
    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "standard_df": standard_df,
        "window_df": window_df,
        "pure_df": pure_df,
        "diagnostics_df": diagnostics_df,
    }


def build_window_config_label(row):
    parts = [
        str(row.get("window_decision_rule") or "unknown"),
        f"w={format_number(row.get('window_size'))}",
        f"t={format_number(row.get('window_foreign_threshold'))}",
    ]
    shared = row.get("window_shared_foreign_threshold")
    if pd.notna(shared):
        parts.append(f"shared={format_number(shared)}")
    return ", ".join(parts)


def build_candidate_display_label(row):
    return f"{display_model_name(row['model'])} · {display_scope_name(row['candidate_scope'])}"


def build_candidate_axis_label(row):
    return f"{display_model_name(row['model'])}\n{display_scope_name(row['candidate_scope'])}"


def build_candidate_annotation_label(row):
    return display_model_name(row["model"])


def compact_config_label(value):
    if value is None or pd.isna(value):
        return "standard"
    text = str(value)
    if text == "standard":
        return text
    return (
        text.replace("legacy_window", "legacy")
        .replace("contextual_hybrid", "context")
        .replace(", ", " ")
        .replace("shared=", "s=")
    )


def format_number(value):
    if value is None or pd.isna(value):
        return "na"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def select_best_rows(frame):
    if frame.empty:
        return frame.copy()

    ordered = frame.sort_values(
        ["task_name", "model", "f1", "recall", "false_positive_rate", "precision"],
        ascending=[True, True, False, False, True, False],
        na_position="last",
    )
    return ordered.groupby(["task_name", "model"], as_index=False, sort=False).head(1).reset_index(drop=True)


def build_candidate_summary(standard_df, window_df):
    standard_best = select_best_rows(standard_df)
    window_best = select_best_rows(window_df)

    summary_frames = []
    if not standard_best.empty:
        standard_rows = standard_best.copy()
        standard_rows["candidate_scope"] = "standard"
        standard_rows["candidate_label"] = standard_rows["model"].astype(str) + " / standard"
        standard_rows["config_label"] = "standard"
        summary_frames.append(standard_rows)

    if not window_best.empty:
        window_rows = window_best.copy()
        window_rows["candidate_scope"] = "window"
        window_rows["config_label"] = window_rows.apply(build_window_config_label, axis=1)
        window_rows["candidate_label"] = (
            window_rows["model"].astype(str) + " / window-best"
        )
        summary_frames.append(window_rows)

    if not summary_frames:
        return pd.DataFrame()

    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    summary["candidate_key"] = summary["model"].astype(str) + "||" + summary["candidate_scope"].astype(str)
    summary["model_display"] = summary["model"].map(display_model_name)
    summary["candidate_display_label"] = summary.apply(build_candidate_display_label, axis=1)
    summary["candidate_axis_label"] = summary.apply(build_candidate_axis_label, axis=1)
    summary["candidate_annotation_label"] = summary.apply(
        build_candidate_annotation_label, axis=1
    )
    return summary.sort_values(
        ["task_name", "f1", "recall", "false_positive_rate", "precision"],
        ascending=[True, False, False, True, False],
        na_position="last",
    ).reset_index(drop=True)


def compute_overall_scores(candidate_summary):
    if candidate_summary.empty:
        return pd.DataFrame()

    grouped = (
        candidate_summary.groupby(
            [
                "candidate_key",
                "candidate_label",
                "candidate_display_label",
                "candidate_axis_label",
                "candidate_annotation_label",
                "candidate_scope",
                "model",
                "model_display",
                "model_family",
            ],
            dropna=False,
        )[["f1", "recall", "precision", "false_positive_rate"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    task_map = (
        candidate_summary.pivot_table(
            index="candidate_key",
            columns="task_name",
            values="config_label",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    grouped = grouped.merge(task_map, on="candidate_key", how="left")
    grouped["tasks_covered"] = candidate_summary.groupby("candidate_key")["task_name"].nunique().reset_index(drop=True)
    grouped = grouped.sort_values(
        ["tasks_covered", "f1", "recall", "false_positive_rate", "precision"],
        ascending=[False, False, False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    grouped["rank"] = range(1, len(grouped) + 1)
    return grouped


def compute_winners(candidate_summary):
    if candidate_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    winners = []
    for task_name in ["injected", "phrase"]:
        task_rows = candidate_summary[candidate_summary["task_name"] == task_name].copy()
        if task_rows.empty:
            continue
        winner = task_rows.sort_values(
            ["f1", "recall", "false_positive_rate", "precision"],
            ascending=[False, False, True, False],
            na_position="last",
        ).head(1).copy()
        winner["winner_type"] = task_name
        winners.append(winner)

    overall_df = compute_overall_scores(candidate_summary)
    if not overall_df.empty:
        overall_winner = overall_df.head(1).copy()
        overall_winner["winner_type"] = "overall"
        winners.append(overall_winner)

    winners_df = pd.concat(winners, ignore_index=True, sort=False) if winners else pd.DataFrame()
    return winners_df, overall_df


def build_plot_candidates(candidate_summary, top_n_annotations=6):
    if candidate_summary.empty:
        return candidate_summary.copy()

    plot_df = candidate_summary.copy()
    plot_df["task_label"] = plot_df["task_name"].map(
        {"injected": "Injected", "phrase": "Phrase"}
    ).fillna(plot_df["task_name"])
    ranked = plot_df.sort_values(
        ["task_name", "f1", "recall", "false_positive_rate", "precision"],
        ascending=[True, False, False, True, False],
        na_position="last",
    )
    top_keys = (
        ranked.groupby("task_name", sort=False)
        .head(top_n_annotations)["candidate_key"]
        .unique()
    )
    plot_df["annotate"] = plot_df["candidate_key"].isin(top_keys)
    plot_df["scope_display"] = plot_df["candidate_scope"].map(display_scope_name)
    return plot_df


def filter_f1_plot_candidates(plot_df, top_per_scope=3):
    selected = []
    for scope in ["window", "standard"]:
        subset = plot_df[plot_df["candidate_scope"] == scope].sort_values(
            ["f1", "recall", "false_positive_rate", "precision"],
            ascending=[False, False, True, False],
            na_position="last",
        )
        selected.append(subset.head(top_per_scope))
    if not selected:
        return plot_df.copy()
    return (
        pd.concat(selected, ignore_index=True)
        .drop_duplicates(subset=["candidate_key", "task_name"])
        .reset_index(drop=True)
    )


def select_scatter_annotations(plot_df, winners_df):
    if plot_df.empty:
        return plot_df.copy()
    wanted_keys = set()
    for winner_type in ["overall", "injected", "phrase"]:
        subset = winners_df[winners_df["winner_type"] == winner_type]
        if not subset.empty:
            wanted_keys.add(subset.iloc[0]["candidate_key"])

    strongest_standard = (
        plot_df[plot_df["candidate_scope"] == "standard"]
        .sort_values(
            ["f1", "recall", "false_positive_rate", "precision"],
            ascending=[False, False, True, False],
            na_position="last",
        )
        .head(1)
    )
    if not strongest_standard.empty:
        wanted_keys.add(strongest_standard.iloc[0]["candidate_key"])

    annotated = plot_df[plot_df["candidate_key"].isin(wanted_keys)].copy()
    annotated = annotated.sort_values(
        ["candidate_key", "f1", "recall", "false_positive_rate", "precision"],
        ascending=[True, False, False, True, False],
        na_position="last",
    )
    return annotated.drop_duplicates(subset=["candidate_key"]).reset_index(drop=True)


def apply_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#D1D5DB",
        }
    )


def build_model_palette(values):
    palette = {}
    fallback_index = 0
    for value in values:
        if value in palette:
            continue
        if value in MODEL_COLORS:
            palette[value] = MODEL_COLORS[value]
        else:
            palette[value] = FALLBACK_MODEL_COLORS[fallback_index % len(FALLBACK_MODEL_COLORS)]
            fallback_index += 1
    return palette


def add_bar_value_labels(ax, decimals=3):
    for container in ax.containers:
        labels = []
        for patch in container:
            height = patch.get_height()
            if pd.isna(height):
                labels.append("")
            else:
                labels.append(f"{height:.{decimals}f}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)


def apply_figure_header(fig, title, subtitle, top=0.88):
    fig.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=18, fontweight="bold")
    if subtitle:
        fig.text(0.08, 0.945, subtitle, ha="left", va="top", fontsize=10, color="#4B5563")
    fig.subplots_adjust(top=top)


def save_plot(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_plots(candidate_summary, winners_df, overall_df, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    apply_plot_style()
    plot_df = build_plot_candidates(candidate_summary)
    if plot_df.empty:
        return generated

    overall_key = None
    if not overall_df.empty:
        overall_key = overall_df.iloc[0]["candidate_key"]
    plot_df["display_label"] = plot_df["candidate_display_label"]
    plot_df["axis_label"] = plot_df["candidate_axis_label"]
    plot_df["winner_marker"] = plot_df["candidate_key"].eq(overall_key)

    f1_plot_df = filter_f1_plot_candidates(plot_df, top_per_scope=3)
    fig, ax = plt.subplots(figsize=(12, 7))
    ordered_labels = (
        f1_plot_df.groupby("display_label")["f1"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    axis_label_map = (
        f1_plot_df.drop_duplicates("display_label")
        .set_index("display_label")["axis_label"]
        .to_dict()
    )
    sns.barplot(
        data=f1_plot_df,
        x="display_label",
        y="f1",
        hue="task_label",
        order=ordered_labels,
        palette=TASK_COLORS,
        ax=ax,
    )
    add_bar_value_labels(ax)
    apply_figure_header(
        fig,
        "Injected vs Phrase F1 for Final Candidates",
        "Top 3 window and top 3 standard candidates, ordered by average F1.",
        top=0.85,
    )
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    tick_positions = ax.get_xticks()
    ax.set_xticks(
        tick_positions,
        [axis_label_map[label] for label in ordered_labels],
        rotation=0,
    )
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, min(1.02, f1_plot_df["f1"].max() + 0.12))
    ax.legend(title="", frameon=False, loc="upper right")
    if overall_key is not None:
        winner_label = (
            f1_plot_df.loc[f1_plot_df["candidate_key"] == overall_key, "display_label"]
            .iloc[0]
        )
        winner_index = list(ordered_labels).index(winner_label)
        for patch in ax.patches:
            center = patch.get_x() + patch.get_width() / 2
            if abs(center - winner_index) < 0.6:
                patch.set_edgecolor("#111827")
                patch.set_linewidth(1.8)
        ax.text(
            winner_index,
            f1_plot_df.groupby("display_label")["f1"].mean().loc[winner_label] + 0.075,
            "winner",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#111827",
        )
    fig.subplots_adjust(bottom=0.18)
    path = plots_dir / "f1_comparison.png"
    save_plot(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=plot_df,
        x="precision",
        y="recall",
        hue="model_display",
        style="candidate_scope",
        palette=build_model_palette(plot_df["model_display"].unique()),
        s=120,
        ax=ax,
    )
    annotated = select_scatter_annotations(plot_df, winners_df)
    offsets = {"window": (8, 6), "standard": (8, -12)}
    for _, row in annotated.iterrows():
        dx, dy = offsets.get(row["candidate_scope"], (6, 6))
        ax.annotate(
            row["candidate_annotation_label"],
            (row["precision"], row["recall"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
    apply_figure_header(
        fig,
        "Precision vs Recall for Final Candidates",
        "Only the overall winner, task winners, and strongest standard baseline are annotated.",
        top=0.86,
    )
    ax.set_xlim(0.24, 0.78)
    ax.set_ylim(0.62, 1.01)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        split_index = labels.index("candidate_scope") if "candidate_scope" in labels else len(labels)
        model_handles = handles[1:split_index]
        model_labels = labels[1:split_index]
        if model_handles:
            legend1 = ax.legend(
                model_handles,
                model_labels,
                title="Model",
                frameon=False,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
            )
            ax.add_artist(legend1)
        if split_index < len(labels):
            ax.legend(
                handles[split_index + 1 :],
                labels[split_index + 1 :],
                title="Technique",
                frameon=False,
                bbox_to_anchor=(1.02, 0.72),
                loc="upper left",
            )
    path = plots_dir / "precision_recall_scatter.png"
    save_plot(fig, path)
    generated.append(path)

    winner_rows = []
    if not winners_df.empty:
        for winner_type in ["overall", "injected", "phrase"]:
            subset = winners_df[winners_df["winner_type"] == winner_type]
            if subset.empty:
                continue
            row = subset.iloc[0]
            winner_rows.append(
                {
                    "winner_type": winner_type.title(),
                    "f1": row["f1"],
                    "label": row.get("candidate_display_label", row.get("candidate_label")),
                    "detail": (
                        row.get("config_label")
                        if winner_type != "overall"
                        else f"I: {row.get('injected', 'standard')} | P: {row.get('phrase', 'standard')}"
                    ),
                }
            )
    winner_frame = pd.DataFrame(winner_rows)
    if not winner_frame.empty:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        sns.barplot(data=winner_frame, x="winner_type", y="f1", ax=ax, color="#2563EB")
        detail_lines = []
        for index, row in winner_frame.iterrows():
            ax.text(
                index,
                row["f1"] + 0.025,
                row["label"],
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            detail_lines.append(f"{row['winner_type']}: {compact_config_label(row['detail'])}")
        ax.set_ylim(0, min(1.02, winner_frame["f1"].max() + 0.18))
        apply_figure_header(
            fig,
            "Winning Technique Summary",
            "Exact configuration detail is moved below the chart to avoid label collisions.",
            top=0.84,
        )
        ax.set_xlabel("")
        ax.set_ylabel("F1")
        fig.text(
            0.08,
            0.03,
            "\n".join(detail_lines),
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#4B5563",
        )
        fig.subplots_adjust(bottom=0.23)
        path = plots_dir / "winner_summary.png"
        save_plot(fig, path)
        generated.append(path)

    return generated


def dataframe_to_markdown(frame):
    if frame.empty:
        return "_No data available._"

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        else:
            display[column] = display[column].fillna("").astype(str)

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def build_report_tables(candidate_summary, winners_df, overall_df):
    comparison_columns = [
        "task_name",
        "model_display",
        "candidate_display_label",
        "candidate_axis_label",
        "candidate_scope",
        "config_label",
        "f1",
        "recall",
        "precision",
        "false_positive_rate",
    ]
    comparison_table = candidate_summary[comparison_columns].copy() if not candidate_summary.empty else pd.DataFrame(columns=comparison_columns)

    overall_columns = [
        "candidate_display_label",
        "candidate_scope",
        "f1",
        "recall",
        "precision",
        "false_positive_rate",
        "injected",
        "phrase",
        "tasks_covered",
        "rank",
    ]
    overall_table = overall_df[overall_columns].copy() if not overall_df.empty else pd.DataFrame(columns=overall_columns)

    return comparison_table, overall_table


def build_winner_line(winners_df, winner_type):
    winner = winners_df[winners_df["winner_type"] == winner_type]
    if winner.empty:
        return f"No {winner_type} winner available."
    row = winner.iloc[0]
    parts = [
        f"{row.get('candidate_display_label', row.get('candidate_label'))}",
        f"F1={row['f1']:.4f}",
        f"Recall={row['recall']:.4f}",
        f"Precision={row['precision']:.4f}",
        f"FPR={row['false_positive_rate']:.4f}",
    ]
    if winner_type == "overall":
        injected_config = row.get("injected")
        phrase_config = row.get("phrase")
        if pd.notna(injected_config) and injected_config != "standard":
            parts.append(f"Injected config: {injected_config}")
        if pd.notna(phrase_config) and phrase_config != "standard":
            parts.append(f"Phrase config: {phrase_config}")
    else:
        config_label = row.get("config_label")
        if pd.notna(config_label) and config_label != "standard":
            parts.append(f"Config: {config_label}")
    return "; ".join(parts)


def write_markdown_report(output_path, run_name, metadata, winners_df, comparison_table, overall_table, pure_df, diagnostics_df, plots):
    models = ", ".join(display_model_name(model["name"]) for model in metadata.get("models", []))
    lines = [
        "# Foreign-Word Detection Analysis",
        "",
        f"Run: `{run_name}`",
        "",
        "## Conclusion",
        f"- Best overall: {build_winner_line(winners_df, 'overall')}",
        f"- Best injected: {build_winner_line(winners_df, 'injected')}",
        f"- Best phrase: {build_winner_line(winners_df, 'phrase')}",
        "",
        "## Run Setup",
        f"- Models: {models}",
        f"- Window modes: {', '.join(metadata.get('window_decision_modes') or []) or 'n/a'}",
        "",
        "## Final Candidate Comparison",
        dataframe_to_markdown(comparison_table),
        "",
        "## Overall Ranking Across Injected + Phrase",
        dataframe_to_markdown(overall_table),
        "",
        "## Interpretation",
    ]

    overall = winners_df[winners_df["winner_type"] == "overall"]
    if not overall.empty and overall.iloc[0]["candidate_scope"] == "window":
        lines.append("- Window-based detection beat the plain per-token methods overall on the tasks that matter most here.")
    else:
        lines.append("- The plain per-token method remained best overall once injected and phrase performance were averaged.")
    lines.append("- Pure-language metrics are reported only as background and do not affect the winner selection.")
    lines.append("")

    if not pure_df.empty:
        pure_table = (
            pure_df.groupby(["scope", "model"], dropna=False)["foreign_false_positive_rate"]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("foreign_false_positive_rate", ascending=True, na_position="last")
        )
        pure_table["model"] = pure_table["model"].map(display_model_name)
        lines.append("## Pure-Language Appendix")
        lines.append(dataframe_to_markdown(pure_table))
        lines.append("")

    lines.append("## Plots")
    for plot_path in plots:
        lines.append(f"- `{plot_path}`")
    lines.append("")

    lines.append("## Diagnostics")
    lines.append(dataframe_to_markdown(diagnostics_df))
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(output_path, run_name, winners_df, comparison_table, overall_table, pure_df, diagnostics_df, plots):
    parts = [
        "<html><head><meta charset='utf-8'><title>Foreign-Word Detection Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;}table{border-collapse:collapse;width:100%;margin:16px 0;}th,td{border:1px solid #ccc;padding:8px;text-align:left;}th{background:#f3f3f3;}img{max-width:100%;margin:18px 0;border:1px solid #ddd;}ul{line-height:1.6}</style>",
        "</head><body>",
        f"<h1>Foreign-Word Detection Analysis</h1><p>Run: <code>{html.escape(run_name)}</code></p>",
        "<h2>Conclusion</h2><ul>",
        f"<li><strong>Best overall:</strong> {html.escape(build_winner_line(winners_df, 'overall'))}</li>",
        f"<li><strong>Best injected:</strong> {html.escape(build_winner_line(winners_df, 'injected'))}</li>",
        f"<li><strong>Best phrase:</strong> {html.escape(build_winner_line(winners_df, 'phrase'))}</li>",
        "</ul>",
        "<h2>Final Candidate Comparison</h2>",
        comparison_table.to_html(index=False, escape=True),
        "<h2>Overall Ranking Across Injected + Phrase</h2>",
        overall_table.to_html(index=False, escape=True),
    ]

    if not pure_df.empty:
        pure_table = (
            pure_df.groupby(["scope", "model"], dropna=False)["foreign_false_positive_rate"]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("foreign_false_positive_rate", ascending=True, na_position="last")
        )
        pure_table["model"] = pure_table["model"].map(display_model_name)
        parts.extend(
            [
                "<h2>Pure-Language Appendix</h2>",
                pure_table.to_html(index=False, escape=True),
            ]
        )

    parts.append("<h2>Plots</h2>")
    for plot_path in plots:
        rel_path = os.path.relpath(plot_path, output_path.parent)
        parts.append(f"<h3>{html.escape(plot_path.name)}</h3>")
        parts.append(f"<img src='{html.escape(rel_path)}' alt='{html.escape(plot_path.name)}'>")

    parts.extend(
        [
            "<h2>Diagnostics</h2>",
            diagnostics_df.to_html(index=False, escape=True),
            "</body></html>",
        ]
    )
    output_path.write_text("".join(parts), encoding="utf-8")


def write_csv_output(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def analyze_run(run_dir, output_dir=None, skip_html=False):
    loaded = load_run_data(run_dir)
    run_dir = loaded["run_dir"]
    output_dir = Path(output_dir) if output_dir else run_dir / "analysis_methods"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary = build_candidate_summary(
        loaded["standard_df"], loaded["window_df"]
    )
    winners_df, overall_df = compute_winners(candidate_summary)
    plots = create_plots(
        candidate_summary, winners_df, overall_df, output_dir / PLOTS_SUBDIR
    )
    comparison_table, overall_table = build_report_tables(
        candidate_summary, winners_df, overall_df
    )

    best_techniques = winners_df.copy()
    write_csv_output(output_dir / "candidate_summary.csv", candidate_summary)
    write_csv_output(output_dir / "best_techniques.csv", best_techniques)

    write_markdown_report(
        output_dir / "report.md",
        run_dir.name,
        loaded["metadata"],
        winners_df,
        comparison_table,
        overall_table,
        loaded["pure_df"],
        loaded["diagnostics_df"],
        plots,
    )
    if not skip_html:
        write_html_report(
            output_dir / "report.html",
            run_dir.name,
            winners_df,
            comparison_table,
            overall_table,
            loaded["pure_df"],
            loaded["diagnostics_df"],
            plots,
        )

    return {
        "candidate_summary": candidate_summary,
        "winners_df": winners_df,
        "overall_df": overall_df,
        "plots": plots,
        "output_dir": output_dir,
    }


def main():
    args = parse_args()
    result = analyze_run(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        skip_html=args.skip_html,
    )
    print(f"Analyzed run {Path(args.run_dir).name} into {result['output_dir']}")
    print(f"Wrote candidate_summary.csv with {len(result['candidate_summary'])} rows")
    print(f"Wrote best_techniques.csv with {len(result['winners_df'])} rows")
    print(f"Generated {len(result['plots'])} plots")


if __name__ == "__main__":
    main()
