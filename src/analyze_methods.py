import argparse
import csv
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


DEFAULT_RESULTS_ROOT = Path("evaluation_results")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "analysis_methods"
PLOTS_SUBDIR = "plots"
TABLES_SUBDIR = "tables"

MODEL_DISPLAY_NAMES = {
    "facebook-fasttext-language-identification": "fasttext",
    "glotlid": "glotlid",
    "lingua": "lingua",
    "lingua-spanish-only": "lingua-es",
    "spanish-binary-baseline": "binary-es",
}

TASK_GROUP_ORDER = ["injected", "phrase", "pure"]
TASK_GROUP_LABELS = {
    "injected": "Injected",
    "phrase": "Phrase",
    "pure": "Pure",
}
TECHNIQUE_TONES = {
    "standard": "#4A4A4A",
    "legacy_window": "#8C8C8C",
    "contextual_hybrid": "#B8B8B8",
    "llm": "#D7D7D7",
}
TECHNIQUE_HATCHES = {
    "standard": "",
    "legacy_window": "///",
    "contextual_hybrid": "\\\\\\",
    "llm": "...",
}
STAGE_METRICS = [
    "accuracy",
    "precision",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
]
FILENAME_TO_TASK = {
    "pure_foreign_detection_metrics.csv": ("pure", "standard"),
    "injected_detection_metrics.csv": ("injected", "standard"),
    "phrase_detection_metrics.csv": ("phrase", "standard"),
    "pure_window_detection_metrics.csv": ("pure", "window"),
    "injected_window_detection_metrics.csv": ("injected", "window"),
    "phrase_window_detection_metrics.csv": ("phrase", "window"),
}
NUMERIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "foreign_false_positive_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "total",
    "window_size",
    "window_foreign_threshold",
    "window_shared_foreign_threshold",
    "chunk_size",
]
AGG_COUNT_COLUMNS = ["tp", "fp", "tn", "fn", "total"]
WINDOW_FIELDS = [
    "window_decision_rule",
    "window_size",
    "window_foreign_threshold",
    "window_shared_foreign_threshold",
]
SLICE_COLUMNS = ["injected_lang", "true_lang"]
REQUIRED_COLUMNS = [
    "run_name",
    "run_family",
    "task_name",
    "task_group",
    "technique_group",
    "technique_detail",
    "technique_label",
    "model",
    "model_display",
    "model_family",
    "candidate_key",
    "candidate_label",
    "configuration_label",
    "source_file",
    "aggregation_mode",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create staged visualization outputs from classical and LLM "
            "evaluation result directories."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory that contains evaluation run subdirectories.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit run directories to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots and tables.",
    )
    parser.add_argument(
        "--include-tasks",
        default="injected,phrase",
        help="Comma-separated task groups to include.",
    )
    parser.add_argument(
        "--include-run-families",
        default="classical,llm",
        help="Comma-separated run families to include.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip chart generation.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip CSV and table image generation.",
    )
    return parser.parse_args()


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def display_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, str(model_name))


def format_number(value):
    if value is None or pd.isna(value):
        return "na"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def compact_text(value, max_length=40):
    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def display_technique_name(kind, compact=False):
    mapping = {
        "standard": "standard",
        "legacy_window": "window",
        "contextual_hybrid": "context" if compact else "context window",
        "llm": "llm",
    }
    return mapping.get(kind, str(kind).replace("_", " "))


def build_window_config_label(row):
    parts = [str(row.get("window_decision_rule") or "window")]
    if pd.notna(row.get("window_size")):
        parts.append(f"w={format_number(row.get('window_size'))}")
    if pd.notna(row.get("window_foreign_threshold")):
        parts.append(f"t={format_number(row.get('window_foreign_threshold'))}")
    shared = row.get("window_shared_foreign_threshold")
    if pd.notna(shared):
        parts.append(f"s={format_number(shared)}")
    return " ".join(parts)


def infer_run_family(run_dir, metadata):
    if "window_decision_modes" in metadata:
        return "classical"
    if "prompt" in metadata:
        return "llm"
    if "llm" in run_dir.name.lower():
        return "llm"
    return "classical"


def discover_run_dirs(results_root):
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    run_dirs = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == DEFAULT_OUTPUT_DIR.name:
            continue
        if (child / "run_metadata.json").exists():
            run_dirs.append(child)
            continue
        if any((child / filename).exists() for filename in FILENAME_TO_TASK):
            run_dirs.append(child)
    return run_dirs


def read_metric_csv(run_dir, filename, metadata, run_family):
    path = Path(run_dir) / filename
    task_group, default_technique_group = FILENAME_TO_TASK[filename]
    diagnostic = {
        "run_name": Path(run_dir).name,
        "run_family": run_family,
        "artifact": filename,
        "task_group": task_group,
        "status": "missing",
        "rows": 0,
    }

    if not path.exists():
        return pd.DataFrame(), diagnostic
    if path.stat().st_size == 0:
        diagnostic["status"] = "empty"
        return pd.DataFrame(), diagnostic

    frame = pd.read_csv(path)
    diagnostic["status"] = "loaded"
    diagnostic["rows"] = len(frame)

    if frame.empty:
        diagnostic["status"] = "empty"
        return pd.DataFrame(), diagnostic

    frame["run_name"] = Path(run_dir).name
    frame["run_family"] = run_family
    frame["task_name"] = path.stem.replace("_metrics", "")
    frame["task_group"] = task_group
    frame["source_file"] = filename
    frame["model_display"] = frame["model"].map(display_model_name)
    frame["model_family"] = frame.get("model_family", pd.Series(["unknown"] * len(frame)))

    for field in WINDOW_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
    for field in SLICE_COLUMNS + ["llm_mode", "chunk_size"]:
        if field not in frame.columns:
            frame[field] = pd.NA

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "llm_mode" in frame.columns and frame["llm_mode"].notna().any():
        frame["technique_group"] = "llm"
        frame["technique_detail"] = frame["llm_mode"].fillna("llm")
        chunk_mask = frame["chunk_size"].notna()
        frame.loc[chunk_mask, "technique_detail"] = (
            frame.loc[chunk_mask, "technique_detail"].astype("string").fillna("llm")
            + " chunk="
            + frame.loc[chunk_mask, "chunk_size"].map(format_number).astype("string")
        )
    elif default_technique_group == "window":
        frame["technique_group"] = "window"
        frame["technique_detail"] = frame.apply(build_window_config_label, axis=1)
    else:
        frame["technique_group"] = "standard"
        frame["technique_detail"] = "standard"

    frame["technique_label"] = frame["technique_group"] + ": " + frame["technique_detail"].astype(str)
    frame["configuration_label"] = frame["technique_detail"].astype(str)
    frame["candidate_key"] = (
        frame["model"].astype(str)
        + "||"
        + frame["run_family"].astype(str)
        + "||"
        + frame["technique_group"].astype(str)
        + "||"
        + frame["technique_detail"].astype(str)
    )
    frame["candidate_label"] = (
        frame["model_display"].astype(str) + " | " + frame["technique_detail"].astype(str)
    )
    frame["aggregation_mode"] = "raw"
    return frame, diagnostic


def load_run_data(run_dir):
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    run_family = infer_run_family(run_dir, metadata)

    frames = []
    diagnostics = []
    for filename in FILENAME_TO_TASK:
        frame, diagnostic = read_metric_csv(run_dir, filename, metadata, run_family)
        diagnostics.append(diagnostic)
        if not frame.empty:
            frames.append(frame)

    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "run_family": run_family,
        "metrics": pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(),
        "diagnostics": pd.DataFrame(diagnostics),
    }


def normalize_columns(frame):
    if frame.empty:
        columns = REQUIRED_COLUMNS + NUMERIC_COLUMNS + WINDOW_FIELDS + SLICE_COLUMNS + ["llm_mode"]
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))

    normalized = frame.copy()
    for column in REQUIRED_COLUMNS + WINDOW_FIELDS + SLICE_COLUMNS + ["llm_mode"]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for column in NUMERIC_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def combine_runs(run_dirs):
    run_payloads = [load_run_data(run_dir) for run_dir in run_dirs]
    metrics = normalize_columns(
        pd.concat(
            [payload["metrics"] for payload in run_payloads if not payload["metrics"].empty],
            ignore_index=True,
            sort=False,
        )
        if run_payloads
        else pd.DataFrame()
    )
    diagnostics = (
        pd.concat([payload["diagnostics"] for payload in run_payloads], ignore_index=True, sort=False)
        if run_payloads
        else pd.DataFrame(
            columns=["run_name", "run_family", "artifact", "task_group", "status", "rows"]
        )
    )
    return metrics, diagnostics


def apply_filters(frame, include_tasks, include_run_families):
    if frame.empty:
        return frame.copy()
    filtered = frame[frame["task_group"].isin(include_tasks)].copy()
    filtered = filtered[filtered["run_family"].isin(include_run_families)].copy()
    return filtered.reset_index(drop=True)


def safe_sum(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.sum(min_count=1)


def first_present(series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def join_unique(series):
    values = sorted({str(value) for value in series.dropna() if str(value)})
    return ", ".join(values)


def compute_rate(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return pd.NA
    return numerator / denominator


def aggregate_candidate_metrics(raw_metrics):
    if raw_metrics.empty:
        return pd.DataFrame()

    group_columns = [
        "candidate_key",
        "candidate_label",
        "model",
        "model_display",
        "model_family",
        "run_family",
        "technique_group",
        "technique_detail",
        "technique_label",
        "task_group",
        "window_decision_rule",
        "window_size",
        "window_foreign_threshold",
        "window_shared_foreign_threshold",
        "llm_mode",
        "chunk_size",
    ]
    aggregated_rows = []
    for keys, group in raw_metrics.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys))
        row["run_names"] = join_unique(group["run_name"])
        row["configuration_label"] = first_present(group["configuration_label"])
        row["source_files"] = join_unique(group["source_file"])
        row["slice_count"] = len(group)
        row["slice_labels"] = join_unique(
            group["injected_lang"].fillna(group["true_lang"]).astype("string")
        )

        count_values = {
            column: safe_sum(group[column]) if column in group.columns else pd.NA
            for column in AGG_COUNT_COLUMNS
        }
        row.update(count_values)

        has_confusion_counts = all(pd.notna(row[column]) for column in ["tp", "fp", "tn", "fn"])
        if has_confusion_counts:
            total = row["tp"] + row["fp"] + row["tn"] + row["fn"]
            row["total"] = total
            row["accuracy"] = compute_rate(row["tp"] + row["tn"], total)
            row["precision"] = compute_rate(row["tp"], row["tp"] + row["fp"])
            row["recall"] = compute_rate(row["tp"], row["tp"] + row["fn"])
            precision = row["precision"]
            recall = row["recall"]
            row["f1"] = (
                pd.NA
                if pd.isna(precision) or pd.isna(recall) or precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
            row["false_positive_rate"] = compute_rate(row["fp"], row["fp"] + row["tn"])
            row["false_negative_rate"] = compute_rate(row["fn"], row["fn"] + row["tp"])
            row["aggregation_mode"] = "confusion_sum"
        else:
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "false_negative_rate",
                "foreign_false_positive_rate",
            ]:
                row[metric] = (
                    pd.to_numeric(group[metric], errors="coerce").mean()
                    if metric in group.columns
                    else pd.NA
                )
            row["aggregation_mode"] = "mean"

        aggregated_rows.append(row)

    aggregated = pd.DataFrame(aggregated_rows)
    return aggregated.sort_values(
        ["task_group", "f1", "accuracy", "precision"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_per_language_metrics(raw_metrics):
    if raw_metrics.empty:
        return pd.DataFrame()
    detail = raw_metrics.copy()
    detail["slice_label"] = detail["injected_lang"].fillna(detail["true_lang"])
    detail = detail[detail["slice_label"].notna()].copy()
    if detail.empty:
        return pd.DataFrame()
    columns = [
        "run_name",
        "run_family",
        "task_group",
        "slice_label",
        "model",
        "model_display",
        "technique_group",
        "technique_detail",
        "window_decision_rule",
        "candidate_label",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "total",
    ]
    return detail[columns].sort_values(
        ["task_group", "slice_label", "f1", "accuracy"],
        ascending=[True, True, False, False],
        na_position="last",
    )


def apply_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#D9D9D9",
            "axes.edgecolor": "#BDBDBD",
            "figure.dpi": 200,
        }
    )


def format_metric_label(metric_name):
    return metric_name.replace("_", " ").title()


def ranking_sort_values(frame):
    return frame.sort_values(
        ["f1", "accuracy", "precision", "false_positive_rate", "false_negative_rate"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )


def choose_best_row(frame):
    if frame.empty:
        return frame.copy()
    return ranking_sort_values(frame).head(1).reset_index(drop=True)


def choose_best_per_model(frame):
    if frame.empty:
        return frame.copy()
    ranked = ranking_sort_values(frame)
    return (
        ranked.groupby("model_display", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def make_setting_label(row):
    parts = [f"w={format_number(row.get('window_size'))}", f"t={format_number(row.get('window_foreign_threshold'))}"]
    if row.get("window_decision_rule") == "contextual_hybrid" and pd.notna(
        row.get("window_shared_foreign_threshold")
    ):
        parts.append(f"s={format_number(row.get('window_shared_foreign_threshold'))}")
    return f"{row['model_display']} | {' '.join(parts)}"


def make_stage_candidate_label(row, stage_name):
    if stage_name == "standard":
        return row["model_display"]
    if stage_name == "legacy_best":
        return f"{row['model_display']} | best window"
    if stage_name == "contextual_best":
        return f"{row['model_display']} | best context"
    if stage_name == "final_standard":
        return f"standard: {row['model_display']}"
    if stage_name == "final_legacy":
        return f"window: {row['model_display']}"
    if stage_name == "final_contextual":
        return f"context: {row['model_display']}"
    if stage_name == "final_llm":
        return f"llm: {row['model_display']}"
    return row["candidate_label"]


def build_task_stage_frames(aggregated_metrics, task_group):
    task_frame = aggregated_metrics[aggregated_metrics["task_group"] == task_group].copy()
    classical = task_frame[task_frame["run_family"] == "classical"].copy()
    llm = task_frame[task_frame["run_family"] == "llm"].copy()

    standard_all = classical[classical["technique_group"] == "standard"].copy()
    standard_all = ranking_sort_values(standard_all).reset_index(drop=True)
    if not standard_all.empty:
        standard_all["stage_label"] = standard_all.apply(
            lambda row: make_stage_candidate_label(row, "standard"), axis=1
        )
    standard_best = choose_best_row(standard_all)

    legacy_all = classical[
        (classical["technique_group"] == "window")
        & (classical["window_decision_rule"] == "legacy_window")
    ].copy()
    legacy_all = ranking_sort_values(legacy_all).reset_index(drop=True)
    if not legacy_all.empty:
        legacy_all["stage_label"] = legacy_all.apply(make_setting_label, axis=1)
    legacy_best_per_model = choose_best_per_model(legacy_all)
    if not legacy_best_per_model.empty:
        legacy_best_per_model["stage_label"] = legacy_best_per_model.apply(
            lambda row: make_stage_candidate_label(row, "legacy_best"), axis=1
        )
    legacy_best_overall = choose_best_row(legacy_best_per_model)

    contextual_all = classical[
        (classical["technique_group"] == "window")
        & (classical["window_decision_rule"] == "contextual_hybrid")
    ].copy()
    contextual_all = ranking_sort_values(contextual_all).reset_index(drop=True)
    if not contextual_all.empty:
        contextual_all["stage_label"] = contextual_all.apply(make_setting_label, axis=1)
    contextual_best_per_model = choose_best_per_model(contextual_all)
    if not contextual_best_per_model.empty:
        contextual_best_per_model["stage_label"] = contextual_best_per_model.apply(
            lambda row: make_stage_candidate_label(row, "contextual_best"), axis=1
        )
    contextual_best_overall = choose_best_row(contextual_best_per_model)

    llm_all = ranking_sort_values(llm).reset_index(drop=True)
    if not llm_all.empty:
        llm_all["stage_label"] = llm_all["model_display"].map(lambda value: f"llm: {value}")

    final_frames = []
    if not standard_best.empty:
        final_standard = standard_best.copy()
        final_standard["stage_label"] = final_standard.apply(
            lambda row: make_stage_candidate_label(row, "final_standard"), axis=1
        )
        final_frames.append(final_standard)
    if not legacy_best_overall.empty:
        final_legacy = legacy_best_overall.copy()
        final_legacy["stage_label"] = final_legacy.apply(
            lambda row: make_stage_candidate_label(row, "final_legacy"), axis=1
        )
        final_frames.append(final_legacy)
    if not contextual_best_overall.empty:
        final_context = contextual_best_overall.copy()
        final_context["stage_label"] = final_context.apply(
            lambda row: make_stage_candidate_label(row, "final_contextual"), axis=1
        )
        final_frames.append(final_context)
    if not llm_all.empty:
        final_llm = llm_all.copy()
        final_llm["stage_label"] = final_llm.apply(
            lambda row: make_stage_candidate_label(row, "final_llm"), axis=1
        )
        final_frames.append(final_llm)

    final_curated = (
        pd.concat(final_frames, ignore_index=True, sort=False) if final_frames else pd.DataFrame()
    )
    if not final_curated.empty:
        final_curated = ranking_sort_values(final_curated).reset_index(drop=True)

    return {
        "standard_all_models": standard_all,
        "standard_best_model": standard_best,
        "legacy_all_settings": legacy_all,
        "legacy_best_per_model": legacy_best_per_model,
        "legacy_best_overall": legacy_best_overall,
        "contextual_all_settings": contextual_all,
        "contextual_best_per_model": contextual_best_per_model,
        "contextual_best_overall": contextual_best_overall,
        "llm_all_models": llm_all,
        "final_curated": final_curated,
    }


def save_plot(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def style_bar_containers(ax, hue_order):
    containers = ax.containers
    for container_index, container in enumerate(containers):
        key = hue_order[container_index] if container_index < len(hue_order) else None
        hatch = TECHNIQUE_HATCHES.get(key, "")
        for patch in container.patches:
            patch.set_hatch(hatch)
            patch.set_edgecolor("#111111")
            patch.set_linewidth(0.9)


def create_bar_chart(frame, metric_name, output_path, title, subtitle, label_column="stage_label", hue_column=None, hue_order=None):
    plot_frame = frame[frame[metric_name].notna()].copy()
    if plot_frame.empty:
        return None

    ascending = metric_name in {"false_positive_rate", "false_negative_rate"}
    plot_frame = plot_frame.sort_values(
        [metric_name, "f1", "accuracy", "precision"],
        ascending=[ascending, False, False, False],
        na_position="last",
    )
    plot_frame["plot_label"] = plot_frame[label_column].map(lambda value: compact_text(value, 34))

    fig_height = max(4.8, 0.55 * len(plot_frame) + 1.8)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    if hue_column:
        sns.barplot(
            data=plot_frame,
            y="plot_label",
            x=metric_name,
            hue=hue_column,
            order=plot_frame["plot_label"],
            hue_order=hue_order,
            palette=[TECHNIQUE_TONES.get(key, "#999999") for key in hue_order],
            dodge=False,
            ax=ax,
        )
        style_bar_containers(ax, hue_order)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Technique")
            legend.set_frame_on(False)
    else:
        sns.barplot(
            data=plot_frame,
            y="plot_label",
            x=metric_name,
            order=plot_frame["plot_label"],
            color="#6E6E6E",
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_edgecolor("#111111")
            patch.set_linewidth(0.9)

    for container in ax.containers:
        labels = []
        for patch in container.patches:
            width = patch.get_width()
            labels.append("" if pd.isna(width) else f"{width:.3f}")
        ax.bar_label(container, labels=labels, padding=4, fontsize=8)

    ax.set_title(title, loc="left")
    ax.set_xlabel(format_metric_label(metric_name))
    ax.set_ylabel("")
    if metric_name in {"false_positive_rate", "false_negative_rate"}:
        ax.set_xlim(0, max(0.05, min(1.0, ax.get_xlim()[1])))
    else:
        ax.set_xlim(0, 1.0)
    fig.text(0.125, 0.94, subtitle, fontsize=9.5, color="#4B5563")
    save_plot(fig, output_path)
    return output_path


def create_scatter_plot(frame, x_metric, y_metric, output_path, title, subtitle, style_by="technique_kind"):
    plot_frame = frame[frame[x_metric].notna() & frame[y_metric].notna()].copy()
    if plot_frame.empty:
        return None

    marker_map = {
        "standard": "o",
        "legacy_window": "s",
        "contextual_hybrid": "D",
        "llm": "^",
    }
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    categories = [value for value in plot_frame[style_by].dropna().unique()]
    for category in categories:
        subset = plot_frame[plot_frame[style_by] == category]
        sns.scatterplot(
            data=subset,
            x=x_metric,
            y=y_metric,
            marker=marker_map.get(category, "o"),
            s=150,
            color=TECHNIQUE_TONES.get(category, "#808080"),
            edgecolor="#111111",
            linewidth=0.8,
            legend=False,
            ax=ax,
        )

    for _, row in ranking_sort_values(plot_frame).head(min(8, len(plot_frame))).iterrows():
        ax.annotate(
            compact_text(row["stage_label"], 24),
            (row[x_metric], row[y_metric]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.95,
            },
        )

    ax.set_title(title, loc="left")
    ax.set_xlabel(format_metric_label(x_metric))
    ax.set_ylabel(format_metric_label(y_metric))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[key],
            color=TECHNIQUE_TONES.get(key, "#808080"),
            linestyle="None",
            markeredgecolor="#111111",
            markersize=8,
            label=display_technique_name(key, compact=False),
        )
        for key in categories
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Series", frameon=False, loc="best")
    fig.text(0.125, 0.94, subtitle, fontsize=9.5, color="#4B5563")
    save_plot(fig, output_path)
    return output_path


def render_table_image(frame, output_path, title, max_rows=18):
    if frame.empty:
        return None

    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
        else:
            display[column] = display[column].fillna("").astype(str).map(
                lambda value: compact_text(value, 26)
            )

    n_rows, n_cols = display.shape
    fig_height = max(2.8, 0.48 * (n_rows + 2))
    fig_width = min(18, max(10, 1.5 * n_cols))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        if row_index == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F8F8F8")
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=16, fontweight="bold")
    save_plot(fig, output_path)
    return output_path


def write_csv_output(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def make_technique_kind(frame, default_kind):
    if frame.empty:
        return frame.copy()
    enriched = frame.copy()
    enriched["technique_kind"] = default_kind
    return enriched


def frame_for_final_plot(frame):
    if frame.empty:
        return frame.copy()
    enriched = frame.copy()
    enriched["technique_kind"] = "standard"
    enriched.loc[enriched["stage_label"].str.startswith("legacy:"), "technique_kind"] = "legacy_window"
    enriched.loc[enriched["stage_label"].str.startswith("context:"), "technique_kind"] = "contextual_hybrid"
    enriched.loc[enriched["stage_label"].str.startswith("llm:"), "technique_kind"] = "llm"
    return enriched


def generate_task_outputs(task_group, stage_frames, plots_root, tables_root, generated_plots, generated_tables, skip_plots, skip_tables):
    task_label = TASK_GROUP_LABELS.get(task_group, task_group.title())
    plot_task_dir = plots_root / task_group
    table_task_dir = tables_root / task_group

    csv_exports = {
        f"standard_all_models_{task_group}.csv": stage_frames["standard_all_models"],
        f"standard_best_model_{task_group}.csv": stage_frames["standard_best_model"],
        f"legacy_all_settings_{task_group}.csv": stage_frames["legacy_all_settings"],
        f"legacy_best_per_model_{task_group}.csv": stage_frames["legacy_best_per_model"],
        f"legacy_best_overall_{task_group}.csv": stage_frames["legacy_best_overall"],
        f"contextual_all_settings_{task_group}.csv": stage_frames["contextual_all_settings"],
        f"contextual_best_per_model_{task_group}.csv": stage_frames["contextual_best_per_model"],
        f"contextual_best_overall_{task_group}.csv": stage_frames["contextual_best_overall"],
        f"llm_all_models_{task_group}.csv": stage_frames["llm_all_models"],
        f"final_curated_{task_group}.csv": stage_frames["final_curated"],
    }

    if not skip_tables:
        for filename, frame in csv_exports.items():
            write_csv_output(table_task_dir / filename, frame)

        table_specs = [
            (
                stage_frames["standard_all_models"][
                    ["model_display", "accuracy", "precision", "f1", "false_positive_rate", "false_negative_rate"]
                ]
                if not stage_frames["standard_all_models"].empty
                else pd.DataFrame(),
                table_task_dir / f"standard_ranking_{task_group}.png",
                f"{task_label} Standard Technique Ranking",
            ),
            (
                stage_frames["legacy_best_per_model"][
                    ["model_display", "technique_detail", "accuracy", "precision", "f1", "false_positive_rate", "false_negative_rate"]
                ]
                if not stage_frames["legacy_best_per_model"].empty
                else pd.DataFrame(),
                table_task_dir / f"legacy_best_per_model_{task_group}.png",
                f"{task_label} Best Window per Model",
            ),
            (
                stage_frames["contextual_best_per_model"][
                    ["model_display", "technique_detail", "accuracy", "precision", "f1", "false_positive_rate", "false_negative_rate"]
                ]
                if not stage_frames["contextual_best_per_model"].empty
                else pd.DataFrame(),
                table_task_dir / f"contextual_best_per_model_{task_group}.png",
                f"{task_label} Best Context Window per Model",
            ),
            (
                stage_frames["final_curated"][
                    ["stage_label", "technique_detail", "accuracy", "precision", "f1", "false_positive_rate", "false_negative_rate"]
                ]
                if not stage_frames["final_curated"].empty
                else pd.DataFrame(),
                table_task_dir / f"final_curated_{task_group}.png",
                f"{task_label} Final Curated Comparison",
            ),
        ]
        for frame, path, title in table_specs:
            generated = render_table_image(frame, path, title)
            if generated is not None:
                generated_tables.append(generated)

    if skip_plots:
        return

    standard_frame = make_technique_kind(stage_frames["standard_all_models"], "standard")
    legacy_all_frame = make_technique_kind(stage_frames["legacy_all_settings"], "legacy_window")
    legacy_best_frame = make_technique_kind(stage_frames["legacy_best_per_model"], "legacy_window")
    contextual_all_frame = make_technique_kind(stage_frames["contextual_all_settings"], "contextual_hybrid")
    contextual_best_frame = make_technique_kind(stage_frames["contextual_best_per_model"], "contextual_hybrid")
    final_frame = frame_for_final_plot(stage_frames["final_curated"])

    metric_groups = [
        ("standard", standard_frame, "All classical models with the standard technique."),
        ("legacy_all", legacy_all_frame, "Every model and window setting combination."),
        ("legacy_best", legacy_best_frame, "Best window setting selected for each model."),
        ("contextual_all", contextual_all_frame, "Every model and context window setting combination."),
        ("contextual_best", contextual_best_frame, "Best context window setting selected for each model."),
        ("final", final_frame, "Best standard, best window, best context, plus task LLM results if available."),
    ]

    for stage_name, frame, subtitle in metric_groups:
        stage_dir = plot_task_dir / stage_name
        if frame.empty:
            continue
        hue_column = None
        hue_order = None
        if stage_name == "final":
            hue_column = "technique_kind"
            hue_order = ["standard", "legacy_window", "contextual_hybrid", "llm"]
        stage_title_name = "standard"
        if stage_name.startswith("legacy"):
            stage_title_name = "window"
        elif stage_name.startswith("contextual"):
            stage_title_name = "context window"
        elif stage_name == "final":
            stage_title_name = "final comparison"
        for metric_name in STAGE_METRICS:
            generated = create_bar_chart(
                frame=frame,
                metric_name=metric_name,
                output_path=stage_dir / f"{metric_name}.png",
                title=f"{task_label}: {stage_title_name.title()} {format_metric_label(metric_name)}",
                subtitle=subtitle,
                hue_column=hue_column,
                hue_order=hue_order,
            )
            if generated is not None:
                generated_plots.append(generated)

    scatter_specs = [
        (
            standard_frame,
            plot_task_dir / "standard" / "precision_vs_f1.png",
            f"{task_label}: Standard Precision vs F1",
            "Standard technique only.",
        ),
        (
            standard_frame,
            plot_task_dir / "standard" / "false_positive_vs_false_negative.png",
            f"{task_label}: Standard Error Tradeoff",
            "Standard technique only.",
        ),
        (
            legacy_best_frame,
            plot_task_dir / "legacy_best" / "precision_vs_f1.png",
            f"{task_label}: Best Window Precision vs F1",
            "Best window setting chosen for each model.",
        ),
        (
            contextual_best_frame,
            plot_task_dir / "contextual_best" / "precision_vs_f1.png",
            f"{task_label}: Best Context Precision vs F1",
            "Best context window setting chosen for each model.",
        ),
        (
            final_frame,
            plot_task_dir / "final" / "precision_vs_f1.png",
            f"{task_label}: Final Curated Precision vs F1",
            "Final comparison contenders for this task.",
        ),
    ]
    for frame, path, title, subtitle in scatter_specs:
        if frame.empty:
            continue
        if "false_positive_vs_false_negative" in str(path):
            generated = create_scatter_plot(
                frame,
                "false_positive_rate",
                "false_negative_rate",
                path,
                title,
                subtitle,
            )
        else:
            generated = create_scatter_plot(
                frame,
                "precision",
                "f1",
                path,
                title,
                subtitle,
            )
        if generated is not None:
            generated_plots.append(generated)


def create_outputs(raw_metrics, diagnostics, output_dir, skip_plots=False, skip_tables=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / PLOTS_SUBDIR
    tables_dir = output_dir / TABLES_SUBDIR

    aggregated = aggregate_candidate_metrics(raw_metrics)
    per_language = build_per_language_metrics(raw_metrics)
    generated_plots = []
    generated_tables = []

    if not skip_tables:
        write_csv_output(output_dir / "combined_metrics_raw.csv", raw_metrics)
        write_csv_output(output_dir / "aggregated_candidate_metrics.csv", aggregated)
        write_csv_output(output_dir / "per_language_metrics.csv", per_language)
        write_csv_output(output_dir / "artifact_diagnostics.csv", diagnostics)

    apply_plot_style()
    task_stage_outputs = {}
    for task_group in ["injected", "phrase"]:
        stage_frames = build_task_stage_frames(aggregated, task_group)
        task_stage_outputs[task_group] = stage_frames
        generate_task_outputs(
            task_group=task_group,
            stage_frames=stage_frames,
            plots_root=plots_dir,
            tables_root=tables_dir,
            generated_plots=generated_plots,
            generated_tables=generated_tables,
            skip_plots=skip_plots,
            skip_tables=skip_tables,
        )

    return {
        "aggregated": aggregated,
        "per_language": per_language,
        "task_stage_outputs": task_stage_outputs,
        "generated_plots": generated_plots,
        "generated_tables": generated_tables,
    }


def analyze_results(
    results_root,
    run_dirs=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    include_tasks=None,
    include_run_families=None,
    skip_plots=False,
    skip_tables=False,
):
    selected_run_dirs = [Path(path) for path in run_dirs] if run_dirs else discover_run_dirs(results_root)
    raw_metrics, diagnostics = combine_runs(selected_run_dirs)

    include_tasks = include_tasks or ["injected", "phrase"]
    include_run_families = include_run_families or ["classical", "llm"]
    raw_metrics = apply_filters(raw_metrics, include_tasks, include_run_families)
    diagnostics = diagnostics[diagnostics["run_family"].isin(include_run_families)].copy()
    diagnostics = diagnostics[diagnostics["task_group"].isin(include_tasks)].copy()

    output = create_outputs(
        raw_metrics=raw_metrics,
        diagnostics=diagnostics,
        output_dir=Path(output_dir),
        skip_plots=skip_plots,
        skip_tables=skip_tables,
    )
    output["run_dirs"] = selected_run_dirs
    output["raw_metrics"] = raw_metrics
    output["diagnostics"] = diagnostics
    output["output_dir"] = Path(output_dir)
    return output


def main():
    args = parse_args()
    result = analyze_results(
        results_root=args.results_root,
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        include_tasks=parse_csv_list(args.include_tasks),
        include_run_families=parse_csv_list(args.include_run_families),
        skip_plots=args.skip_plots,
        skip_tables=args.skip_tables,
    )
    print(f"Analyzed {len(result['run_dirs'])} run directories into {result['output_dir']}")
    print(f"Loaded {len(result['raw_metrics'])} raw metric rows")
    print(f"Wrote {len(result['generated_tables'])} table images")
    print(f"Generated {len(result['generated_plots'])} plots")


if __name__ == "__main__":
    main()
