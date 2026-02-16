# /// script
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
#     "scipy",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2
from itertools import combinations
import numpy as np

from detect_foreign_words import LANG_NAMES

# Configuration
TEXT_RESULTS_FILE = 'evaluation_results.jsonl'
WORD_RESULTS_FILE = 'word_results.jsonl'
PLOTS_DIR = 'plots'

# Model display names
MODEL_NAMES = {
    'fasttext-lid.176': 'FastText 176',
    'fasttext-lid.176compressed': 'FastText Compressed',
    'glotlid': 'GlotLID',
    'lingua': 'Lingua'
}

# ============================================================================
# Helper Functions
# ============================================================================

def mcnemar_test(contingency_table):
    """
    Perform McNemar's test on a 2x2 contingency table.

    contingency_table format: [[both_correct, model1_only],
                                [model2_only, both_wrong]]

    Returns: namedtuple with statistic and pvalue
    """
    from collections import namedtuple

    b = contingency_table[0][1]  # model1 correct, model2 wrong
    c = contingency_table[1][0]  # model1 wrong, model2 correct

    # McNemar's test statistic with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0

    # p-value from chi-square distribution with 1 degree of freedom
    pvalue = 1 - chi2.cdf(statistic, df=1)

    Result = namedtuple('McNemarResult', ['statistic', 'pvalue'])
    return Result(statistic, pvalue)

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load evaluation results from JSONL files."""
    print("Loading data...")

    # Load text-level results
    text_df = pd.read_json(TEXT_RESULTS_FILE, lines=True)

    # Load word-level results
    word_df = pd.read_json(WORD_RESULTS_FILE, lines=True)

    # Add 'correct' column to word results
    word_df['correct'] = word_df['true_lang'] == word_df['detected_lang']

    print(f"Loaded {len(text_df)} text-level evaluations")
    print(f"Loaded {len(word_df)} word-level evaluations")
    print(f"Models: {sorted(text_df['model'].unique())}")
    print(f"Languages: {sorted(text_df['true_lang'].unique())}")
    print()

    return text_df, word_df

# ============================================================================
# Text-Level Analysis Functions
# ============================================================================

def analyze_text_level(text_df):
    """Perform text-level analysis."""
    print("=" * 80)
    print("TEXT-LEVEL EVALUATION")
    print("=" * 80)
    print()

    # Overall accuracy by model
    print("Overall Model Accuracy:")
    accuracy_by_model = text_df.groupby('model')['correct'].mean() * 100
    for model in sorted(accuracy_by_model.index, key=lambda x: MODEL_NAMES.get(x, x)):
        print(f"  {MODEL_NAMES.get(model, model):25s}: {accuracy_by_model[model]:5.2f}%")
    print()

    # Confidence statistics
    print("Mean Confidence Score:")
    conf_by_model = text_df.groupby('model')['confidence'].mean()
    for model in sorted(conf_by_model.index, key=lambda x: MODEL_NAMES.get(x, x)):
        print(f"  {MODEL_NAMES.get(model, model):25s}: {conf_by_model[model]:.4f}")
    print()

    # Per-language accuracy
    print("Per-Language Accuracy:")
    lang_accuracy = text_df.pivot_table(
        values='correct',
        index='true_lang',
        columns='model',
        aggfunc='mean'
    ) * 100

    # Sort by language name
    lang_accuracy = lang_accuracy.loc[sorted(lang_accuracy.index, key=lambda x: LANG_NAMES.get(x, x))]

    # Print formatted table
    model_order = sorted(lang_accuracy.columns, key=lambda x: MODEL_NAMES.get(x, x))
    header = f"  {'Language':15s} " + " ".join([f"{MODEL_NAMES.get(m, m):12s}" for m in model_order])
    print(header)
    print("  " + "-" * (len(header) - 2))

    for lang in lang_accuracy.index:
        lang_name = LANG_NAMES.get(lang, lang)
        values = " ".join([f"{lang_accuracy.loc[lang, m]:12.2f}" for m in model_order])
        print(f"  {lang_name:15s} {values}")
    print()

    # Most common misclassifications
    print("Most Common Misclassifications (across all models):")
    misclassifications = text_df[~text_df['correct']].groupby(['true_lang', 'detected_lang']).size()
    misclassifications = misclassifications.sort_values(ascending=False).head(10)

    for (true_lang, detected_lang), count in misclassifications.items():
        true_name = LANG_NAMES.get(true_lang, true_lang)
        detected_name = LANG_NAMES.get(detected_lang, detected_lang)
        print(f"  {true_name:15s} -> {detected_name:15s}: {count:3d}")
    print()

    return accuracy_by_model, lang_accuracy

def analyze_word_level(word_df):
    """Perform word-level analysis."""
    print("=" * 80)
    print("WORD-LEVEL EVALUATION")
    print("=" * 80)
    print()

    # Overall accuracy by model
    print("Overall Model Accuracy:")
    accuracy_by_model = word_df.groupby('model')['correct'].mean() * 100
    for model in sorted(accuracy_by_model.index, key=lambda x: MODEL_NAMES.get(x, x)):
        print(f"  {MODEL_NAMES.get(model, model):25s}: {accuracy_by_model[model]:5.2f}%")
    print()

    # Confidence statistics
    print("Mean Confidence Score:")
    conf_by_model = word_df.groupby('model')['confidence'].mean()
    for model in sorted(conf_by_model.index, key=lambda x: MODEL_NAMES.get(x, x)):
        print(f"  {MODEL_NAMES.get(model, model):25s}: {conf_by_model[model]:.4f}")
    print()

    # Per-language accuracy
    print("Per-Language Accuracy:")
    lang_accuracy = word_df.pivot_table(
        values='correct',
        index='true_lang',
        columns='model',
        aggfunc='mean'
    ) * 100

    # Sort by language name
    lang_accuracy = lang_accuracy.loc[sorted(lang_accuracy.index, key=lambda x: LANG_NAMES.get(x, x))]

    # Print formatted table
    model_order = sorted(lang_accuracy.columns, key=lambda x: MODEL_NAMES.get(x, x))
    header = f"  {'Language':15s} " + " ".join([f"{MODEL_NAMES.get(m, m):12s}" for m in model_order])
    print(header)
    print("  " + "-" * (len(header) - 2))

    for lang in lang_accuracy.index:
        lang_name = LANG_NAMES.get(lang, lang)
        values = " ".join([f"{lang_accuracy.loc[lang, m]:12.2f}" for m in model_order])
        print(f"  {lang_name:15s} {values}")
    print()

    # Most common misclassifications
    print("Most Common Misclassifications (across all models):")
    misclassifications = word_df[~word_df['correct']].groupby(['true_lang', 'detected_lang']).size()
    misclassifications = misclassifications.sort_values(ascending=False).head(10)

    for (true_lang, detected_lang), count in misclassifications.items():
        true_name = LANG_NAMES.get(true_lang, true_lang)
        detected_name = LANG_NAMES.get(detected_lang, detected_lang)
        print(f"  {true_name:15s} -> {detected_name:15s}: {count:3d}")
    print()

    return accuracy_by_model, lang_accuracy

# ============================================================================
# Statistical Significance Testing
# ============================================================================

def perform_mcnemar_tests(df, level_name):
    """Perform pairwise McNemar's tests between models."""
    print(f"Statistical Significance Tests ({level_name}):")
    print("(McNemar's test, α = 0.05)")
    print()

    models = sorted(df['model'].unique())

    # Create a table of results for each pair
    for model1, model2 in combinations(models, 2):
        # Get results for both models on same samples
        # We need to ensure we're comparing the same samples
        if 'text' in df.columns:
            # Text level - group by text
            df1 = df[df['model'] == model1].set_index('text')
            df2 = df[df['model'] == model2].set_index('text')
        else:
            # Word level - group by word and true_lang
            df1 = df[df['model'] == model1].set_index(['word', 'true_lang'])
            df2 = df[df['model'] == model2].set_index(['word', 'true_lang'])

        # Get common samples
        common_idx = df1.index.intersection(df2.index)

        # Build contingency table for McNemar's test
        # Format: [[both_correct, model1_correct_model2_wrong],
        #          [model1_wrong_model2_correct, both_wrong]]
        model1_correct = df1.loc[common_idx, 'correct']
        model2_correct = df2.loc[common_idx, 'correct']

        both_correct = ((model1_correct) & (model2_correct)).sum()
        model1_only = ((model1_correct) & (~model2_correct)).sum()
        model2_only = ((~model1_correct) & (model2_correct)).sum()
        both_wrong = ((~model1_correct) & (~model2_correct)).sum()

        contingency = [[both_correct, model1_only],
                       [model2_only, both_wrong]]

        # Perform McNemar's test
        result = mcnemar_test(contingency)

        m1_name = MODEL_NAMES.get(model1, model1)
        m2_name = MODEL_NAMES.get(model2, model2)
        sig = "***" if result.pvalue < 0.001 else "**" if result.pvalue < 0.01 else "*" if result.pvalue < 0.05 else "ns"

        print(f"  {m1_name:20s} vs {m2_name:20s}: p = {result.pvalue:.4f} {sig}")

    print()
    print("  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print()

# ============================================================================
# Visualization Functions
# ============================================================================

def setup_plot_style():
    """Set up consistent plot styling."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def plot_accuracy_comparison(text_acc, word_acc, output_file):
    """Create bar chart comparing model accuracy at text and word level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    models = sorted(text_acc.index, key=lambda x: MODEL_NAMES.get(x, x))
    model_labels = [MODEL_NAMES.get(m, m) for m in models]

    x = np.arange(len(models))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, [text_acc[m] for m in models], width, label='Text-level', color='#2E86AB')
    bars2 = ax.bar(x + width/2, [word_acc[m] for m in models], width, label='Word-level', color='#A23B72')

    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison: Text vs Word Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_confidence_distributions(text_df, word_df, output_file):
    """Create box plots showing confidence distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    models = sorted(text_df['model'].unique(), key=lambda x: MODEL_NAMES.get(x, x))
    text_df['model_name'] = text_df['model'].map(lambda x: MODEL_NAMES.get(x, x))
    word_df['model_name'] = word_df['model'].map(lambda x: MODEL_NAMES.get(x, x))

    # Text-level
    sns.boxplot(data=text_df, x='model_name', y='confidence',
                order=[MODEL_NAMES.get(m, m) for m in models],
                palette='Set2', ax=ax1)
    ax1.set_title('Text-Level Confidence Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Confidence Score', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)

    # Word-level
    sns.boxplot(data=word_df, x='model_name', y='confidence',
                order=[MODEL_NAMES.get(m, m) for m in models],
                palette='Set2', ax=ax2)
    ax2.set_title('Word-Level Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Confidence Score', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_language_heatmap(lang_accuracy, title, output_file):
    """Create heatmap of per-language accuracy by model."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reorder columns by model name
    model_order = sorted(lang_accuracy.columns, key=lambda x: MODEL_NAMES.get(x, x))
    lang_accuracy_ordered = lang_accuracy[model_order]
    lang_accuracy_ordered.columns = [MODEL_NAMES.get(m, m) for m in model_order]

    # Reorder rows by language name
    lang_accuracy_ordered.index = [LANG_NAMES.get(lang, lang) for lang in lang_accuracy_ordered.index]

    # Create heatmap
    sns.heatmap(lang_accuracy_ordered, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Language', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_confusion_matrix(df, model, output_file):
    """Create confusion matrix heatmap for a specific model."""
    # Filter data for this model
    model_df = df[df['model'] == model]

    # Get unique languages (sorted)
    languages = sorted(model_df['true_lang'].unique(), key=lambda x: LANG_NAMES.get(x, x))
    lang_labels = [LANG_NAMES.get(lang, lang) for lang in languages]

    # Create confusion matrix
    cm = confusion_matrix(model_df['true_lang'], model_df['detected_lang'], labels=languages)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=lang_labels, yticklabels=lang_labels,
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5, linecolor='gray', ax=ax)

    model_name = MODEL_NAMES.get(model, model)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Detected Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Language', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Created/verified plots directory: {PLOTS_DIR}")
    print()

    # Set up plotting style
    setup_plot_style()

    # Load data
    text_df, word_df = load_data()

    # Text-level analysis
    text_accuracy, text_lang_accuracy = analyze_text_level(text_df)
    perform_mcnemar_tests(text_df, "Text-Level")

    # Word-level analysis
    word_accuracy, word_lang_accuracy = analyze_word_level(word_df)
    perform_mcnemar_tests(word_df, "Word-Level")

    # Comparison summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    best_text_model = text_accuracy.idxmax()
    best_word_model = word_accuracy.idxmax()

    print(f"Best Model (Text-Level): {MODEL_NAMES.get(best_text_model, best_text_model)} ({text_accuracy[best_text_model]:.2f}%)")
    print(f"Best Model (Word-Level): {MODEL_NAMES.get(best_word_model, best_word_model)} ({word_accuracy[best_word_model]:.2f}%)")
    print()

    print("Accuracy Drop (Text -> Word):")
    for model in sorted(text_accuracy.index, key=lambda x: MODEL_NAMES.get(x, x)):
        drop = text_accuracy[model] - word_accuracy[model]
        print(f"  {MODEL_NAMES.get(model, model):25s}: -{drop:.2f} percentage points")
    print()

    # Create visualizations
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()

    plot_accuracy_comparison(
        text_accuracy, word_accuracy,
        os.path.join(PLOTS_DIR, '01_accuracy_comparison.png')
    )

    plot_confidence_distributions(
        text_df, word_df,
        os.path.join(PLOTS_DIR, '02_confidence_distributions.png')
    )

    plot_language_heatmap(
        text_lang_accuracy,
        'Per-Language Accuracy: Text-Level',
        os.path.join(PLOTS_DIR, '03_text_language_heatmap.png')
    )

    plot_language_heatmap(
        word_lang_accuracy,
        'Per-Language Accuracy: Word-Level',
        os.path.join(PLOTS_DIR, '04_word_language_heatmap.png')
    )

    # Confusion matrices for each model
    models = sorted(text_df['model'].unique(), key=lambda x: MODEL_NAMES.get(x, x))
    for idx, model in enumerate(models, start=5):
        plot_confusion_matrix(
            text_df, model,
            os.path.join(PLOTS_DIR, f'{idx:02d}_confusion_matrix_{model.replace(".", "_")}.png')
        )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {PLOTS_DIR}/")
    print()

if __name__ == '__main__':
    main()
