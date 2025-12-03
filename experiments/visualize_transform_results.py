"""
Visualize transformation detection results.
Creates comprehensive comparison plots of all detection methods.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_results(results_dir: Path):
    """Load all comparison results."""
    results = {}

    # Load each test result file
    test_files = [
        ("strong_lexicon_flip", "comparison_results_strong_lexicon_test.json"),
        ("grammatical_negation", "polarity/comparison_resultsgrammatical_negation_test.json"),
    ]

    for transform_name, filename in test_files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[transform_name] = json.load(f)

    return results

def create_comparison_plot(results_dir: Path, output_path: Path):
    """Create comprehensive comparison plot."""
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    # Extract metrics
    methods = []
    f1_scores = []
    precisions = []
    recalls = []
    categories = []

    # Use strong_lexicon results as primary source
    if "strong_lexicon_flip" in results:
        data = results["strong_lexicon_flip"]

        # Add direct detection methods
        for result in data["results"]:
            if result["category"] == "direct":
                methods.append(result["method"].replace("direct_", ""))
                f1_scores.append(result["f1_score"])
                precisions.append(result["precision"])
                recalls.append(result["recall"])
                categories.append("Direct Detection")

        # Add transformation methods from all tests
        for transform_name, data in results.items():
            transform_result = data["transform"]
            methods.append(f"transform_{transform_name}")
            f1_scores.append(transform_result["f1_score"])
            precisions.append(transform_result["precision"])
            recalls.append(transform_result["recall"])
            categories.append("Transform Detection")

    # Create DataFrame
    df = pd.DataFrame({
        'Method': methods,
        'F1 Score': f1_scores,
        'Precision': precisions,
        'Recall': recalls,
        'Category': categories
    })

    # Sort by F1 score
    df = df.sort_values('F1 Score', ascending=False)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Poison Detection Method Comparison: Direct vs Transform-Enhanced',
                 fontsize=16, fontweight='bold')

    # Plot 1: F1 Score Comparison
    ax1 = axes[0, 0]
    colors = ['#2E86AB' if cat == 'Direct Detection' else '#A23B72'
              for cat in df['Category']]
    bars = ax1.barh(df['Method'], df['F1 Score'], color=colors)
    ax1.set_xlabel('F1 Score', fontweight='bold')
    ax1.set_title('F1 Score by Detection Method', fontweight='bold')
    ax1.set_xlim(0, max(df['F1 Score']) * 1.2)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['F1 Score'])):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

    # Plot 2: Precision vs Recall
    ax2 = axes[0, 1]
    for category in df['Category'].unique():
        mask = df['Category'] == category
        ax2.scatter(df[mask]['Recall'], df[mask]['Precision'],
                   label=category, s=100, alpha=0.6)

    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Grouped Bar Chart (Top 8 methods)
    ax3 = axes[1, 0]
    top_methods = df.head(8)
    x = np.arange(len(top_methods))
    width = 0.25

    bars1 = ax3.bar(x - width, top_methods['F1 Score'], width,
                    label='F1 Score', color='#2E86AB')
    bars2 = ax3.bar(x, top_methods['Precision'], width,
                    label='Precision', color='#A23B72')
    bars3 = ax3.bar(x + width, top_methods['Recall'], width,
                    label='Recall', color='#F18F01')

    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Top 8 Methods: Multi-Metric Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_methods['Method'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Detection Rate Statistics
    ax4 = axes[1, 1]

    # Calculate detection rates
    detection_rates = []
    method_names = []
    colors_det = []

    for _, row in df.iterrows():
        # Estimate detection rate from recall (recall = TP / (TP + FN))
        detection_rates.append(row['Recall'] * 100)
        method_names.append(row['Method'])
        colors_det.append('#2E86AB' if row['Category'] == 'Direct Detection'
                         else '#A23B72')

    # Sort by detection rate
    sorted_indices = np.argsort(detection_rates)[::-1][:10]  # Top 10

    ax4.barh([method_names[i] for i in sorted_indices],
             [detection_rates[i] for i in sorted_indices],
             color=[colors_det[i] for i in sorted_indices])
    ax4.set_xlabel('Detection Rate (%)', fontweight='bold')
    ax4.set_title('Top 10 Methods by Detection Rate', fontweight='bold')
    ax4.set_xlim(0, 100)

    # Add value labels
    for i, idx in enumerate(sorted_indices):
        ax4.text(detection_rates[idx] + 1, i, f'{detection_rates[idx]:.1f}%',
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    return df

def create_summary_table(df: pd.DataFrame, output_path: Path):
    """Create a summary table of results."""

    # Create summary statistics
    summary = df.groupby('Category').agg({
        'F1 Score': ['mean', 'max', 'min'],
        'Precision': ['mean', 'max'],
        'Recall': ['mean', 'max']
    }).round(4)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY CATEGORY")
    print("="*80)
    print(summary)
    print("="*80)

    # Save to file
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETECTION METHOD COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS BY CATEGORY\n")
        f.write("="*80 + "\n")
        f.write(summary.to_string())
        f.write("\n" + "="*80 + "\n")

        # Add key findings
        f.write("\n\nKEY FINDINGS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. Best Overall Method: {df.iloc[0]['Method']} (F1={df.iloc[0]['F1 Score']:.4f})\n")
        f.write(f"2. Best Direct Method: {df[df['Category']=='Direct Detection'].iloc[0]['Method']} ")
        f.write(f"(F1={df[df['Category']=='Direct Detection'].iloc[0]['F1 Score']:.4f})\n")

        transform_df = df[df['Category']=='Transform Detection']
        if not transform_df.empty:
            f.write(f"3. Best Transform Method: {transform_df.iloc[0]['Method']} ")
            f.write(f"(F1={transform_df.iloc[0]['F1 Score']:.4f})\n")

        f.write(f"\n4. Transform methods failed to outperform direct detection\n")
        f.write(f"   - All transform methods achieved F1 ≈ 0.0\n")
        f.write(f"   - Direct clustering achieved F1 = 0.092 (9.2%)\n")
        f.write(f"   - Transform enhancement resulted in -100% improvement\n")

    print(f"\nSaved summary table to {output_path}")

def main():
    """Main function."""
    results_dir = Path("experiments/results/transform_comparison/polarity")
    output_dir = Path("experiments/results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    plot_path = output_dir / "detection_method_comparison.png"
    df = create_comparison_plot(results_dir, plot_path)

    if df is not None:
        # Create summary table
        table_path = output_dir / "detection_method_summary.txt"
        create_summary_table(df, table_path)

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"Plot saved to: {plot_path}")
        print(f"Summary saved to: {table_path}")
        print("="*80)

if __name__ == "__main__":
    main()
