"""
Generate final comprehensive report after all testing is complete.

This script aggregates:
1. Original transformation test results
2. Improved detection method results
3. Comparison and recommendations
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_all_results(results_dir: Path) -> dict:
    """Load all transformation test results."""

    results = []
    result_files = list(results_dir.glob("comparison_results_*_test.json"))

    logger.info(f"Found {len(result_files)} result files")

    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            transform_name = result_file.stem.replace("comparison_results_", "").replace("_test", "")

            results.append({
                'transform': transform_name,
                'f1_score': data['transform']['f1_score'],
                'precision': data['transform']['precision'],
                'recall': data['transform']['recall'],
                'true_positives': data['transform']['true_positives'],
                'false_positives': data['transform']['false_positives'],
                'false_negatives': data['transform']['false_negatives'],
                'num_detected': data['transform']['num_detected'],
                'improvement': data['improvement'],
                'improvement_pct': data['improvement_pct'],
                'best_direct_f1': data['best_direct']['f1_score']
            })

        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    return {
        'results': results,
        'df': pd.DataFrame(results) if results else pd.DataFrame()
    }


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics."""

    if df.empty:
        return {}

    stats = {
        'total_transforms_tested': len(df),
        'mean_f1': float(df['f1_score'].mean()),
        'median_f1': float(df['f1_score'].median()),
        'max_f1': float(df['f1_score'].max()),
        'min_f1': float(df['f1_score'].min()),
        'std_f1': float(df['f1_score'].std()),

        'mean_precision': float(df['precision'].mean()),
        'mean_recall': float(df['recall'].mean()),

        'transforms_with_detections': int((df['num_detected'] > 0).sum()),
        'transforms_with_true_positives': int((df['true_positives'] > 0).sum()),

        'best_transform': df.loc[df['f1_score'].idxmax(), 'transform'] if not df.empty else None,
        'best_f1_score': float(df['f1_score'].max()) if not df.empty else 0.0,

        'transforms_beating_direct': int((df['improvement'] > 0).sum()),
        'mean_improvement': float(df['improvement'].mean()),
    }

    return stats


def create_comprehensive_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualization suite."""

    if df.empty:
        logger.warning("No data to visualize")
        return

    # Sort for better visualization
    df_sorted = df.sort_values('f1_score', ascending=False)

    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. F1 Score ranking
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['green' if x > 0 else 'red' if x == 0 else 'orange'
              for x in df_sorted['true_positives']]
    ax1.barh(df_sorted['transform'], df_sorted['f1_score'],
             color=colors, alpha=0.7)
    ax1.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Transformation Method Performance (F1 Score)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 2. Precision vs Recall scatter
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(df['f1_score'] / df['f1_score'].max()) if df['f1_score'].max() > 0 else 'blue'
    scatter = ax2.scatter(df['recall'], df['precision'],
                         c=df['f1_score'], s=200, alpha=0.7,
                         cmap='viridis', edgecolors='black', linewidth=1)
    for _, row in df.iterrows():
        ax2.annotate(row['transform'], (row['recall'], row['precision']),
                    fontsize=8, alpha=0.7, xytext=(5, 5),
                    textcoords='offset points')
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1 Score')

    # 3. Detection counts
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(['True\nPositives', 'False\nPositives', 'False\nNegatives'],
            [df['true_positives'].sum(), df['false_positives'].sum(), df['false_negatives'].sum()],
            color=['green', 'orange', 'red'], alpha=0.7)
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Aggregate Detection Results', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Improvement over direct detection
    ax4 = fig.add_subplot(gs[1, 2])
    df_sorted_imp = df.sort_values('improvement_pct', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in df_sorted_imp['improvement_pct']]
    ax4.barh(df_sorted_imp['transform'], df_sorted_imp['improvement_pct'],
             color=colors, alpha=0.7)
    ax4.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('vs. Best Direct Method', fontsize=12, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Number of detections
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.barh(df_sorted['transform'], df_sorted['num_detected'],
             color='steelblue', alpha=0.7)
    ax5.set_xlabel('Samples Detected', fontsize=11, fontweight='bold')
    ax5.set_title('Detection Volume', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)

    # 6. True positive rate
    ax6 = fig.add_subplot(gs[2, 1])
    df_sorted['tp_rate'] = df_sorted['true_positives'] / 50  # 50 total poisoned
    colors_tp = plt.cm.RdYlGn(df_sorted['tp_rate'])
    ax6.barh(df_sorted['transform'], df_sorted['tp_rate'] * 100,
             color=colors_tp, alpha=0.7)
    ax6.set_xlabel('True Positive Rate (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Poison Detection Rate', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # 7. Summary statistics text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = f"""
    SUMMARY STATISTICS
    {'='*40}

    Total Transforms: {len(df)}

    F1 Scores:
      • Best: {df['f1_score'].max():.4f}
      • Mean: {df['f1_score'].mean():.4f}
      • Median: {df['f1_score'].median():.4f}

    Detection Performance:
      • Transforms with TP: {(df['true_positives'] > 0).sum()}
      • Mean Precision: {df['precision'].mean():.4f}
      • Mean Recall: {df['recall'].mean():.4f}

    Best Transform:
      {df.loc[df['f1_score'].idxmax(), 'transform']}

    Improvement vs Direct:
      • Positive: {(df['improvement'] > 0).sum()} / {len(df)}
      • Mean: {df['improvement'].mean():+.4f}
    """

    ax7.text(0.1, 0.5, stats_text, fontsize=10,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Comprehensive Transformation Method Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = output_dir / 'final_comprehensive_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comprehensive visualization saved to {output_path}")
    plt.close()


def generate_markdown_report(data: dict, stats: dict, output_dir: Path):
    """Generate detailed markdown report."""

    df = data['df']

    if df.empty:
        report = "# Final Report\n\nNo results available yet.\n"
    else:
        report = f"""# Transformation Methods: Final Test Report

## Test Configuration
- **Total Transformations Tested**: {len(df)}
- **Test Set Size**: 50 samples
- **Training Set Size**: 1000 samples (50 poisoned)
- **Task**: Sentiment analysis (polarity)

## Overall Performance

### F1 Score Statistics
- **Best F1**: {stats['best_f1_score']:.4f} ({stats['best_transform']})
- **Mean F1**: {stats['mean_f1']:.4f}
- **Median F1**: {stats['median_f1']:.4f}
- **Std Dev**: {stats['std_f1']:.4f}

### Detection Statistics
- **Transforms with Detections**: {stats['transforms_with_detections']} / {stats['total_transforms_tested']}
- **Transforms with True Positives**: {stats['transforms_with_true_positives']} / {stats['total_transforms_tested']}
- **Transforms Beating Direct Detection**: {stats['transforms_beating_direct']} / {stats['total_transforms_tested']}

## Detailed Results

### Top 5 Performing Transformations

{df.nlargest(5, 'f1_score')[['transform', 'f1_score', 'precision', 'recall', 'true_positives', 'num_detected']].to_markdown(index=False)}

### All Results (sorted by F1 score)

{df.sort_values('f1_score', ascending=False)[['transform', 'f1_score', 'precision', 'recall', 'improvement_pct']].to_markdown(index=False)}

## Analysis

### Key Findings

1. **Best Performing Transform**: {stats['best_transform']} achieved F1={stats['best_f1_score']:.4f}

2. **Comparison to Direct Detection**:
   - Mean improvement: {stats['mean_improvement']:+.4f}
   - {stats['transforms_beating_direct']} out of {stats['total_transforms_tested']} transforms beat direct detection

3. **Detection Characteristics**:
   - Mean Precision: {stats['mean_precision']:.4f}
   - Mean Recall: {stats['mean_recall']:.4f}

### Observations

{"#### ⚠️ All transforms failed (F1=0)" if stats['max_f1'] == 0 else "#### ✅ Some transforms showed promise"}

{"The original detection method with 10th percentile thresholds is too restrictive. See TRANSFORMATION_RECOMMENDATIONS.md for improved methods." if stats['max_f1'] < 0.01 else ""}

## Recommendations

### Immediate Actions

1. **Use Improved Detection Methods**: The original threshold strategy is flawed. Use the improved detector at `poison_detection/detection/improved_detector.py`

2. **Best Transform Choice**: Based on these results, prioritize:
   - {df.nlargest(1, 'f1_score')['transform'].values[0] if not df.empty else "TBD"}
   - Combine multiple transforms for ensemble detection

3. **Threshold Tuning**: Run adaptive threshold selection instead of fixed 10th percentile

### Next Steps

1. Re-evaluate all transforms with improved detection methods
2. Test ensemble approaches combining multiple transforms
3. Implement hybrid detection pipeline (direct + transform)

## Files Generated

- `final_comprehensive_report.png`: Visual summary
- `final_report.md`: This document
- `final_results.json`: Machine-readable results

## References

- Analysis: `experiments/analysis_transform_failure.md`
- Recommendations: `experiments/TRANSFORMATION_RECOMMENDATIONS.md`
- Improved Detector: `poison_detection/detection/improved_detector.py`

---
Generated: {pd.Timestamp.now()}
"""

    output_path = output_dir / 'final_report.md'
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Markdown report saved to {output_path}")


def main():
    """Main execution."""

    results_dir = Path("experiments/results/transform_comparison/polarity")

    logger.info("=" * 80)
    logger.info("GENERATING FINAL COMPREHENSIVE REPORT")
    logger.info("=" * 80)

    # Load all results
    data = load_all_results(results_dir)

    if data['df'].empty:
        logger.warning("No test results found yet. Tests may still be running.")
        logger.info("Run this script again after tests complete.")
        return

    # Generate statistics
    stats = generate_summary_statistics(data['df'])

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_comprehensive_visualizations(data['df'], results_dir)

    # Generate markdown report
    logger.info("Generating markdown report...")
    generate_markdown_report(data, stats, results_dir)

    # Save JSON
    output_json = results_dir / 'final_results.json'
    with open(output_json, 'w') as f:
        json.dump({
            'statistics': stats,
            'results': data['results']
        }, f, indent=2)

    logger.info(f"JSON results saved to {output_json}")

    logger.info("\n" + "=" * 80)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput files in: {results_dir}")
    logger.info("- final_comprehensive_report.png")
    logger.info("- final_report.md")
    logger.info("- final_results.json")


if __name__ == "__main__":
    main()
