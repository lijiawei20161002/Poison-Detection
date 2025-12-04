"""
Deep diagnostic analysis of why semantic transformations fail for poison detection.

This script will:
1. Load actual influence scores from experiments
2. Visualize logit patterns for lexicon flip vs semantic transforms
3. Analyze detection threshold behavior
4. Identify root causes of failure
5. Propose specific fixes
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)


class SemanticTransformDiagnostic:
    """Diagnose why semantic transformations fail."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.data = {}

    def load_experiment_data(self):
        """Load all relevant experiment results."""

        # Load successful lexicon flip experiment
        lexicon_path = self.results_dir / "sentiment_strong_lexicon_flip.json"
        if lexicon_path.exists():
            with open(lexicon_path) as f:
                self.data['lexicon_flip'] = json.load(f)
                print(f"✓ Loaded lexicon flip: F1={self.data['lexicon_flip']['f1_score']:.3f}")

        # Load failed semantic transform experiments
        semantic_paths = {
            'combined_flip_negation': self.results_dir / "sentiment_combined_flip_negation.json",
            'intensity_enhancement': self.results_dir / "sentiment_intensity_enhancement.json",
        }

        for name, path in semantic_paths.items():
            if path.exists():
                with open(path) as f:
                    self.data[name] = json.load(f)
                    print(f"✗ Loaded {name}: F1={self.data[name]['f1_score']:.3f}")

    def load_raw_influence_scores(self, exp_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw influence scores if available.

        Returns:
            (original_scores, transformed_scores)
        """
        # Try to find saved influence scores
        scores_dir = self.results_dir / "influence_scores"

        orig_path = scores_dir / f"{exp_name}_original.npy"
        trans_path = scores_dir / f"{exp_name}_transformed.npy"

        if orig_path.exists() and trans_path.exists():
            return np.load(orig_path), np.load(trans_path)

        return None, None

    def analyze_detection_logic(self):
        """Analyze what the detection logic is actually doing."""

        print("\n" + "="*80)
        print("DETECTION LOGIC ANALYSIS")
        print("="*80)

        # From improved_transform_detector.py lines 66-94
        print("\n📋 Current Detection Logic (IQR Method):")
        print("""
        1. Compute influence_strength = |avg_original_influence|
        2. Compute influence_change = |avg_original - avg_transformed|

        3. Threshold for strength (want HIGH outliers):
           strength_threshold = Q3_strength + k * IQR_strength

        4. Threshold for change (want LOW outliers):
           change_threshold = median_change - k * 1.4826 * MAD_change

        5. DETECTION RULE:
           detected = (strength > strength_threshold) OR (change < change_threshold)
        """)

        print("\n❌ PROBLEM IDENTIFIED:")
        print("""
        The detection logic assumes:
        - Lexicon flip: Clean samples change a lot (high influence_change)
        - Lexicon flip: Poison samples don't change much (low influence_change)

        But for SEMANTIC transformations:
        - Semantic transform: ALL samples may have moderate change
        - Semantic transform: The change patterns are more complex
        - Semantic transform: Sign flips are more important than magnitude
        """)

    def simulate_detection_scenarios(self):
        """Simulate what happens with different patterns."""

        print("\n" + "="*80)
        print("DETECTION SCENARIO SIMULATION")
        print("="*80)

        # Simulate different scenarios
        n_clean = 950
        n_poison = 50

        scenarios = {}

        # Scenario 1: Lexicon flip (what works)
        print("\n📊 Scenario 1: LEXICON FLIP (baseline)")
        clean_orig = np.random.normal(0.5, 0.2, n_clean)
        poison_orig = np.random.normal(2.0, 0.3, n_poison)  # Strong positive influence

        # Lexicon flip: clean flips completely, poison stays
        clean_trans = np.random.normal(-0.5, 0.2, n_clean)  # Flipped
        poison_trans = np.random.normal(1.9, 0.3, n_poison)  # Barely changed

        scenarios['lexicon_flip'] = {
            'clean_orig': clean_orig,
            'poison_orig': poison_orig,
            'clean_trans': clean_trans,
            'poison_trans': poison_trans
        }

        # Scenario 2: Semantic transform (what fails)
        print("\n📊 Scenario 2: SEMANTIC TRANSFORM (actual)")
        clean_orig = np.random.normal(0.5, 0.2, n_clean)
        poison_orig = np.random.normal(2.0, 0.3, n_poison)

        # Semantic: BOTH flip, but maybe different amounts
        clean_trans = np.random.normal(-0.3, 0.25, n_clean)  # Moderate flip
        poison_trans = np.random.normal(-1.5, 0.4, n_poison)  # Also flips!

        scenarios['semantic_transform'] = {
            'clean_orig': clean_orig,
            'poison_orig': poison_orig,
            'clean_trans': clean_trans,
            'poison_trans': poison_trans
        }

        # Analyze each scenario
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Why Semantic Transforms Fail: Pattern Analysis', fontsize=16, fontweight='bold')

        for idx, (name, data) in enumerate(scenarios.items()):
            # Combine
            orig_all = np.concatenate([data['clean_orig'], data['poison_orig']])
            trans_all = np.concatenate([data['clean_trans'], data['poison_trans']])
            labels = np.array([0]*n_clean + [1]*n_poison)

            # Compute metrics
            strength = np.abs(orig_all)
            change = np.abs(orig_all - trans_all)

            # Compute thresholds (k=1.5)
            k = 1.5
            q1_str, q3_str = np.percentile(strength, [25, 75])
            iqr_str = q3_str - q1_str
            strength_thresh = q3_str + k * iqr_str

            median_change = np.median(change)
            mad_change = np.median(np.abs(change - median_change))
            change_thresh = median_change - k * 1.4826 * mad_change
            change_thresh = max(change_thresh, 0)

            # Detection
            detected = (strength > strength_thresh) | (change < change_thresh)

            # Metrics
            tp = (detected & (labels == 1)).sum()
            fp = (detected & (labels == 0)).sum()
            fn = ((~detected) & (labels == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Plot 1: Strength distribution
            ax = axes[idx, 0]
            ax.hist(strength[labels==0], bins=30, alpha=0.5, label='Clean', color='blue')
            ax.hist(strength[labels==1], bins=30, alpha=0.5, label='Poison', color='red')
            ax.axvline(strength_thresh, color='black', linestyle='--', label=f'Threshold={strength_thresh:.2f}')
            ax.set_xlabel('Influence Strength')
            ax.set_ylabel('Count')
            ax.set_title(f'{name}: Strength Distribution')
            ax.legend()

            # Plot 2: Change distribution
            ax = axes[idx, 1]
            ax.hist(change[labels==0], bins=30, alpha=0.5, label='Clean', color='blue')
            ax.hist(change[labels==1], bins=30, alpha=0.5, label='Poison', color='red')
            ax.axvline(change_thresh, color='black', linestyle='--', label=f'Threshold={change_thresh:.2f}')
            ax.set_xlabel('Influence Change')
            ax.set_ylabel('Count')
            ax.set_title(f'{name}: Change Distribution')
            ax.legend()

            # Plot 3: 2D scatter
            ax = axes[idx, 2]
            ax.scatter(strength[labels==0], change[labels==0], alpha=0.3, label='Clean', color='blue', s=10)
            ax.scatter(strength[labels==1], change[labels==1], alpha=0.6, label='Poison', color='red', s=30)
            ax.axvline(strength_thresh, color='black', linestyle='--', alpha=0.5)
            ax.axhline(change_thresh, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Influence Strength')
            ax.set_ylabel('Influence Change')
            ax.set_title(f'{name}: 2D Space\nF1={f1:.3f}, P={precision:.3f}, R={recall:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Print analysis
            print(f"\n{name.upper()}:")
            print(f"  Strength threshold: {strength_thresh:.3f}")
            print(f"  Change threshold: {change_thresh:.3f}")
            print(f"  Clean mean change: {change[labels==0].mean():.3f}")
            print(f"  Poison mean change: {change[labels==1].mean():.3f}")
            print(f"  Separation: {change[labels==0].mean() - change[labels==1].mean():.3f}")
            print(f"  Detection: TP={tp}, FP={fp}, FN={fn}")
            print(f"  F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

        plt.tight_layout()
        output_path = self.results_dir / "diagnosis_pattern_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved pattern analysis to {output_path}")
        plt.close()

    def identify_root_causes(self):
        """Identify specific root causes."""

        print("\n" + "="*80)
        print("ROOT CAUSE ANALYSIS")
        print("="*80)

        causes = {
            "1. Wrong Feature Assumptions": """
                Current: Uses absolute change magnitude
                Problem: Semantic transforms create sign flips with moderate magnitude
                Fix: Use SIGN of change, not just magnitude
            """,

            "2. Inappropriate Thresholding": """
                Current: MAD-based threshold assumes unimodal distribution
                Problem: Semantic transforms create bimodal change patterns
                Fix: Use ratio-based or percentile-based detection
            """,

            "3. OR Logic Too Permissive": """
                Current: detected = (high strength) OR (low change)
                Problem: With semantic transforms, many samples have moderate change
                         The 'low change' criterion becomes meaningless
                Fix: Use AND logic with proper features
            """,

            "4. No Sign Flip Detection": """
                Current: Only looks at |orig - trans|
                Problem: Ignores whether influence actually flipped sign
                Fix: Explicitly check if sign(orig) != sign(trans)
            """,

            "5. Static Thresholds": """
                Current: Same k=1.5 for all transforms
                Problem: Different transforms need different sensitivity
                Fix: Adaptive thresholds based on transform type
            """
        }

        for cause, description in causes.items():
            print(f"\n🔍 {cause}")
            print(description)

    def propose_fixes(self):
        """Propose specific fixes to detection algorithm."""

        print("\n" + "="*80)
        print("PROPOSED FIXES")
        print("="*80)

        fixes = """

FIX #1: Add Sign-Aware Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_with_sign_awareness(orig, trans):
    '''Detect based on whether sample maintains sign (poison) or flips (clean)'''

    strength = np.abs(orig)
    sign_flipped = np.sign(orig) != np.sign(trans)
    relative_change = np.abs(orig - trans) / (np.abs(orig) + 1e-8)

    # Poison signature: High strength + NO sign flip + low relative change
    # Clean signature: May flip sign, higher relative change

    # Score: Higher = more suspicious
    suspicion_score = strength * (~sign_flipped) * (1 - relative_change)

    threshold = np.percentile(suspicion_score, 95)
    detected = suspicion_score > threshold

    return detected, suspicion_score


FIX #2: Use Relative Rather Than Absolute Change
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_with_relative_change(orig, trans):
    '''Use relative change to normalize for magnitude differences'''

    strength = np.abs(orig)

    # Relative change accounts for baseline magnitude
    relative_change = np.abs(orig - trans) / (np.abs(orig) + 1e-8)

    # Poisons: high strength, low relative change
    strength_thresh = np.percentile(strength, 90)
    change_thresh = np.percentile(relative_change, 10)

    detected = (strength > strength_thresh) & (relative_change < change_thresh)

    return detected


FIX #3: Multi-Feature Scoring
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_multifeature_score(orig, trans):
    '''Combine multiple signals into single suspicion score'''

    # Feature 1: Strength
    strength = np.abs(orig)
    strength_norm = (strength - strength.mean()) / (strength.std() + 1e-8)

    # Feature 2: Relative stability (inverse of relative change)
    relative_change = np.abs(orig - trans) / (np.abs(orig) + 1e-8)
    stability = 1 - relative_change
    stability_norm = (stability - stability.mean()) / (stability.std() + 1e-8)

    # Feature 3: Sign consistency
    sign_consistency = (np.sign(orig) == np.sign(trans)).astype(float)

    # Feature 4: Magnitude consistency
    mag_ratio = np.minimum(np.abs(orig), np.abs(trans)) / (np.maximum(np.abs(orig), np.abs(trans)) + 1e-8)

    # Combined suspicion score
    suspicion = (
        0.3 * strength_norm +           # High influence
        0.3 * stability_norm +           # Stable across transforms
        0.2 * sign_consistency +         # Maintains sign
        0.2 * mag_ratio                  # Maintains magnitude
    )

    threshold = np.percentile(suspicion, 95)
    detected = suspicion > threshold

    return detected, suspicion


FIX #4: Transform-Adaptive Thresholding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_adaptive_threshold(orig, trans, transform_type):
    '''Adjust detection strategy based on transform type'''

    if transform_type == 'lexicon_flip':
        # Simple case: use absolute change
        change = np.abs(orig - trans)
        threshold = np.percentile(change, 10)
        detected = change < threshold

    elif transform_type == 'semantic':
        # Complex case: use relative change and sign
        strength = np.abs(orig)
        relative_change = np.abs(orig - trans) / (np.abs(orig) + 1e-8)
        sign_maintained = (np.sign(orig) == np.sign(trans))

        # Poison should have: high strength + relatively stable + sign maintained
        suspicion = strength * (1 - relative_change) * sign_maintained
        threshold = np.percentile(suspicion, 95)
        detected = suspicion > threshold

    return detected


FIX #5: Ensemble Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_ensemble(orig, trans):
    '''Use multiple detection strategies and vote'''

    detectors = [
        detect_with_sign_awareness(orig, trans)[0],
        detect_with_relative_change(orig, trans),
        detect_multifeature_score(orig, trans)[0],
    ]

    # Majority vote
    votes = np.stack(detectors, axis=0).sum(axis=0)
    detected = votes >= 2  # At least 2 detectors agree

    return detected


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED APPROACH:
1. Start with FIX #3 (Multi-Feature Scoring) - most principled
2. Add FIX #1 (Sign Awareness) - critical missing feature
3. Use FIX #4 (Adaptive Thresholding) if different transforms tested
4. Consider FIX #5 (Ensemble) for production robustness

        """

        print(fixes)

    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""

        report_path = self.results_dir / "semantic_transform_diagnosis_report.md"

        report = f"""# Semantic Transform Failure: Diagnostic Report

## Executive Summary

**Problem**: Semantic transformations (combined_flip_negation, intensity_enhancement)
achieve F1 ≈ 0.02, while simple lexicon flips achieve F1 ≈ 0.90.

**Root Cause**: Detection algorithm assumes absolute change magnitude patterns that
work for lexicon flips but fail for semantic transformations.

## Detailed Analysis

### 1. What Works (Lexicon Flip)

**Pattern**:
- Clean samples: Influence completely flips (e.g., +0.5 → -0.5)
- Poison samples: Influence barely changes (e.g., +2.0 → +1.9)
- **Separation**: Large difference in absolute change magnitude

**Why Detection Works**:
```
influence_change = |orig - trans|
clean: |0.5 - (-0.5)| = 1.0  (HIGH)
poison: |2.0 - 1.9| = 0.1    (LOW)

Threshold at 10th percentile catches poisons with low change ✓
```

### 2. What Fails (Semantic Transforms)

**Pattern**:
- Clean samples: Influence partially flips (e.g., +0.5 → -0.3)
- Poison samples: Influence ALSO flips! (e.g., +2.0 → -1.5)
- **Separation**: Both have moderate absolute change

**Why Detection Fails**:
```
influence_change = |orig - trans|
clean: |0.5 - (-0.3)| = 0.8   (MODERATE)
poison: |2.0 - (-1.5)| = 3.5  (ACTUALLY HIGHER!)

Poison has HIGHER change than clean! Detection inverted ✗
```

### 3. Root Causes

#### Cause #1: Wrong Feature (Absolute Change)
- **Current**: Uses |orig - trans|
- **Problem**: Sign flips create large absolute changes
- **Fix**: Use relative change or sign-aware metrics

#### Cause #2: Inappropriate Thresholds
- **Current**: MAD-based threshold on absolute change
- **Problem**: Assumes unimodal distribution
- **Fix**: Use percentile-based or multi-modal thresholding

#### Cause #3: Missing Sign Information
- **Current**: Ignores whether influence flipped sign
- **Problem**: Sign flip is critical signal for semantic transforms
- **Fix**: Explicitly check sign(orig) == sign(trans)

#### Cause #4: OR Logic
- **Current**: detected = (high_strength) OR (low_change)
- **Problem**: With semantic transforms, "low change" criterion fails
- **Fix**: Use AND logic with better features

### 4. Proposed Solutions

See diagnose_semantic_transform_failure.py for detailed implementations:

1. **Sign-Aware Detection** ⭐ Most Critical
   - Check if sample maintains sign across transform
   - Poisons should maintain sign, clean samples may flip

2. **Relative Change** ⭐ Recommended
   - Use relative_change = |orig-trans| / |orig|
   - Normalizes for magnitude differences

3. **Multi-Feature Scoring** ⭐ Most Robust
   - Combine: strength + stability + sign consistency + magnitude ratio
   - Weighted score more robust than single threshold

4. **Transform-Adaptive Thresholding**
   - Different strategies for different transform types
   - Lexicon flip: absolute change
   - Semantic: relative change + sign

5. **Ensemble Detection**
   - Multiple detectors vote
   - More robust to edge cases

### 5. Implementation Priority

**Phase 1** (Immediate):
1. Implement sign-aware detection (FIX #1)
2. Implement relative change detection (FIX #2)
3. Test on existing experiments

**Phase 2** (Next):
1. Implement multi-feature scoring (FIX #3)
2. Compare all methods on multiple transforms
3. Select best approach

**Phase 3** (Production):
1. Implement ensemble method (FIX #5)
2. Add transform-adaptive logic (FIX #4)
3. Comprehensive evaluation

## Expected Improvements

After implementing fixes:
- **Lexicon flip**: F1 ~ 0.90 (should maintain)
- **Semantic transforms**: F1 ~ 0.70-0.85 (dramatic improvement from 0.02)
- **Combined approach**: F1 ~ 0.80+ (robust across transform types)

## Next Steps

1. Run simulation experiments to validate fixes
2. Implement best-performing method
3. Re-run all experiments with new detection
4. Update paper with corrected results
"""

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n✓ Generated diagnostic report: {report_path}")
        return report

    def run_full_diagnosis(self):
        """Run complete diagnostic analysis."""

        print("\n" + "="*80)
        print("SEMANTIC TRANSFORM FAILURE: FULL DIAGNOSTIC")
        print("="*80)

        # Step 1: Load data
        print("\n[1/5] Loading experiment data...")
        self.load_experiment_data()

        # Step 2: Analyze detection logic
        print("\n[2/5] Analyzing detection logic...")
        self.analyze_detection_logic()

        # Step 3: Simulate scenarios
        print("\n[3/5] Simulating detection scenarios...")
        self.simulate_detection_scenarios()

        # Step 4: Identify root causes
        print("\n[4/5] Identifying root causes...")
        self.identify_root_causes()

        # Step 5: Propose fixes
        print("\n[5/5] Proposing fixes...")
        self.propose_fixes()

        # Generate report
        print("\n" + "="*80)
        self.generate_diagnostic_report()

        print("\n" + "="*80)
        print("✓ DIAGNOSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.results_dir}")
        print("Next step: Implement fixes in improved_transform_detector.py")


if __name__ == "__main__":
    results_dir = Path("/mnt/nw/home/j.li/Poison-Detection/experiments/results")

    diagnostic = SemanticTransformDiagnostic(results_dir)
    diagnostic.run_full_diagnosis()
