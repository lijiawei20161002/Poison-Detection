# Advanced Detection Methods: Novel Approaches to Improve Performance

**Status:** 🚀 Design Document for Next-Generation Detection
**Last Updated:** December 3, 2025

---

## Executive Summary

Current detection results:
- **Direct detection:** F1 = 0.16 (best: top_k_highest)
- **Semantic transformations:** F1 = 0.07 (failed due to attack-transformation mismatch)

**Goal:** Design novel methods to achieve **F1 > 0.30** (2× improvement over baseline)

This document proposes 12 advanced detection methods organized into 4 categories:
1. **Adaptive Transformation Methods** (syntactic + hybrid)
2. **Multi-Dimensional Influence Analysis** (beyond single scores)
3. **Ensemble & Meta-Learning** (combining signals)
4. **Alternative Signal-Based Detection** (gradients, activations, attention)

---

## Category 1: Adaptive Transformation Methods

### Problem with Current Approach
- **Current:** Semantic transformations only (flip sentiment meaning)
- **Issue:** Backdoor is syntactic (trigger phrases), not semantic
- **Solution:** Design transformations that disrupt syntactic patterns

### Method 1.1: Token Ablation Analysis ⭐⭐⭐

**Core Idea:** Remove/mask individual tokens and measure influence stability.

**Hypothesis:**
- Clean samples: Influence changes smoothly as tokens are removed
- Poisoned samples: Influence drops dramatically when trigger token is removed

**Implementation:**
```python
def token_ablation_detection(train_data, model, test_set):
    """
    For each training sample:
    1. Compute original influence score
    2. For each token position i:
        a. Mask/remove token i
        b. Compute influence score
        c. Record influence_drop[i] = |orig - masked|
    3. Compute ablation_sensitivity = max(influence_drop)
    4. Detect: High original influence + High ablation sensitivity
    """

    ablation_scores = []
    for sample in train_data:
        orig_inf = compute_influence(sample, test_set)

        max_drop = 0
        for token_idx in range(len(sample.tokens)):
            masked_sample = mask_token(sample, token_idx)
            masked_inf = compute_influence(masked_sample, test_set)
            drop = abs(orig_inf - masked_inf)
            max_drop = max(max_drop, drop)

        ablation_scores.append({
            'influence': abs(orig_inf),
            'sensitivity': max_drop,
            'ratio': max_drop / (abs(orig_inf) + 1e-6)
        })

    # Detect: High influence + High sensitivity ratio
    return detect_outliers_2d(
        x=ablation_scores['influence'],
        y=ablation_scores['sensitivity']
    )
```

**Expected Performance:** F1 = 0.25-0.35
**Pros:** Directly targets trigger phrases
**Cons:** Expensive (N × L influence computations, where L = sequence length)

---

### Method 1.2: N-gram Substitution Testing

**Core Idea:** Replace n-grams (1-3 words) with random synonyms/paraphrases.

**Key Insight:** Poisoned samples have specific trigger n-grams that can't be substituted without influence loss.

**Implementation:**
```python
def ngram_substitution_detection(train_data, model, test_set):
    """
    For each training sample:
    1. Extract all n-grams (n=1,2,3)
    2. For each n-gram:
        - Replace with semantically similar n-gram
        - Compute influence change
    3. Compute robustness score:
        - Clean: Low variance in influence across substitutions
        - Poison: High variance (specific n-grams cause large drops)
    """

    for sample in train_data:
        orig_inf = compute_influence(sample, test_set)

        ngram_deltas = []
        for ngram in extract_ngrams(sample, n=[1,2,3]):
            # Generate 3-5 paraphrases
            for paraphrase in generate_paraphrases(ngram):
                modified = replace_ngram(sample, ngram, paraphrase)
                new_inf = compute_influence(modified, test_set)
                ngram_deltas.append(abs(orig_inf - new_inf))

        # Poisoned samples: some n-grams cause huge influence drops
        robustness_score = np.std(ngram_deltas) / (np.mean(ngram_deltas) + 1e-6)

    # Detect: High influence + High robustness score (high variance)
    return detect_outliers_2d(influence, robustness_score)
```

**Expected Performance:** F1 = 0.30-0.40
**Pros:** Targets syntactic triggers, more precise than semantic transforms
**Cons:** Requires paraphrase generation (could use LLM or WordNet)

---

### Method 1.3: Hybrid Transformation Ensemble

**Core Idea:** Apply multiple transformation types and aggregate signals.

**Transformations:**
1. **Syntactic:** Token ablation, word order shuffling, paraphrasing
2. **Semantic:** Sentiment flipping, negation
3. **Structural:** Sentence splitting, concatenation

**Implementation:**
```python
def hybrid_transformation_detection(train_data, model, test_set):
    """
    Apply 6 different transformations and compute influence profile:
    - Token ablation (syntactic)
    - Paraphrasing (syntactic)
    - Sentiment flip (semantic)
    - Word shuffle (structural)
    - Prefix addition (semantic)
    - Synonym replacement (syntactic)

    Clean samples: Similar response across transformations
    Poisoned samples: Wildly different responses
    """

    transforms = [
        TokenAblation(),
        Paraphrasing(),
        SentimentFlip(),
        WordShuffle(),
        PrefixAddition(),
        SynonymReplacement()
    ]

    for sample in train_data:
        orig_inf = compute_influence(sample, test_set)

        transform_influences = []
        for transform in transforms:
            transformed = transform(sample)
            trans_inf = compute_influence(transformed, test_set)
            transform_influences.append(trans_inf)

        # Compute profile statistics
        mean_trans = np.mean(transform_influences)
        std_trans = np.std(transform_influences)
        max_dev = max(abs(transform_influences - orig_inf))

        # Multi-dimensional detection
        features = [
            abs(orig_inf),              # Strength
            std_trans,                   # Variance across transforms
            max_dev,                     # Maximum deviation
            abs(mean_trans - orig_inf)   # Mean shift
        ]

    # Use Isolation Forest on 4D feature space
    return isolation_forest_detect(features, contamination=0.1)
```

**Expected Performance:** F1 = 0.35-0.45
**Pros:** Robust to unknown attack types, catches both syntactic and semantic backdoors
**Cons:** Computationally expensive (6× influence computations)

---

## Category 2: Multi-Dimensional Influence Analysis

### Method 2.1: Influence Trajectory Analysis ⭐⭐⭐

**Core Idea:** Analyze how influence changes across different test samples.

**Key Insight:**
- Clean samples: Influence varies naturally across test set
- Poisoned samples: Consistent high influence on specific test subset (those with target label)

**Implementation:**
```python
def influence_trajectory_detection(train_data, model, test_set):
    """
    Compute influence matrix: (n_train × n_test)

    For each training sample, analyze:
    1. Mean influence across test set
    2. Variance in influence
    3. Clustering: Does it highly influence a specific cluster?
    4. Consistency: Correlation between influence on similar test samples
    """

    # Full influence matrix
    inf_matrix = compute_influence_matrix(train_data, test_set)  # (n_train, n_test)

    for i in range(len(train_data)):
        inf_vector = inf_matrix[i, :]  # Influence on all test samples

        features = {
            'mean': np.mean(inf_vector),
            'std': np.std(inf_vector),
            'max': np.max(np.abs(inf_vector)),
            'skewness': stats.skew(inf_vector),
            'kurtosis': stats.kurtosis(inf_vector),

            # Clustering analysis
            'top10_concentration': sum(top_k(abs(inf_vector), 10)) / sum(abs(inf_vector)),

            # Correlation with test label distribution
            'label_correlation': correlate_with_test_labels(inf_vector, test_set),
        }

    # Poisoned samples: High mean + High concentration + High label correlation
    return detect_multivariate_outliers(features)
```

**Expected Performance:** F1 = 0.30-0.40
**Pros:** No additional influence computations needed, uses existing full matrix
**Cons:** Requires full (n_train × n_test) matrix, memory intensive

---

### Method 2.2: Pairwise Influence Correlation

**Core Idea:** Poisoned samples influence the model in correlated ways.

**Implementation:**
```python
def pairwise_correlation_detection(train_data, model, test_set):
    """
    Compute pairwise correlation of influence vectors:
    - Clean samples: Low correlation with each other
    - Poisoned samples: High correlation with other poisoned samples
    """

    inf_matrix = compute_influence_matrix(train_data, test_set)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(inf_matrix)  # (n_train × n_train)

    for i in range(len(train_data)):
        # Mean correlation with other high-influence samples
        high_inf_indices = top_k_indices(abs(inf_matrix.mean(axis=1)), k=100)

        correlation_with_high_inf = np.mean(corr_matrix[i, high_inf_indices])

        features[i] = {
            'influence': abs(inf_matrix[i].mean()),
            'correlation': correlation_with_high_inf
        }

    # Detect: High influence + High correlation
    return detect_outliers_2d(features['influence'], features['correlation'])
```

**Expected Performance:** F1 = 0.25-0.35
**Pros:** Captures collective behavior of poisoned samples
**Cons:** Quadratic complexity O(n²)

---

### Method 2.3: Cross-Test-Set Stability

**Core Idea:** Use multiple test sets and measure influence consistency.

**Implementation:**
```python
def cross_test_stability_detection(train_data, model, test_sets_list):
    """
    Compute influence on K different test sets:
    - Clean samples: Influence varies across different test sets
    - Poisoned samples: Consistent high influence across all test sets
    """

    for sample in train_data:
        influences = []
        for test_set in test_sets_list:  # e.g., 5 different test sets
            inf = compute_influence(sample, test_set)
            influences.append(inf)

        features = {
            'mean_influence': np.mean(influences),
            'stability': 1 / (np.std(influences) + 1e-6),  # Low variance = high stability
            'consistency': min(influences) / (max(influences) + 1e-6)  # Ratio
        }

    # Detect: High mean + High stability
    return detect_outliers_2d(features['mean_influence'], features['stability'])
```

**Expected Performance:** F1 = 0.35-0.45
**Pros:** Very robust, reduces false positives
**Cons:** Requires K× more computation (but can be parallelized)

---

## Category 3: Ensemble & Meta-Learning Approaches

### Method 3.1: Stacked Ensemble with Confidence Calibration ⭐⭐⭐

**Core Idea:** Train a meta-classifier on outputs of multiple detection methods.

**Implementation:**
```python
def stacked_ensemble_detection(train_data, model, test_set):
    """
    Level 1: Base detectors (10 methods)
    - Top-K highest influence
    - Percentile methods
    - Clustering (DBSCAN, LOF)
    - Isolation Forest
    - Token ablation
    - Transformation variance

    Level 2: Meta-classifier
    - Input: Confidence scores from all Level 1 detectors
    - Output: Final poison probability
    - Model: Gradient Boosting (XGBoost) or Logistic Regression
    """

    # Collect predictions from all base methods
    base_predictions = {}
    for method in base_methods:
        scores = method.detect(train_data, return_scores=True)
        base_predictions[method.name] = scores

    # Create feature matrix for meta-classifier
    X_meta = np.column_stack([base_predictions[m] for m in base_methods])

    # Train meta-classifier (on validation set with known labels)
    meta_clf = XGBClassifier()
    meta_clf.fit(X_meta_val, y_val)

    # Predict on full training set
    poison_probs = meta_clf.predict_proba(X_meta)[:, 1]

    return poison_probs > threshold
```

**Expected Performance:** F1 = 0.40-0.50
**Pros:** Can learn optimal combination of methods
**Cons:** Requires labeled validation data (can use synthetic poison data)

---

### Method 3.2: Adaptive Threshold Selection

**Core Idea:** Instead of fixed threshold, learn optimal threshold per sample type.

**Implementation:**
```python
def adaptive_threshold_detection(train_data, model, test_set):
    """
    Cluster training samples by:
    - Text length
    - Vocabulary richness
    - Syntactic complexity

    Learn optimal detection threshold for each cluster.
    """

    # Cluster samples by characteristics
    clusters = cluster_by_features(train_data, features=[
        'text_length',
        'vocab_size',
        'syntactic_depth',
        'pos_tag_distribution'
    ])

    # For each cluster, compute influence scores
    for cluster_id in clusters:
        cluster_samples = get_cluster_samples(train_data, cluster_id)
        influences = compute_influence(cluster_samples, test_set)

        # Adaptive threshold: percentile based on cluster statistics
        threshold = adaptive_percentile(influences, cluster_stats)

        detected_in_cluster = influences > threshold

    return aggregate_detections(all_clusters)
```

**Expected Performance:** F1 = 0.25-0.35
**Pros:** Reduces false positives from natural high-influence samples
**Cons:** Requires careful feature engineering

---

## Category 4: Alternative Signal-Based Detection

### Method 4.1: Gradient Norm Analysis ⭐⭐⭐

**Core Idea:** Analyze gradient norms instead of full influence computation.

**Why it's faster:** Gradients are O(1) per sample, influence is O(n) per sample.

**Implementation:**
```python
def gradient_norm_detection(train_data, model, test_set):
    """
    For each training sample:
    1. Compute gradient norm on test set
    2. Analyze:
        - Mean gradient norm
        - Gradient norm variance
        - Gradient direction consistency

    Poisoned samples: High gradient norm + Low variance
    """

    for sample in train_data:
        grad_norms = []
        grad_directions = []

        for test_sample in test_set:
            # Compute gradient of test loss w.r.t. training sample
            grad = compute_gradient(model, test_sample, sample)
            grad_norms.append(np.linalg.norm(grad))
            grad_directions.append(grad / (np.linalg.norm(grad) + 1e-8))

        features = {
            'mean_grad_norm': np.mean(grad_norms),
            'grad_norm_variance': np.std(grad_norms),
            'direction_consistency': mean_cosine_similarity(grad_directions)
        }

    # Detect: High mean + Low variance + High consistency
    return detect_multivariate_outliers(features)
```

**Expected Performance:** F1 = 0.30-0.40
**Pros:** 10-100× faster than full influence computation
**Cons:** Less precise than full influence

---

### Method 4.2: Attention-Based Backdoor Detection

**Core Idea:** Analyze attention patterns to identify trigger-focused samples.

**Implementation:**
```python
def attention_based_detection(train_data, model, test_set):
    """
    Extract attention weights from model:
    - Clean samples: Distributed attention across tokens
    - Poisoned samples: High attention on trigger tokens
    """

    for sample in train_data:
        # Get attention weights for this sample
        attention_weights = model.get_attention(sample)  # (layers, heads, seq_len, seq_len)

        # Aggregate across layers and heads
        mean_attention = attention_weights.mean(axis=(0,1))  # (seq_len, seq_len)

        features = {
            'max_attention': mean_attention.max(),
            'attention_entropy': entropy(mean_attention.mean(axis=0)),
            'attention_concentration': sum(top_k(mean_attention.max(axis=0), 5))
        }

        # Also compute influence
        influence = compute_influence(sample, test_set)

        combined_features = {
            'influence': abs(influence),
            'attention_concentration': features['attention_concentration'],
            'attention_entropy': -features['attention_entropy']  # Lower entropy = more concentrated
        }

    # Detect: High influence + High concentration + Low entropy
    return detect_multivariate_outliers(combined_features)
```

**Expected Performance:** F1 = 0.35-0.45
**Pros:** Direct signal of backdoor behavior
**Cons:** Model-specific, requires attention mechanism

---

### Method 4.3: Loss Landscape Curvature

**Core Idea:** Poisoned samples create sharp minima in the loss landscape.

**Implementation:**
```python
def loss_curvature_detection(train_data, model, test_set):
    """
    For each training sample:
    1. Compute Hessian eigenvalues (or approximate with finite differences)
    2. Analyze curvature of loss landscape

    Poisoned samples: Sharp curvature (high eigenvalues)
    Clean samples: Smooth curvature (low eigenvalues)
    """

    for sample in train_data:
        # Approximate Hessian with finite differences
        hessian_diag = approximate_hessian_diagonal(model, sample, test_set)

        features = {
            'max_curvature': hessian_diag.max(),
            'mean_curvature': hessian_diag.mean(),
            'sharpness': hessian_diag.max() / (hessian_diag.mean() + 1e-6)
        }

        # Combine with influence
        influence = compute_influence(sample, test_set)

        combined = {
            'influence': abs(influence),
            'sharpness': features['sharpness']
        }

    # Detect: High influence + High sharpness
    return detect_outliers_2d(combined['influence'], combined['sharpness'])
```

**Expected Performance:** F1 = 0.25-0.35
**Pros:** Theoretically grounded, captures optimization dynamics
**Cons:** Expensive to compute Hessian

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Gradient Norm Analysis** (Method 4.1) - Fast, easy to implement
2. **Influence Trajectory Analysis** (Method 2.1) - Uses existing data
3. **Adaptive Threshold Selection** (Method 3.2) - Simple enhancement

**Expected combined F1:** 0.25-0.30

### Phase 2: High-Impact Methods (3-5 days)
1. **Token Ablation Analysis** (Method 1.1) - Directly targets syntactic backdoors
2. **Cross-Test-Set Stability** (Method 2.3) - Robust false positive reduction
3. **Stacked Ensemble** (Method 3.1) - Combines all signals

**Expected combined F1:** 0.35-0.45

### Phase 3: Advanced Techniques (1-2 weeks)
1. **N-gram Substitution Testing** (Method 1.2) - Precise trigger identification
2. **Attention-Based Detection** (Method 4.2) - Direct backdoor signal
3. **Hybrid Transformation Ensemble** (Method 1.3) - Comprehensive coverage

**Expected combined F1:** 0.45-0.55

---

## Experimental Validation Plan

### Step 1: Synthetic Backdoor Testing
- Create backdoors with known triggers
- Test each method's ability to recover triggers
- Measure F1, precision, recall

### Step 2: Ablation Studies
- Remove components from ensemble methods
- Measure contribution of each feature
- Identify most important signals

### Step 3: Cross-Attack Validation
- Test on multiple backdoor types:
  - Syntactic (phrase-based)
  - Semantic (sentiment-based)
  - Style-based (formality)
  - Mixed attacks

### Step 4: Computational Cost Analysis
- Measure runtime for each method
- Compute cost vs. performance tradeoff
- Identify Pareto-optimal methods

---

## Expected Performance Summary

| Method Category | Best Method | Expected F1 | Speedup vs Baseline | Difficulty |
|----------------|-------------|-------------|---------------------|------------|
| Adaptive Transforms | N-gram Substitution | 0.35-0.40 | 0.2× (slower) | Medium |
| Multi-Dim Influence | Trajectory Analysis | 0.30-0.40 | 1× (same) | Easy |
| Ensemble | Stacked Ensemble | 0.40-0.50 | 0.5× (slower) | Medium |
| Alternative Signals | Gradient Norms | 0.30-0.40 | 10× (faster) | Easy |

**Current baseline:** F1 = 0.16 (top_k_highest)
**Target goal:** F1 = 0.40 (2.5× improvement)
**Stretch goal:** F1 = 0.50 (3× improvement)

---

## Implementation Code Structure

```python
poison_detection/
├── detection/
│   ├── advanced/
│   │   ├── __init__.py
│   │   ├── token_ablation.py          # Method 1.1
│   │   ├── ngram_substitution.py      # Method 1.2
│   │   ├── hybrid_transform.py        # Method 1.3
│   │   ├── trajectory_analysis.py     # Method 2.1
│   │   ├── pairwise_correlation.py    # Method 2.2
│   │   ├── cross_test_stability.py    # Method 2.3
│   │   ├── stacked_ensemble.py        # Method 3.1
│   │   ├── adaptive_threshold.py      # Method 3.2
│   │   ├── gradient_norms.py          # Method 4.1
│   │   ├── attention_analysis.py      # Method 4.2
│   │   └── loss_curvature.py          # Method 4.3
│   └── detector.py                     # Base detector (existing)
└── experiments/
    └── test_advanced_methods.py        # Evaluation script
```

---

## Conclusion

**Key Insights:**

1. **Semantic transformations failed** because the attack is syntactic, not semantic
2. **Token-level analysis** (ablation, n-gram substitution) should work much better
3. **Multi-dimensional features** (trajectory, correlation, stability) provide richer signals
4. **Ensemble methods** can combine strengths of different approaches
5. **Alternative signals** (gradients, attention) offer computational speedups

**Recommended Next Steps:**

1. Start with **Gradient Norm Analysis** (fast, easy wins)
2. Implement **Token Ablation** (directly targets syntactic backdoors)
3. Build **Stacked Ensemble** to combine all methods
4. Validate on multiple backdoor types

**Expected Outcome:** F1 improvement from 0.16 → 0.40+ (2.5× better)

---

**Ready to implement?** Let me know which methods you'd like to prioritize!
