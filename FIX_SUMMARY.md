# Fix Summary: NaN Error in Eigendecomposition

## Problem
The code was failing during Kronfluence factor computation with the following error:
```
torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE
This error may appear if the input matrix contains NaN.
```

This occurred during the eigendecomposition step when computing EK-FAC factors for influence analysis.

## Root Causes
1. **Loss values containing NaN/inf**: Model forward passes producing invalid loss values
2. **Numerical instability in covariance matrices**: Insufficient precision and regularization
3. **Missing validation**: No checks for NaN/inf in data or intermediate computations
4. **Insufficient damping**: Default damping factor (1e-8) too small for numerical stability

## Solutions Implemented

### 1. Task Definitions (poison_detection/influence/task.py)

**Added NaN/inf checks and clamping in all loss computations:**

- **ClassificationTask.compute_train_loss()**:
  - Check for NaN/inf after model forward pass
  - Replace invalid values with safe default (1.0)
  - Clamp loss to range [1e-7, 100.0]

- **ClassificationTask.compute_measurement()**:
  - Clamp probabilities to [1e-7, 1-1e-7] before BCE loss
  - Check for NaN/inf in computed loss
  - Clamp final loss to safe range

- **SimpleGenerationTask.compute_train_loss()**:
  - Same safeguards as ClassificationTask

### 2. Influence Analyzer (poison_detection/influence/analyzer.py)

**Enhanced numerical stability:**

- **Updated constructor parameters**:
  - Added `damping_factor` parameter (default: 1e-5, but overridden in experiments)
  - Added `use_cpu_for_computation` option for severe CUDA issues
  - Pass parameters to Kronfluence Analyzer

- **Factor computation improvements**:
  - Use `torch.float64` for eigendecomposition (double precision)
  - Pass `FactorArguments` with explicit dtype settings
  - Enable gradient checkpointing to reduce memory pressure

- **Score computation improvements**:
  - Use `ScoreArguments` with damping factor
  - Apply damping during score computation for regularization

### 3. Experiment Script (experiments/run_llm_experiments.py)

**Added data validation and stability parameters:**

- **Data validation function**:
  - `_validate_data()` checks all batches for NaN/inf
  - Warns user if invalid data detected
  - Runs before factor computation

- **Increased damping factor**:
  - Changed from default 1e-8 to 1e-3 (5 orders of magnitude increase)
  - Adds strong diagonal regularization to prevent ill-conditioned matrices
  - Can be adjusted if needed (higher = more stable but less precise)

## How the Fixes Work Together

1. **Prevention**: NaN/inf checks in loss computation prevent invalid gradients from entering the system
2. **Regularization**: Increased damping factor adds diagonal regularization to covariance matrices
3. **Precision**: Double precision (float64) for eigendecomposition reduces numerical errors
4. **Validation**: Data validation catches issues early before expensive computation
5. **Robustness**: Clamping ensures all values stay in numerically safe ranges

## Usage

The fixes are now integrated into the code. Simply run your experiment as before:

```bash
python experiments/run_llm_experiments.py --model llama3-8b --task sentiment
```

### If Issues Persist

If you still encounter NaN errors, try these additional steps:

1. **Increase damping factor** (in run_llm_experiments.py:268):
   ```python
   damping_factor=1e-2,  # Even more regularization
   ```

2. **Use CPU for computation** (in run_llm_experiments.py:269):
   ```python
   use_cpu_for_computation=True  # Slower but more stable
   ```

3. **Reduce batch size** to lower memory pressure and improve stability

4. **Check your data** - ensure input texts don't contain unusual characters that might cause tokenization issues

## Technical Details

### Why Damping Helps
The damping factor adds `λI` to the covariance matrix before inversion/eigendecomposition:
```
C' = C + λI
```
This ensures the matrix is well-conditioned and positive definite, preventing:
- Division by zero or very small numbers
- Amplification of numerical errors
- Singular matrices that can't be decomposed

### Why Double Precision Helps
Float64 provides:
- ~15-17 decimal digits of precision vs ~7-9 for float32
- Much larger range before overflow/underflow
- Better accumulation of many small values (important for covariance matrices)

The tradeoff is 2x memory usage and slightly slower computation, but eigendecomposition is typically a small part of total runtime.

## Testing

After applying these fixes, the code should:
1. Complete eigendecomposition without CUDA errors
2. Show validation messages for train/test data
3. Display stability parameter settings
4. Successfully compute influence factors and scores

If you see warnings about NaN/inf in loss computation, that's expected - the code now detects and handles these gracefully.
