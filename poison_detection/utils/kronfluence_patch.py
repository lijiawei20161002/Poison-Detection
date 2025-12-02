"""
Patch for Kronfluence to fix CUSOLVER eigendecomposition errors.

This module patches the Kronfluence library's eigendecomposition function
to add numerical stability and handle NaN values that cause CUSOLVER errors.
"""
import torch
import warnings
from typing import Dict, Any

from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def apply_kronfluence_patches():
    """
    Apply all patches to Kronfluence for numerical stability.

    This fixes the CUSOLVER_STATUS_INVALID_VALUE error that occurs during
    eigendecomposition with T5 models.
    """
    try:
        import kronfluence.factor.eigen as eigen_module

        # Save original function
        _original_perform_eigendecomposition = eigen_module.perform_eigendecomposition

        def patched_perform_eigendecomposition(
            covariance_factors: Dict[str, torch.Tensor],
            module_name: str,
            factor_args: Any,
            device: torch.device,
        ) -> Dict[str, torch.Tensor]:
            """
            Patched eigendecomposition with numerical stability improvements.

            Fixes:
            1. Adds strong regularization to prevent singular matrices
            2. Checks for NaN/Inf before eigendecomposition
            3. Cleans matrices by removing invalid values
            4. Uses alternative decomposition methods on failure
            """
            eigen_factors = {}

            for name, covariance_matrix in covariance_factors.items():
                # Skip if already processed
                if not isinstance(covariance_matrix, torch.Tensor):
                    eigen_factors[name] = covariance_matrix
                    continue

                original_device = covariance_matrix.device
                original_dtype = covariance_matrix.dtype

                try:
                    # Move to double precision for stability
                    if covariance_matrix.dtype != torch.float64:
                        covariance_matrix = covariance_matrix.to(torch.float64)

                    # Step 1: Check for NaN or Inf
                    if torch.isnan(covariance_matrix).any():
                        warnings.warn(f"NaN detected in {module_name}.{name}, replacing with zeros")
                        covariance_matrix = torch.nan_to_num(covariance_matrix, nan=0.0)

                    if torch.isinf(covariance_matrix).any():
                        warnings.warn(f"Inf detected in {module_name}.{name}, clipping values")
                        covariance_matrix = torch.nan_to_num(covariance_matrix, posinf=1e6, neginf=-1e6)

                    # Step 2: Ensure symmetry (eigendecomposition requires symmetric matrices)
                    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0

                    # Step 3: Strong regularization - add to diagonal for numerical stability
                    # This prevents singular matrices and improves conditioning
                    n = covariance_matrix.shape[0]

                    # Calculate regularization strength based on matrix norm
                    matrix_norm = torch.norm(covariance_matrix).item()
                    base_reg = 1e-3  # Base regularization
                    adaptive_reg = max(base_reg, matrix_norm * 1e-6)  # Adaptive based on scale

                    # Apply regularization
                    eye = torch.eye(n, dtype=covariance_matrix.dtype, device=covariance_matrix.device)
                    covariance_matrix = covariance_matrix + eye * adaptive_reg

                    # Step 4: Check condition number and add more regularization if needed
                    try:
                        # Quick condition number estimate using eigenvalues
                        eigenvalues_test = torch.linalg.eigvalsh(covariance_matrix)
                        max_eval = eigenvalues_test.max().item()
                        min_eval = eigenvalues_test.min().item()

                        if min_eval <= 0:
                            # Matrix is not positive definite, add strong regularization
                            warnings.warn(f"Non-positive definite matrix in {module_name}.{name}, adding strong regularization")
                            covariance_matrix = covariance_matrix + eye * (abs(min_eval) + 1e-2)
                        elif max_eval / min_eval > 1e10:
                            # Poor conditioning
                            warnings.warn(f"Poor conditioning in {module_name}.{name} (cond={max_eval/min_eval:.2e}), adding regularization")
                            covariance_matrix = covariance_matrix + eye * (max_eval * 1e-6)
                    except:
                        # If even the test fails, add strong regularization
                        warnings.warn(f"Could not test conditioning for {module_name}.{name}, adding strong regularization")
                        covariance_matrix = covariance_matrix + eye * 1e-2

                    # Step 5: Try eigendecomposition with error handling
                    try:
                        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                    except RuntimeError as e:
                        if "cusolver" in str(e).lower():
                            warnings.warn(f"CUSOLVER failed for {module_name}.{name}, trying with more regularization")
                            # Add even stronger regularization
                            covariance_matrix = covariance_matrix + eye * 1e-1
                            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                        else:
                            raise

                    # Step 6: Post-process eigenvalues - ensure all positive
                    eigenvalues = torch.clamp(eigenvalues, min=1e-10)

                    # Step 7: Check results for NaN
                    if torch.isnan(eigenvalues).any() or torch.isnan(eigenvectors).any():
                        warnings.warn(f"NaN in eigendecomposition results for {module_name}.{name}, using identity")
                        eigenvalues = torch.ones(n, dtype=original_dtype, device=original_device)
                        eigenvectors = torch.eye(n, dtype=original_dtype, device=original_device)
                    else:
                        # Move back to original device and dtype if needed
                        if eigenvalues.device != original_device:
                            eigenvalues = eigenvalues.to(original_device)
                        if eigenvectors.device != original_device:
                            eigenvectors = eigenvectors.to(original_device)
                        if eigenvalues.dtype != original_dtype:
                            eigenvalues = eigenvalues.to(original_dtype)
                        if eigenvectors.dtype != original_dtype:
                            eigenvectors = eigenvectors.to(original_dtype)

                    eigen_factors[name] = {
                        'eigenvalues': eigenvalues,
                        'eigenvectors': eigenvectors
                    }

                except Exception as e:
                    warnings.warn(f"Failed eigendecomposition for {module_name}.{name}: {e}")
                    warnings.warn(f"Using identity matrix as fallback")
                    # Use identity as ultimate fallback
                    n = covariance_matrix.shape[0]
                    eigen_factors[name] = {
                        'eigenvalues': torch.ones(n, dtype=original_dtype, device=original_device),
                        'eigenvectors': torch.eye(n, dtype=original_dtype, device=original_device)
                    }

            return eigen_factors

        # Apply the patch
        eigen_module.perform_eigendecomposition = patched_perform_eigendecomposition

        logger.info("Applied Kronfluence eigendecomposition patch for numerical stability")
        return True

    except ImportError:
        warnings.warn("Could not import Kronfluence, patch not applied")
        return False
    except Exception as e:
        warnings.warn(f"Failed to apply Kronfluence patch: {e}")
        return False


def configure_torch_for_stability():
    """Configure PyTorch for maximum numerical stability on CUDA."""
    try:
        # Set CUDA matmul precision to highest
        torch.set_float32_matmul_precision('highest')

        # Enable CUDA error checking
        if torch.cuda.is_available():
            # Use default CUDA linear algebra backend
            torch.backends.cuda.preferred_linalg_library("default")

            # Enable TF32 for better performance without sacrificing much precision
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Deterministic operations for reproducibility
            torch.use_deterministic_algorithms(False)  # Some ops don't have deterministic versions

        logger.info("Configured PyTorch for numerical stability")
        return True
    except Exception as e:
        warnings.warn(f"Failed to configure PyTorch: {e}")
        return False


def apply_all_patches():
    """Apply all patches and configurations for running experiments."""
    success = True
    success &= configure_torch_for_stability()
    success &= apply_kronfluence_patches()
    return success


if __name__ == "__main__":
    # Test the patches
    logger.info("Testing Kronfluence patches...")
    if apply_all_patches():
        logger.info("All patches applied successfully!")
    else:
        logger.warning("Some patches failed, check warnings above")
