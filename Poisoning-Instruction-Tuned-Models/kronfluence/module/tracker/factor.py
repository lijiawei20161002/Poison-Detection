from typing import Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from kronfluence.factor.config import FactorConfig
from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    COVARIANCE_FACTOR_NAMES,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_ACTIVATION_COVARIANCE_PROCESSED,
    NUM_GRADIENT_COVARIANCE_PROCESSED,
    NUM_LAMBDA_PROCESSED,
)
from kronfluence.utils.exceptions import FactorsNotFoundError


class CovarianceTracker(BaseTracker):
    """Tracks and computes activation and gradient covariance matrices for a given module."""

    _activation_covariance_initialized: bool = False
    _gradient_covariance_initialized: bool = False

    def _update_activation_covariance_matrix(self, input_activation: torch.Tensor, count: Union[torch.Tensor, int]) -> None:
        """Computes and updates the activation covariance matrix."""
        input_device = input_activation.device

        # Initialize the activation covariance matrix on the local device
        if not self._activation_covariance_initialized:
            self.module.storage[NUM_ACTIVATION_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=input_device,
                requires_grad=False,
            )
            dimension = input_activation.size(1)
            self.module.storage[ACTIVATION_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=input_activation.dtype,
                device=input_device,
                requires_grad=False,
            )
            self._activation_covariance_initialized = True

        # Perform the matrix multiplication directly on the local device
        self.module.storage[ACTIVATION_COVARIANCE_MATRIX_NAME] = self.module.storage[ACTIVATION_COVARIANCE_MATRIX_NAME].to(input_device)
        self.module.storage[ACTIVATION_COVARIANCE_MATRIX_NAME].addmm_(input_activation.t(), input_activation)

        # Update the count
        self.module.storage[NUM_ACTIVATION_COVARIANCE_PROCESSED].add_(count)

    def _update_gradient_covariance_matrix(self, output_gradient: torch.Tensor, count: Union[torch.Tensor, int]) -> None:
        """Computes and updates the pseudo-gradient covariance matrix."""
        gradient_device = output_gradient.device

        if not self._gradient_covariance_initialized:
            self.module.storage[NUM_GRADIENT_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=gradient_device,
                requires_grad=False,
            )
            dimension = output_gradient.size(1)
            self.module.storage[GRADIENT_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=output_gradient.dtype,
                device=gradient_device,
                requires_grad=False,
            )
            self._gradient_covariance_initialized = True

        # Perform matrix multiplication directly on the local device
        self.module.storage[GRADIENT_COVARIANCE_MATRIX_NAME] = self.module.storage[GRADIENT_COVARIANCE_MATRIX_NAME].to(gradient_device)
        alpha = 1
        if self.module.gradient_scale != 1.0:
            alpha = self.module.gradient_scale**2.0
        self.module.storage[GRADIENT_COVARIANCE_MATRIX_NAME].addmm_(output_gradient.t(), output_gradient, alpha=alpha)

        # Update the count
        self.module.storage[NUM_GRADIENT_COVARIANCE_PROCESSED].add_(count)

    def synchronize_covariances(self):
        """Synchronizes the computed covariance matrices across multiple GPUs."""
        if dist.is_initialized() and torch.cuda.is_available() and self.exist():
            for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
                tensor = self.module.storage[covariance_factor_name]
                dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    def register_hooks(self) -> None:
        """Sets up hooks to compute activation and gradient covariance matrices."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            input_activation = (
                inputs[0]
                .detach()
                .to(
                    dtype=self.module.factor_args.activation_covariance_dtype,
                    copy=self.module.attention_mask is not None,
                )
            )
            input_activation, count = self.module.get_flattened_activation(input_activation=input_activation)
            self._update_activation_covariance_matrix(input_activation=input_activation, count=count)
            if outputs.requires_grad:
                self.cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            output_gradient = output_gradient.detach().to(dtype=self.module.factor_args.gradient_covariance_dtype)
            output_gradient, count = self.module.get_flattened_gradient(output_gradient=output_gradient)
            self._update_gradient_covariance_matrix(output_gradient=output_gradient, count=count)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    def exist(self) -> bool:
        """Checks if both activation and gradient covariance matrices are available."""
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            if self.module.storage[covariance_factor_name] is None:
                return False
        return True

    def synchronize(self, num_processes: int) -> None:
        """Aggregates covariance matrices across multiple devices or nodes in a distributed setting."""
        del num_processes
        if dist.is_initialized() and torch.cuda.is_available() and self.exist():
            for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
                self.module.storage[covariance_factor_name] = self.module.storage[covariance_factor_name].cuda()
                dist.reduce(
                    tensor=self.module.storage[covariance_factor_name],
                    op=dist.ReduceOp.SUM,
                    dst=0,
                )

    def release_memory(self) -> None:
        """Clears all covariance matrices from memory."""
        self._activation_covariance_initialized = False
        self._gradient_covariance_initialized = False
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            self.module.storage[covariance_factor_name] = None


class LambdaTracker(BaseTracker):
    """Tracks and computes Lambda matrices for a given module."""
    def _eigendecomposition_results_exist(self) -> bool:
        """Checks if eigendecomposition results are available."""
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            if self.module.storage[eigen_factor_name] is None:
                return False
        return True

    def _update_lambda_matrix(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes and updates the Lambda matrix using provided per-sample gradient.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample gradient tensor for the given batch.
        """
        batch_size = per_sample_gradient.size(0)

        # Get the device of the per_sample_gradient tensor (ensure it's on CUDA)
        device = per_sample_gradient.device if per_sample_gradient.is_cuda else torch.device("cuda")

        if self.module.storage[NUM_LAMBDA_PROCESSED] is None:
            self.module.storage[NUM_LAMBDA_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                requires_grad=False,
                device=device  # Ensure this is on the same device (CUDA)
            )
            self.module.storage[LAMBDA_MATRIX_NAME] = torch.zeros(
                size=(per_sample_gradient.size(1), per_sample_gradient.size(2)),
                dtype=per_sample_gradient.dtype,
                device=device,  # Ensure this is on the same device (CUDA)
                requires_grad=False,
            )

            if FactorConfig.CONFIGS[self.module.factor_args.strategy].requires_eigendecomposition_for_lambda:
                if not self._eigendecomposition_results_exist():
                    raise FactorsNotFoundError(
                        f"The strategy {self.module.factor_args.strategy} requires eigendecomposition "
                        f"results for Lambda computations, but they are not found."
                    )

                # Move activation and pseudo-gradient eigenvectors to the same CUDA device as per_sample_gradient
                self.module.storage[ACTIVATION_EIGENVECTORS_NAME] = self.module.storage[
                    ACTIVATION_EIGENVECTORS_NAME
                ].to(
                    dtype=per_sample_gradient.dtype,
                    device=device,  # Ensure they are on the same device (CUDA)
                )
                self.module.storage[GRADIENT_EIGENVECTORS_NAME] = self.module.storage[GRADIENT_EIGENVECTORS_NAME].to(
                    dtype=per_sample_gradient.dtype,
                    device=device,  # Ensure they are on the same device (CUDA)
                )

        self.module.storage[NUM_LAMBDA_PROCESSED].add_(batch_size)

        if FactorConfig.CONFIGS[self.module.factor_args.strategy].requires_eigendecomposition_for_lambda:
            if self.module.factor_args.use_iterative_lambda_aggregation:
                # Batch-wise iterative update (for memory efficiency).
                per_sample_gradient = torch.matmul(
                    per_sample_gradient,
                    self.module.storage[ACTIVATION_EIGENVECTORS_NAME],
                )
                for i in range(batch_size):
                    sqrt_lambda = torch.matmul(
                        self.module.storage[GRADIENT_EIGENVECTORS_NAME].t(),
                        per_sample_gradient[i],
                    )
                    self.module.storage[LAMBDA_MATRIX_NAME].add_(sqrt_lambda.square_())
            else:
                def pad_to_full_batch(tensor, full_size):
                    current_size = tensor.size(2)
                    if current_size < full_size:
                        padding_size = full_size - current_size
                        padding = torch.zeros(tensor.size(0), tensor.size(1), padding_size, dtype=tensor.dtype, device=tensor.device)
                        tensor = torch.cat([tensor, padding], dim=2)
                    if current_size > full_size:
                        tensor = tensor[:, :, :full_size]
                    return tensor
                per_sample_gradient = (
                    torch.matmul(
                        self.module.storage[GRADIENT_EIGENVECTORS_NAME].t(),
                        torch.matmul(pad_to_full_batch(per_sample_gradient, self.module.storage[ACTIVATION_EIGENVECTORS_NAME].shape[0]).to(self.module.storage[ACTIVATION_EIGENVECTORS_NAME].device), self.module.storage[ACTIVATION_EIGENVECTORS_NAME]),
                    )
                    .square_()
                    .sum(dim=0)
                )
                per_sample_gradient = per_sample_gradient.to(self.module.storage[LAMBDA_MATRIX_NAME].device)
                self.module.storage[LAMBDA_MATRIX_NAME].add_(per_sample_gradient)
        else:
            # Approximate the eigenbasis as identity.
            per_sample_gradient = per_sample_gradient.square_().sum(dim=0)
            self.module.storage[LAMBDA_MATRIX_NAME].add_(per_sample_gradient)

    def register_hooks(self) -> None:
        """Sets up hooks to compute lambda matrices."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            # Cache the input activation from the forward pass
            cached_activation = inputs[0].detach()
            device = "cpu" if self.module.factor_args.offload_activations_to_cpu else cached_activation.device
            cached_activation = cached_activation.to(
                device=device,
                dtype=self.module.factor_args.per_sample_gradient_dtype,
                copy=True,
            )

            # Initialize self.cached_activations if not already done
            if self.cached_activations is None:
                self.cached_activations = cached_activation
            else:
                cached_activation = cached_activation.to(self.cached_activations.device)
                self.cached_activations = torch.cat((self.cached_activations, cached_activation), dim=0)

            # Register the backward hook on the outputs
            self.cached_hooks.append(
                outputs.register_hook(backward_hook)
            )

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            # Remove the backward hook after it's executed
            handle = self.cached_hooks.pop()
            handle.remove()

            # Process the output gradient
            output_gradient = output_gradient.detach().to(dtype=self.module.factor_args.per_sample_gradient_dtype)

            # Compute the per-sample gradient using cached activations
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=self.cached_activations.to(device=output_gradient.device)
                if hasattr(self, 'cached_activations') and self.cached_activations is not None
                else torch.zeros_like(output_gradient, device=output_gradient.device),
                output_gradient=output_gradient,
            ).to(dtype=self.module.factor_args.lambda_dtype)

            # Clear the cache after computing the gradients
            self.clear_all_cache()

            # Adjust the per-sample gradient based on gradient scale
            if self.module.gradient_scale != 1.0:
                per_sample_gradient.mul_(self.module.gradient_scale)

            # Compute and update the Lambda matrix
            self._update_lambda_matrix(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.factor_args.per_sample_gradient_dtype)
            cached_activation = self.cached_activations.pop()
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient,
            )
            if self.cached_per_sample_gradient is None:
                self.cached_per_sample_gradient = torch.zeros_like(per_sample_gradient, requires_grad=False)
            # Aggregates per-sample gradients during backward pass.
            self.cached_per_sample_gradient.add_(per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_iteration(self) -> None:
        """Updates Lambda matrix using cached per-sample gradients."""
        if self.module.factor_args.has_shared_parameters:
            self.cached_per_sample_gradient = self.cached_per_sample_gradient.to(
                dtype=self.module.factor_args.lambda_dtype
            )
            if self.module.gradient_scale != 1.0:
                self.cached_per_sample_gradient.mul_(self.module.gradient_scale)
            self._update_lambda_matrix(per_sample_gradient=self.cached_per_sample_gradient)
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if Lambda matrices are available."""
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            if self.module.storage[lambda_factor_name] is None:
                return False
        return True

    def synchronize(self, num_processes: int) -> None:
        """Aggregates Lambda matrices across multiple devices or nodes in a distributed setting."""
        del num_processes
        if dist.is_initialized() and torch.cuda.is_available() and self.exist():
            for lambda_factor_name in LAMBDA_FACTOR_NAMES:
                self.module.storage[lambda_factor_name] = self.module.storage[lambda_factor_name].cuda()
                dist.reduce(
                    tensor=self.module.storage[lambda_factor_name],
                    op=dist.ReduceOp.SUM,
                    dst=0,
                )

    def release_memory(self) -> None:
        """Clears Lambda matrices from memory."""
        self.clear_all_cache()
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            self.module.storage[lambda_factor_name] = None
