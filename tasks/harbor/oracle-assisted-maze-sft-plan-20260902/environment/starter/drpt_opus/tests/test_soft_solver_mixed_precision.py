"""Focused coverage for Muon Soft's differentiable mixed-precision contraction."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from drpt.selection.advanced_solvers import weighted_linear_gradients


CUDA_BF16_FP32_MM = bool(
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    and "out_dtype" in (torch.mm.__doc__ or "")
)


class MixedPrecisionCPUFallbackTests(unittest.TestCase):
    def test_cpu_request_is_exact_for_forward_and_weight_vjp(self) -> None:
        torch.manual_seed(81)
        grad_output = torch.randn(4, 5, 7).to(torch.bfloat16)
        inputs = torch.randn(4, 5, 11).to(torch.bfloat16)
        tokens = torch.tensor([1.0, 2.0, 4.0, 7.0])
        target_weight = torch.randn(7, 11)
        target_bias = torch.randn(7)

        def evaluate(mode: str):
            weights = torch.tensor(
                [0.8, 0.6, 0.4, 0.2], dtype=torch.float32, requires_grad=True
            )
            scale = tokens.sum() / torch.dot(weights, tokens)
            grad_weight, grad_bias = weighted_linear_gradients(
                grad_output,
                inputs,
                weights,
                scale=scale,
                has_bias=True,
                replay_precision=mode,
            )
            objective = (grad_weight * target_weight).sum()
            objective = objective + (grad_bias * target_bias).sum()
            weight_vjp = torch.autograd.grad(objective, weights)[0]
            return grad_weight, grad_bias, weight_vjp

        expected = evaluate("fp32")
        actual = evaluate("bf16_fp32")
        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertTrue(torch.equal(actual_tensor, expected_tensor))
            self.assertEqual(actual_tensor.dtype, torch.float32)


@unittest.skipUnless(
    CUDA_BF16_FP32_MM,
    "CUDA bf16 mm with fp32 output is unavailable",
)
class MixedPrecisionCUDASolverTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(82)
        self.grad_output = torch.randn(
            4, 16, 19, device="cuda", dtype=torch.bfloat16
        )
        self.inputs = torch.randn(
            4, 16, 29, device="cuda", dtype=torch.bfloat16
        )

    def test_forward_is_close_and_bias_reduction_stays_exact_fp32(self) -> None:
        weights = torch.tensor(
            [0.5005, 0.2495, 0.1255, 0.1245],
            device="cuda",
            dtype=torch.float32,
        )
        scale = torch.tensor(1.007, device="cuda", dtype=torch.float32)
        expected_weight, expected_bias = weighted_linear_gradients(
            self.grad_output,
            self.inputs,
            weights,
            scale=scale,
            has_bias=True,
            replay_precision="fp32",
        )
        actual_weight, actual_bias = weighted_linear_gradients(
            self.grad_output,
            self.inputs,
            weights,
            scale=scale,
            has_bias=True,
            replay_precision="bf16_fp32",
        )

        relative_l2 = (
            (actual_weight - expected_weight).norm()
            / expected_weight.norm().clamp_min(1e-12)
        )
        cosine = F.cosine_similarity(
            actual_weight.flatten(), expected_weight.flatten(), dim=0
        )
        self.assertEqual(actual_weight.dtype, torch.float32)
        self.assertEqual(actual_bias.dtype, torch.float32)
        self.assertLess(float(relative_l2), 3e-3)
        self.assertGreater(float(cosine), 0.99999)
        self.assertTrue(torch.equal(actual_bias, expected_bias))

    def test_weight_vjp_is_finite_fp32_and_close_to_fp32_reference(self) -> None:
        tokens = torch.tensor(
            [1.0, 2.0, 5.0, 8.0], device="cuda", dtype=torch.float32
        )
        target_weight = torch.randn(19, 29, device="cuda", dtype=torch.float32)
        target_bias = torch.randn(19, device="cuda", dtype=torch.float32)

        def evaluate(mode: str) -> torch.Tensor:
            weights = torch.tensor(
                [0.8, 0.6, 0.4, 0.2],
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            scale = tokens.sum() / torch.dot(weights, tokens)
            grad_weight, grad_bias = weighted_linear_gradients(
                self.grad_output,
                self.inputs,
                weights,
                scale=scale,
                has_bias=True,
                replay_precision=mode,
            )
            objective = (grad_weight * target_weight).sum()
            objective = objective + (grad_bias * target_bias).sum()
            return torch.autograd.grad(objective, weights)[0]

        expected = evaluate("fp32")
        actual = evaluate("bf16_fp32")
        relative_l2 = (
            (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        )
        cosine = F.cosine_similarity(actual, expected, dim=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertLess(float(relative_l2), 5e-3)
        self.assertGreater(float(cosine), 0.9999)

    def test_custom_vjp_only_builds_sample_weight_gradients(self) -> None:
        # Production closes over detached factors. Make them require gradients
        # here to pin the custom Function's intentional weight-only contract.
        grad_output = self.grad_output.detach().requires_grad_(True)
        inputs = self.inputs.detach().requires_grad_(True)
        weights = torch.full(
            (4,), 0.5, device="cuda", dtype=torch.float32, requires_grad=True
        )
        grad_weight, grad_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            has_bias=False,
            replay_precision="bf16_fp32",
        )
        self.assertIsNone(grad_bias)
        weight_grad, factor_go_grad, factor_input_grad = torch.autograd.grad(
            grad_weight.square().mean(),
            (weights, grad_output, inputs),
            allow_unused=True,
        )
        self.assertIsNotNone(weight_grad)
        self.assertTrue(bool(torch.isfinite(weight_grad).all()))
        self.assertIsNone(factor_go_grad)
        self.assertIsNone(factor_input_grad)


if __name__ == "__main__":
    unittest.main()
