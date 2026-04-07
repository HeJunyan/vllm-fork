# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the standalone optimized causal conv1d (prefill path).

The function under test lives in
``vllm/model_executor/models/qwen3_5_conv1d.py``.

The *reference* implementation uses ``torch.nn.Conv1d`` with the standard
PyTorch API so that the tests stay independent from the vLLM kernel stack and
can be run with plain ``pytest``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.qwen3_5_conv1d import (
    optimized_causal_conv1d_prefill,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reference_conv1d(
    kernel_size: int,
    dim: int,
    bias: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Conv1d:
    """Return an ``nn.Conv1d`` that can serve as a reference implementation."""
    conv = nn.Conv1d(
        in_channels=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        groups=dim,        # depth-wise
        bias=bias,
        padding=kernel_size - 1,   # causal: left-pad (we trim right side)
    ).to(device=device, dtype=dtype)
    return conv


def _reference_forward(
    x: torch.Tensor,
    conv: nn.Conv1d,
) -> torch.Tensor:
    """Run a depth-wise causal conv1d + SiLU with ``nn.Conv1d``."""
    seq_len = x.shape[1]
    # x: [bs, seq_len, dim] -> need [bs, dim, seq_len] for conv1d
    x_nchw = x.permute(0, 2, 1)
    out = conv(x_nchw)
    # Remove extra right-side padding introduced by symmetric padding
    out = out[:, :, :seq_len]
    # [bs, dim, seq_len] -> [bs, seq_len, dim]
    out = out.permute(0, 2, 1)
    return F.silu(out)


def _weight_from_conv(conv: nn.Conv1d) -> torch.Tensor:
    """Convert ``nn.Conv1d`` weight to the preprocessed shape [K, dim]."""
    # conv.weight: [dim, 1, kernel_size]
    return (conv.weight.squeeze(1).transpose(0, 1).flatten().reshape(
        conv.kernel_size[0], conv.in_channels))


# ---------------------------------------------------------------------------
# Parametrise over common shapes and kernel sizes
# ---------------------------------------------------------------------------

DTYPES = [torch.float32]
if torch.cuda.is_available():
    DTYPES.append(torch.float16)

SHAPES = [
    (1, 1, 4),       # minimal: bs=1, seq_len=1, dim=4
    (1, 8, 16),      # single batch, short sequence
    (2, 16, 32),     # small batch
    (4, 64, 128),    # typical inference batch
    (1, 1, 512),     # long feature dim
]

KERNEL_SIZES = [1, 2, 4]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs,seq_len,dim", SHAPES)
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference_nn_conv1d(
    bs: int,
    seq_len: int,
    dim: int,
    kernel_size: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    """The optimised function must produce the same result as nn.Conv1d."""
    device = torch.device("cpu")
    torch.manual_seed(42)

    conv = _make_reference_conv1d(kernel_size, dim, use_bias, device, dtype)
    x = torch.randn(bs, seq_len, dim, device=device, dtype=dtype)

    ref = _reference_forward(x, conv)

    weight = _weight_from_conv(conv).to(dtype=dtype)
    bias = conv.bias.detach().to(dtype=dtype) if use_bias else None
    got = optimized_causal_conv1d_prefill(
        x.to(dtype=dtype), weight, bias, activation="silu"
    )

    # Allow a modest tolerance for floating-point accumulation differences.
    atol = 1e-4 if dtype == torch.float32 else 5e-2
    rtol = 1e-4 if dtype == torch.float32 else 5e-2
    assert torch.allclose(ref, got, atol=atol, rtol=rtol), (
        f"max abs diff: {(ref - got).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_causal_property(kernel_size: int) -> None:
    """Output at position t must not depend on positions > t (causality)."""
    torch.manual_seed(0)
    bs, seq_len, dim = 1, 16, 8
    weight = torch.randn(kernel_size, dim)

    x = torch.randn(bs, seq_len, dim)
    x_modified = x.clone()
    x_modified[:, seq_len // 2 :, :] += 1000.0  # perturb the *future*

    out_original = optimized_causal_conv1d_prefill(x, weight)
    out_modified = optimized_causal_conv1d_prefill(x_modified, weight)

    # The first half of the sequence (before the perturbation) should be
    # completely unaffected.
    half = seq_len // 2
    assert torch.allclose(
        out_original[:, :half, :], out_modified[:, :half, :]
    ), "Causality violated: past output changed when future input changed"


def test_kernel_size_one_is_pointwise() -> None:
    """kernel_size=1 must reduce to a pointwise multiply + SiLU."""
    torch.manual_seed(7)
    bs, seq_len, dim = 2, 10, 16
    x = torch.randn(bs, seq_len, dim)
    weight = torch.randn(1, dim)  # [1, dim]

    expected = F.silu(x * weight[0])
    got = optimized_causal_conv1d_prefill(x, weight)
    assert torch.allclose(expected, got, atol=1e-6), (
        f"max abs diff: {(expected - got).abs().max().item():.2e}"
    )


def test_zero_weight_produces_zero_output() -> None:
    """Zero weight must give zero output (SiLU(0) == 0)."""
    bs, seq_len, dim = 2, 5, 8
    x = torch.randn(bs, seq_len, dim)
    weight = torch.zeros(3, dim)

    out = optimized_causal_conv1d_prefill(x, weight)
    assert torch.all(out == 0.0), "Expected all-zero output for zero weight"


def test_bias_is_added() -> None:
    """Bias must shift the pre-activation values."""
    torch.manual_seed(3)
    bs, seq_len, dim = 2, 8, 16
    kernel_size = 2
    x = torch.randn(bs, seq_len, dim)
    weight = torch.randn(kernel_size, dim)
    bias = torch.randn(dim)

    out_no_bias = optimized_causal_conv1d_prefill(x, weight, bias=None)
    out_bias = optimized_causal_conv1d_prefill(x, weight, bias=bias)

    # out_bias should differ from out_no_bias (unless bias happens to be 0,
    # which is astronomically unlikely for random tensors).
    assert not torch.allclose(out_no_bias, out_bias), (
        "Adding a non-zero bias should change the output"
    )


def test_seq_len_one_decode_like() -> None:
    """seq_len=1 simulates the single-token decode path."""
    torch.manual_seed(99)
    kernel_size, dim = 4, 32
    x = torch.randn(1, 1, dim)
    weight = torch.randn(kernel_size, dim)

    out = optimized_causal_conv1d_prefill(x, weight)
    assert out.shape == (1, 1, dim)


@pytest.mark.parametrize("bs,seq_len,dim", SHAPES)
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_output_shape(bs: int, seq_len: int, dim: int,
                      kernel_size: int) -> None:
    """Output shape must equal input shape regardless of kernel size."""
    x = torch.randn(bs, seq_len, dim)
    weight = torch.randn(kernel_size, dim)

    out = optimized_causal_conv1d_prefill(x, weight)
    assert out.shape == x.shape, (
        f"Expected output shape {x.shape}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------


def test_wrong_x_dim_raises() -> None:
    with pytest.raises(ValueError, match="3-D"):
        optimized_causal_conv1d_prefill(
            torch.randn(4, 8),   # 2-D, not 3-D
            torch.randn(2, 8),
        )


def test_wrong_weight_dim_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        optimized_causal_conv1d_prefill(
            torch.randn(1, 4, 8),
            torch.randn(2, 4, 8),   # 3-D, not 2-D
        )


def test_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="weight.shape"):
        optimized_causal_conv1d_prefill(
            torch.randn(1, 4, 8),
            torch.randn(2, 16),   # dim mismatch
        )


def test_bias_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="bias shape"):
        optimized_causal_conv1d_prefill(
            torch.randn(1, 4, 8),
            torch.randn(2, 8),
            bias=torch.randn(16),   # wrong bias dim
        )


def test_unsupported_activation_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        optimized_causal_conv1d_prefill(
            torch.randn(1, 4, 8),
            torch.randn(2, 8),
            activation="relu",
        )
