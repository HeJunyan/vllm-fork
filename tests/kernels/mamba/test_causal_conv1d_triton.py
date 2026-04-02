# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the causal conv1d Triton kernel.

Tests verify correctness of the Triton kernel against a pure-PyTorch reference
implementation that mirrors the original algorithm from qwen3_5.py.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d_triton import (
    causal_conv1d_torch_ref,
    causal_conv1d_triton,
)
from vllm.triton_utils import HAS_TRITON

# All tests require CUDA since Triton compiles GPU kernels.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="CUDA not available"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed"),
]

DEVICE = "cuda"


# ---- helper ----------------------------------------------------------------
def _run_and_compare(
    batch: int,
    seq_len: int,
    dim: int,
    conv_kernel_size: int,
    apply_silu: bool,
    dtype: torch.dtype,
):
    """Run both Triton and reference, then compare outputs."""
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=dtype)
    weight = torch.randn(conv_kernel_size, dim, device=DEVICE,
                          dtype=torch.float32)

    # Reference: operate in float32 for accuracy
    out_ref = causal_conv1d_torch_ref(x.float(), weight, apply_silu=apply_silu)

    # Triton kernel
    out_tri = causal_conv1d_triton(x.float(), weight, apply_silu=apply_silu)

    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(out_tri, out_ref, rtol=rtol, atol=atol)

    return out_tri, out_ref


# ---- basic correctness tests -----------------------------------------------

@pytest.mark.parametrize("conv_kernel_size", [2, 3, 4, 7])
@pytest.mark.parametrize("seq_len", [1, 4, 16, 64, 128, 256])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_basic_shapes(batch, seq_len, conv_kernel_size):
    """Test various combinations of batch, seq_len, and kernel size."""
    dim = 64
    _run_and_compare(batch, seq_len, dim, conv_kernel_size,
                     apply_silu=True, dtype=torch.float32)


@pytest.mark.parametrize("dim", [1, 15, 32, 63, 64, 128, 255, 256, 512])
def test_various_dims(dim):
    """Test non-power-of-two and edge-case dimensions."""
    _run_and_compare(batch=2, seq_len=32, dim=dim, conv_kernel_size=4,
                     apply_silu=True, dtype=torch.float32)


@pytest.mark.parametrize("apply_silu", [True, False])
def test_silu_toggle(apply_silu):
    """Test with and without SiLU activation."""
    _run_and_compare(batch=2, seq_len=32, dim=64, conv_kernel_size=4,
                     apply_silu=apply_silu, dtype=torch.float32)


# ---- dtype tests -----------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16,
                                    torch.bfloat16])
def test_dtypes(dtype):
    """Test that the kernel works with different dtypes."""
    torch.manual_seed(42)
    batch, seq_len, dim, kernel = 2, 32, 64, 4
    x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=dtype)
    weight = torch.randn(kernel, dim, device=DEVICE, dtype=torch.float32)

    out_ref = causal_conv1d_torch_ref(x.float(), weight, apply_silu=True)
    out_tri = causal_conv1d_triton(x.float(), weight, apply_silu=True)

    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(out_tri, out_ref, rtol=rtol, atol=atol)


# ---- causal property tests -------------------------------------------------

def test_causal_property():
    """Verify that the convolution is causal: output at time t depends only
    on inputs at times <= t."""
    torch.manual_seed(42)
    batch, seq_len, dim, kernel = 1, 16, 32, 4

    x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(kernel, dim, device=DEVICE, dtype=torch.float32)

    out_full = causal_conv1d_triton(x, weight, apply_silu=False)

    # Zero out future inputs and check that past outputs are unchanged.
    # For time step t, zero out everything after t.
    for t in range(seq_len):
        x_masked = x.clone()
        x_masked[:, t + 1:, :] = 0.0
        out_masked = causal_conv1d_triton(x_masked, weight, apply_silu=False)
        torch.testing.assert_close(
            out_full[:, t, :], out_masked[:, t, :],
            rtol=1e-5, atol=1e-5,
            msg=f"Causal violation at time step {t}",
        )


# ---- equivalence with F.conv1d --------------------------------------------

def test_equivalence_with_torch_conv1d():
    """Verify that the kernel produces the same result as torch.nn.functional
    conv1d with appropriate parameters (depthwise, causal padding)."""
    torch.manual_seed(42)
    batch, seq_len, dim, kernel = 2, 32, 64, 4

    x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(kernel, dim, device=DEVICE, dtype=torch.float32)

    # Triton kernel result (no silu for easier comparison)
    out_tri = causal_conv1d_triton(x, weight, apply_silu=False)

    # Equivalent F.conv1d: depthwise conv1d with causal (left) padding.
    # F.conv1d expects (batch, channels, length); weight as (out_ch, 1, kW)
    x_t = x.transpose(1, 2)  # (batch, dim, seq_len)
    # weight needs to be (dim, 1, kernel_size) with each channel independent
    # Our weight is (kernel_size, dim), F.conv1d groups weight is
    # (dim, 1, kernel_size)
    w_conv = weight.t().unsqueeze(1)  # (dim, 1, kernel)
    out_conv = F.conv1d(x_t, w_conv, padding=kernel - 1,
                        groups=dim)[:, :, :seq_len]
    out_conv = out_conv.transpose(1, 2)  # back to (batch, seq_len, dim)

    torch.testing.assert_close(out_tri, out_conv, rtol=1e-4, atol=1e-4)


# ---- edge cases ------------------------------------------------------------

def test_seq_len_one():
    """Sequence length of 1 should still work."""
    _run_and_compare(batch=1, seq_len=1, dim=64, conv_kernel_size=4,
                     apply_silu=True, dtype=torch.float32)


def test_kernel_size_one():
    """Kernel size of 1 means no temporal mixing, just element-wise mul."""
    torch.manual_seed(42)
    batch, seq_len, dim = 2, 16, 64
    x = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(1, dim, device=DEVICE, dtype=torch.float32)

    out = causal_conv1d_triton(x, weight, apply_silu=False)
    expected = x * weight[0]
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_zero_input():
    """Zero input should produce zero output (before SiLU) or 0.0 for SiLU
    since SiLU(0) = 0."""
    batch, seq_len, dim, kernel = 2, 16, 64, 4
    x = torch.zeros(batch, seq_len, dim, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(kernel, dim, device=DEVICE, dtype=torch.float32)

    out = causal_conv1d_triton(x, weight, apply_silu=True)
    torch.testing.assert_close(
        out, torch.zeros_like(out), rtol=0, atol=1e-6)


def test_large_sequence():
    """Test with a larger sequence length that requires multiple blocks."""
    _run_and_compare(batch=2, seq_len=1024, dim=128, conv_kernel_size=4,
                     apply_silu=True, dtype=torch.float32)


def test_large_dim():
    """Test with a large dimension that requires multiple blocks."""
    _run_and_compare(batch=1, seq_len=64, dim=2048, conv_kernel_size=4,
                     apply_silu=True, dtype=torch.float32)
