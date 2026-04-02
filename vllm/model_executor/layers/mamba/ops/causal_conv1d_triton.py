# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernel for causal 1D depthwise convolution with SiLU activation.

Extracted from the optimized conv1d implementation in
vllm/model_executor/models/qwen3_5.py (Qwen3NextGatedDeltaNet).

The algorithm performs:
  1. Left-pad the input with zeros (causal padding of conv_kernel_size - 1)
  2. For each kernel position k in [0, conv_kernel_size):
       output[:, t, d] += input[:, t + k - (conv_kernel_size - 1), d]
                           * weight[k, d]
     (where out-of-bound input positions are zero due to padding)
  3. Apply SiLU activation: output = output * sigmoid(output)

Input layout:  (batch, seq_len, dim)   -- channel-last
Weight layout: (conv_kernel_size, dim)  -- already transposed / flattened
Output layout: (batch, seq_len, dim)
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _causal_conv1d_fwd_kernel(
        # Pointers
        output_ptr,
        input_ptr,
        weight_ptr,
        # Dimensions
        batch: tl.constexpr,
        seq_len,
        dim,
        conv_kernel_size: tl.constexpr,
        # Strides for input  (batch, seq_len, dim)
        stride_ib,
        stride_is,
        stride_id,
        # Strides for weight (conv_kernel_size, dim)
        stride_wk,
        stride_wd,
        # Strides for output (batch, seq_len, dim)
        stride_ob,
        stride_os,
        stride_od,
        # Meta
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        APPLY_SILU: tl.constexpr,
    ):
        """Each program instance computes a tile of [BLOCK_SEQ, BLOCK_DIM]
        elements for one batch element."""
        pid_b = tl.program_id(0)  # batch index
        pid_s = tl.program_id(1)  # seq tile index
        pid_d = tl.program_id(2)  # dim tile index

        seq_offs = pid_s * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        dim_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

        # Masks
        seq_mask = seq_offs < seq_len
        dim_mask = dim_offs < dim

        # Accumulator
        acc = tl.zeros((BLOCK_SEQ, BLOCK_DIM), dtype=tl.float32)

        # Loop over kernel positions
        for k in tl.static_range(conv_kernel_size):
            # The padded input index: t_padded = seq_offs + k
            # Actual input index (before padding of conv_kernel_size - 1):
            #   t_in = t_padded - (conv_kernel_size - 1) = seq_offs + k
            #          - (conv_kernel_size - 1)
            t_in = seq_offs + k - (conv_kernel_size - 1)
            in_bounds = (t_in >= 0) & (t_in < seq_len) & seq_mask

            # Load input slice: input[pid_b, t_in, dim_offs]
            inp_ptrs = (input_ptr + pid_b * stride_ib +
                        t_in[:, None] * stride_is +
                        dim_offs[None, :] * stride_id)
            inp = tl.load(inp_ptrs,
                          mask=in_bounds[:, None] & dim_mask[None, :],
                          other=0.0)

            # Load weight slice: weight[k, dim_offs]
            w_ptrs = weight_ptr + k * stride_wk + dim_offs * stride_wd
            w = tl.load(w_ptrs, mask=dim_mask, other=0.0)

            # Element-wise multiply and accumulate
            acc += inp * w[None, :]

        # Apply SiLU activation: x * sigmoid(x)
        if APPLY_SILU:
            acc = acc * tl.sigmoid(acc)

        # Store output: output[pid_b, seq_offs, dim_offs]
        out_ptrs = (output_ptr + pid_b * stride_ob +
                    seq_offs[:, None] * stride_os +
                    dim_offs[None, :] * stride_od)
        tl.store(out_ptrs,
                 acc.to(output_ptr.dtype.element_ty),
                 mask=seq_mask[:, None] & dim_mask[None, :])


def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Causal 1D depthwise convolution using a Triton kernel.

    Args:
        x: Input tensor of shape (batch, seq_len, dim).
        weight: Convolution weights of shape (conv_kernel_size, dim).
            This is the pre-processed weight format used in Qwen3.5, where
            the original conv1d weight has been squeezed, transposed, and
            reshaped.
        apply_silu: Whether to apply SiLU activation after convolution.

    Returns:
        Output tensor of shape (batch, seq_len, dim).
    """
    assert x.ndim == 3, f"Expected 3D input (batch, seq_len, dim), got {x.ndim}D"
    assert weight.ndim == 2, \
        f"Expected 2D weight (kernel_size, dim), got {weight.ndim}D"

    batch, seq_len, dim = x.shape
    conv_kernel_size = weight.shape[0]
    assert weight.shape[1] == dim, \
        f"Weight dim {weight.shape[1]} != input dim {dim}"

    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output
    output = torch.empty_like(x)

    # Compute grid and block sizes
    BLOCK_SEQ = min(triton.next_power_of_2(seq_len), 128)
    BLOCK_DIM = min(triton.next_power_of_2(dim), 128)

    grid = (
        batch,
        triton.cdiv(seq_len, BLOCK_SEQ),
        triton.cdiv(dim, BLOCK_DIM),
    )

    _causal_conv1d_fwd_kernel[grid](
        output,
        x,
        weight,
        batch,
        seq_len,
        dim,
        conv_kernel_size,
        # input strides
        x.stride(0),
        x.stride(1),
        x.stride(2),
        # weight strides
        weight.stride(0),
        weight.stride(1),
        # output strides
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # Meta
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        APPLY_SILU=apply_silu,
    )

    return output


def causal_conv1d_torch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Reference PyTorch implementation of causal 1D depthwise convolution.

    This replicates the algorithm from qwen3_5.py lines 388-399.

    Args:
        x: Input tensor of shape (batch, seq_len, dim).
        weight: Convolution weights of shape (conv_kernel_size, dim).
        apply_silu: Whether to apply SiLU activation after convolution.

    Returns:
        Output tensor of shape (batch, seq_len, dim).
    """
    conv_kernel_size = weight.shape[0]
    seq_len = x.shape[1]

    # Pad with zeros on the left (causal padding)
    x_padded = F.pad(x, (0, 0, conv_kernel_size - 1, 0))

    # Accumulate convolution across kernel positions
    output = None
    for k in range(conv_kernel_size):
        x_slice = x_padded[:, k:(k + seq_len), :]
        contrib = x_slice * weight[k]
        if output is None:
            output = contrib
        else:
            output = output + contrib

    if apply_silu:
        output = F.silu(output)

    return output
