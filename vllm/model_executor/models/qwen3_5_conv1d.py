# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone optimized causal conv1d (prefill path) extracted from Qwen3.5.

The original implementation lives in
``vllm/model_executor/models/qwen3_5.py``, inside
``Qwen3_5HybridAttentionDecoderLayer.forward`` (lines 377-412).

The prefill path avoids calling ``F.conv1d`` (which allocates a large
intermediate tensor) by performing the depth-wise convolution as a loop over
kernel positions with in-place accumulation.  Each iteration multiplies a
shifted view of the padded input by the corresponding weight row and adds it
to the running sum, yielding a fully vectorised but memory-efficient causal
conv1d followed by a SiLU activation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def optimized_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Optimised depth-wise causal conv1d for the prefill (prompt) phase.

    This function implements the same mathematical operation as a standard
    ``torch.nn.Conv1d`` with ``groups=dim``, ``padding=kernel_size-1``,
    and ``bias``, followed by a non-linearity, but avoids allocating the
    large intermediate tensor that ``F.conv1d`` would produce.

    Algorithm
    ---------
    1. Left-pad ``x`` with ``kernel_size - 1`` zeros along the sequence
       dimension so that the output has the same sequence length as the
       input (causal padding).
    2. For each kernel index ``k`` in ``[0, kernel_size)``:
       - Extract the slice ``x_padded[:, k : k + seq_len, :]``.
       - Multiply element-wise by ``weight[k]`` (shape ``[dim]``), which
         broadcasts over the batch and sequence dimensions.
       - Accumulate the result in-place into the output buffer.
    3. Optionally add a bias term.
    4. Apply the requested activation function (``"silu"`` by default).

    Parameters
    ----------
    x:
        Input tensor of shape ``[batch, seq_len, dim]``.
    weight:
        Pre-processed conv weight of shape ``[kernel_size, dim]``.
        This corresponds to ``nn.Conv1d.weight`` reshaped as::

            weight = conv1d.weight.squeeze(1).transpose(0, 1)
                       .flatten()
                       .reshape(kernel_size, dim)

    bias:
        Optional bias tensor of shape ``[dim]``.
    activation:
        Name of the activation function to apply after the convolution.
        Currently only ``"silu"`` is supported.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``[batch, seq_len, dim]`` with the same
        dtype as ``x``.
    """
    if x.dim() != 3:
        raise ValueError(
            f"x must be a 3-D tensor [batch, seq_len, dim], got {x.dim()}-D")
    if weight.dim() != 2:
        raise ValueError(
            f"weight must be a 2-D tensor [kernel_size, dim], "
            f"got {weight.dim()}-D")

    bs, seq_len, dim = x.shape
    kernel_size = weight.shape[0]

    if weight.shape[1] != dim:
        raise ValueError(
            f"weight.shape[1] ({weight.shape[1]}) must match x.shape[2] "
            f"({dim})")
    if bias is not None and bias.shape != (dim, ):
        raise ValueError(
            f"bias shape {tuple(bias.shape)} must be ({dim},)")
    if activation != "silu":
        raise ValueError(
            f"Unsupported activation '{activation}'. Only 'silu' is supported."
        )

    # Left-pad the sequence dimension with (kernel_size - 1) zeros so the
    # output keeps the same sequence length (causal padding).
    x_padded = F.pad(x, (0, 0, kernel_size - 1, 0))

    # Accumulate contributions from each kernel position.
    output: torch.Tensor | None = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k:k + seq_len, :]   # [bs, seq_len, dim]
        contribution = x_slice * weight[k]          # broadcasts over bs/seq
        if output is None:
            output = contribution
        else:
            output.add_(contribution)

    assert output is not None  # kernel_size >= 1 guaranteed by weight shape

    if bias is not None:
        output = output + bias

    return F.silu(output)
