"""NVTX profiling utilities for nsys profiling."""

import torch


def nvtx_range(name: str):
    """Context manager for NVTX range. Only active when nsys is profiling."""
    return torch.cuda.nvtx.range(name)


def nvtx_push(name: str):
    """Push NVTX range. Must pair with nvtx_pop."""
    torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    torch.cuda.nvtx.range_pop()
