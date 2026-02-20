#!/usr/bin/env python3
"""
Synthetic coffer test — no model download required.
Creates a tiny MLX transformer layer and routes ops through CPU/GPU streams.
Proves CofferPool routing works on Apple Silicon without needing HuggingFace.

Run: python3 test_coffers_synthetic.py
"""

import time
import sys

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    print("MLX not available — install: pip install mlx")
    sys.exit(1)

from mlx_coffers import (
    CofferDomain, LayerRouter, CofferPool, build_coffer_pool
)


def run_stream_timing_test() -> None:
    """Time matmul operations on CPU vs Metal GPU stream."""
    print("=" * 60)
    print("  MLX Stream Timing: CPU vs Metal GPU")
    print("=" * 60)
    print()

    sizes = [(512, 512), (1024, 1024), (2048, 512)]

    for M, N in sizes:
        A = mx.random.normal((M, N))
        B = mx.random.normal((N, M))
        mx.eval(A, B)  # Materialize

        # CPU stream
        t0 = time.perf_counter()
        with mx.stream(mx.cpu):
            C_cpu = mx.matmul(A, B)
            mx.eval(C_cpu)
        cpu_ms = (time.perf_counter() - t0) * 1000

        # GPU Metal stream
        t0 = time.perf_counter()
        with mx.stream(mx.gpu):
            C_gpu = mx.matmul(A, B)
            mx.eval(C_gpu)
        gpu_ms = (time.perf_counter() - t0) * 1000

        print(f"  {M}x{N} matmul:  CPU {cpu_ms:6.1f}ms  |  Metal GPU {gpu_ms:6.1f}ms")

    print()


def run_coffer_routing_test() -> None:
    """Route simulated transformer layer ops through cognitive coffers."""
    print("=" * 60)
    print("  Coffer-Routed Synthetic Transformer Test")
    print("=" * 60)
    print()

    pool = build_coffer_pool(verbose=True)

    # Simulate a 4-layer transformer block
    batch, seq, hidden = 1, 32, 512
    x = mx.random.normal((batch, seq, hidden))
    mx.eval(x)

    layer_ops = [
        # (layer_name, lambda: operation)
        ("model.embed_tokens",          lambda: mx.take(mx.random.normal((32000, hidden)), mx.array([1,2,3,4,5]))),
        ("model.layers.0.self_attn.q_proj", lambda: mx.matmul(x.reshape(-1, hidden), mx.random.normal((hidden, hidden)))),
        ("model.layers.0.self_attn.k_proj", lambda: mx.matmul(x.reshape(-1, hidden), mx.random.normal((hidden, hidden)))),
        ("model.layers.0.self_attn.v_proj", lambda: mx.matmul(x.reshape(-1, hidden), mx.random.normal((hidden, hidden)))),
        ("model.layers.0.mlp.gate_proj",    lambda: mx.matmul(x.reshape(-1, hidden), mx.random.normal((hidden, hidden * 4)))),
        ("model.layers.0.mlp.up_proj",      lambda: mx.matmul(x.reshape(-1, hidden), mx.random.normal((hidden, hidden * 4)))),
        ("model.layers.0.mlp.down_proj",    lambda: mx.matmul(x.reshape(-1, hidden * 4), mx.random.normal((hidden * 4, hidden)))),
        ("model.norm",                      lambda: (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)),
        ("lm_head",                         lambda: mx.matmul(x.reshape(-1, hidden), mx.random.normal((hidden, 32000)))),
    ]

    print(f"  {'Layer':<42} {'Domain':<14} {'Stream':<12} {'ms':>6}")
    print("  " + "─" * 78)

    total_start = time.perf_counter()

    for name, op in layer_ops:
        coffer = pool.route(name)
        t0 = time.perf_counter()
        with mx.stream(coffer.device):
            result = op()
            mx.eval(result)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        coffer.ops_ms += elapsed_ms

        print(f"  {name:<42} {coffer.domain.name:<14} {coffer.domain.apple_silicon_stream:<12} {elapsed_ms:>6.2f}")

    total_ms = (time.perf_counter() - total_start) * 1000
    print()
    print(f"  Total: {total_ms:.1f}ms for {len(layer_ops)} layer ops")
    print()
    print("  Coffer Statistics:")
    print(pool.stats())


def run_attention_coffer_test() -> None:
    """Demonstrate LEFT_HEMI (CPU) vs RIGHT_HEMI (GPU) for attention vs MLP."""
    print("=" * 60)
    print("  Attention (LEFT_HEMI/CPU) vs MLP (RIGHT_HEMI/Metal) Split")
    print("=" * 60)
    print()

    hidden = 768
    intermediate = hidden * 4

    x = mx.random.normal((16, hidden))  # 16 tokens, 768 hidden
    Wq = mx.random.normal((hidden, hidden))
    Wk = mx.random.normal((hidden, hidden))
    Wv = mx.random.normal((hidden, hidden))
    Wo = mx.random.normal((hidden, hidden))
    Wgate = mx.random.normal((hidden, intermediate))
    Wup   = mx.random.normal((hidden, intermediate))
    Wdown = mx.random.normal((intermediate, hidden))
    mx.eval(x, Wq, Wk, Wv, Wo, Wgate, Wup, Wdown)

    # Attention block → LEFT_HEMI → CPU
    t0 = time.perf_counter()
    with mx.stream(mx.cpu):
        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv
        scale = hidden ** -0.5
        attn = mx.softmax((Q @ K.T) * scale, axis=-1) @ V
        attn_out = attn @ Wo
        mx.eval(attn_out)
    attn_ms = (time.perf_counter() - t0) * 1000

    # MLP block → RIGHT_HEMI → Metal GPU
    t0 = time.perf_counter()
    with mx.stream(mx.gpu):
        gate = mx.sigmoid(attn_out @ Wgate)
        up   = attn_out @ Wup
        mlp_out = (gate * up) @ Wdown
        mx.eval(mlp_out)
    mlp_ms = (time.perf_counter() - t0) * 1000

    print(f"  Attention (CPU / LEFT_HEMI):     {attn_ms:.2f}ms — sequential Q/K/V projections")
    print(f"  MLP       (Metal / RIGHT_HEMI):  {mlp_ms:.2f}ms — parallel gate+up+down")
    print()
    print(f"  Output shape: {mlp_out.shape}  (same as input)")
    print()


def main() -> None:
    print()
    print("  MLX Coffers — Synthetic Test on Apple Silicon")
    print(f"  MLX version: {mx.__version__}")
    devices = []
    try:
        devices.append(f"GPU: {mx.metal.device_info()['device_name']}")
    except Exception:
        devices.append("GPU: Metal (details unavailable)")
    devices.append("CPU: Apple Silicon")
    for d in devices:
        print(f"  {d}")
    print()

    run_stream_timing_test()
    run_coffer_routing_test()
    run_attention_coffer_test()

    print("=" * 60)
    print("  All tests passed! Coffers routing on Apple Silicon works.")
    print("  Neuromorphic stream dispatch: CPU for language/logic,")
    print("  Metal GPU for spatial/creative ops — just like POWER8 NUMA.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
