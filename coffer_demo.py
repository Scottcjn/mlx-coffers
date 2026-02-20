#!/usr/bin/env python3
"""
MLX Coffers Demo
================
Downloads TinyLlama (or uses a local model) and runs coffer-routed inference
on Apple Silicon M-series hardware.

Run on M2 Mac Mini (.134):
    pip install mlx mlx-lm
    python3 coffer_demo.py

Or with a local model:
    python3 coffer_demo.py --model /path/to/model
"""

import argparse
import sys
import time

# ── Check MLX availability ────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_coffers import (
    CofferDomain,
    LayerRouter,
    CofferPool,
    MLXCofferInference,
    build_coffer_pool,
)


# ── Demo: routing table without a real model ──────────────────

def demo_routing_table() -> None:
    """Show how different layer names map to cognitive coffers."""
    print("=" * 60)
    print("  MLX Coffers — Layer Routing Demo")
    print("=" * 60)
    print()

    router = LayerRouter(total_layers=32)

    sample_layers = [
        # Attention
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        # MLP
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        # Embedding / norm
        "model.embed_tokens",
        "model.norm",
        "lm_head",
        # KV cache (hypothetical)
        "model.layers.0.kv_cache",
        "past_key_values.0",
    ]

    print(f"  {'Layer Name':<42} {'Domain':<14} {'Stream'}")
    print("  " + "─" * 72)

    for name in sample_layers:
        domain = router.classify(name)
        print(f"  {name:<42} {domain.name:<14} {domain.apple_silicon_stream}")

    print()

    # Show coffer summary
    pool = build_coffer_pool(verbose=True)
    for domain in CofferDomain:
        c = pool.get(domain)
        print(f"  Coffer {domain.value}: {domain.brain_region:<22} "
              f"│ {domain.cognitive_function}")

    print()


# ── Demo: NUMA analogy explanation ───────────────────────────

def demo_numa_analogy() -> None:
    print("─" * 60)
    print("  POWER8 NUMA → Apple Silicon Stream Mapping")
    print("─" * 60)
    rows = [
        ("Coffer", "Brain Region", "POWER8 NUMA", "Apple Silicon"),
        ("──────", "───────────────────", "───────────", "─────────────"),
        ("  0", "Prefrontal Cortex", "Node 3 (189GB)", "Metal GPU"),
        ("  1", "Left Hemisphere", "Node 1 (178GB)", "CPU (P-cores)"),
        ("  2", "Right Hemisphere", "Node 0 (114GB)", "Metal GPU"),
        ("  3", "Temporal Lobe", "Node 2 (43GB)", "CPU (large buf)"),
    ]
    for row in rows:
        print(f"  {row[0]:<8} {row[1]:<22} {row[2]:<18} {row[3]}")
    print()
    print("  On POWER8: NUMA node = physical memory bank")
    print("  On M2:     NUMA node = compute stream (CPU vs Metal)")
    print("  Routing insight is identical — only the substrate changes.")
    print()


# ── Demo: MLX inference with coffers ─────────────────────────

def demo_mlx_inference(model_path: str) -> None:
    if not HAS_MLX:
        print("[!] MLX not installed. Run: pip install mlx mlx-lm")
        return

    try:
        import mlx_lm
    except ImportError:
        print("[!] mlx-lm not installed. Run: pip install mlx-lm")
        return

    print("─" * 60)
    print("  Loading model:", model_path)
    print("─" * 60)

    model, tokenizer = mlx_lm.load(model_path)

    pool = build_coffer_pool(verbose=True)
    inf  = MLXCofferInference(model, pool)

    prompts = [
        "The neuromorphic properties of silicon are",
        "Apple Silicon's unified memory architecture enables",
        "In Hebbian learning, neurons that fire together",
    ]

    print("─" * 60)
    print("  Coffer-routed generation:")
    print("─" * 60)

    total_tokens = 0
    t_start = time.perf_counter()

    for prompt in prompts:
        tokens = mx.array(tokenizer.encode(prompt))[None]
        out_tokens = inf.generate_routed(tokens, max_tokens=40, temp=0.7)
        out_text = tokenizer.decode(out_tokens)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {out_text.strip()}")
        total_tokens += len(out_tokens)

    elapsed = time.perf_counter() - t_start
    tps = total_tokens / elapsed if elapsed > 0 else 0

    print()
    print("─" * 60)
    print(f"  Generated {total_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    print()
    print("  Coffer Statistics:")
    print(pool.stats())


# ── Demo: compare baseline vs coffers ────────────────────────

def demo_benchmark(model_path: str) -> None:
    if not HAS_MLX:
        print("[!] MLX not installed.")
        return

    try:
        import mlx_lm
        from mlx_coffers import benchmark
    except ImportError:
        print("[!] mlx-lm not installed.")
        return

    model, tokenizer = mlx_lm.load(model_path)

    print("─" * 60)
    print("  Benchmark: Baseline vs Coffer-routed")
    print("─" * 60)

    prompt = "The history of artificial intelligence begins with"

    print("\n  [Baseline] all ops on default device:")
    base = benchmark(model, tokenizer, prompt=prompt, n_tokens=20, runs=2, use_coffers=False)
    for r in base["runs"]:
        print(f"    Run {r['run']}: {r['tokens_per_sec']:.1f} tok/s ({r['tokens_generated']} tokens)")
    print(f"    Avg: {base['avg_tokens_per_sec']:.1f} tok/s")

    print("\n  [Coffers] neuromorphic stream routing:")
    coffer = benchmark(model, tokenizer, prompt=prompt, n_tokens=20, runs=2, use_coffers=True)
    for r in coffer["runs"]:
        print(f"    Run {r['run']}: {r['tokens_per_sec']:.1f} tok/s ({r['tokens_generated']} tokens)")
    print(f"    Avg: {coffer['avg_tokens_per_sec']:.1f} tok/s")

    print()
    print(coffer.get("coffer_stats", ""))


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Coffers Demo")
    parser.add_argument("--model", default="mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
                        help="Model path or HuggingFace ID (default: TinyLlama 1.1B 4-bit)")
    parser.add_argument("--bench", action="store_true",
                        help="Run baseline vs coffer benchmark")
    parser.add_argument("--routing-only", action="store_true",
                        help="Just show routing table (no model load)")
    args = parser.parse_args()

    demo_routing_table()
    demo_numa_analogy()

    if args.routing_only:
        print("  (--routing-only: skipping model load)")
        return

    if not HAS_MLX:
        print("  MLX not available — showing routing demo only.")
        print("  Install with: pip install mlx mlx-lm")
        return

    if args.bench:
        demo_benchmark(args.model)
    else:
        demo_mlx_inference(args.model)


if __name__ == "__main__":
    main()
