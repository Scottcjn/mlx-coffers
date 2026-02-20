# MLX Coffers 🧠

**Apple Silicon port of [RAM Coffers](https://github.com/Scottcjn/ram-coffers) — neuromorphic compute routing for M-series Macs**

RAM Coffers maps transformer layer types to NUMA memory banks on POWER8 (512GB, 4 nodes).
MLX Coffers maps the same cognitive domains to **CPU vs Metal GPU streams** on Apple Silicon.
The routing insight is identical — only the substrate changes.

---

## The Concept

On POWER8, "coffers" are NUMA memory banks assigned to brain regions:

```
Coffer 0 → Prefrontal Cortex  → NUMA Node 3 (189GB, fastest)
Coffer 1 → Left Hemisphere    → NUMA Node 1 (178GB, language/logic)
Coffer 2 → Right Hemisphere   → NUMA Node 0 (114GB, spatial/creative)
Coffer 3 → Temporal Lobe      → NUMA Node 2 (43GB, KV cache/memory)
```

On Apple Silicon, "NUMA node" becomes **compute stream**:

```
Coffer 0 → Prefrontal Cortex  → Metal GPU   (executive coordination)
Coffer 1 → Left Hemisphere    → CPU          (sequential language/logic)
Coffer 2 → Right Hemisphere   → Metal GPU   (creative MLP synthesis)
Coffer 3 → Temporal Lobe      → CPU          (KV cache, context buffer)
```

This exploits a real architectural fact about Apple Silicon:
- **CPU** excels at sequential, cache-coherent operations (attention projections, autoregressive logic)
- **Metal GPU** excels at massively parallel SIMD ops (MLP expansions, pattern matching, embedding lookups)

MLX exposes both as explicit compute streams via `mx.cpu` and `mx.gpu` devices.

---

## Layer Routing

Each transformer layer is classified by name into a cognitive domain:

| Pattern | Domain | Stream |
|---------|--------|--------|
| `q_proj`, `k_proj`, `v_proj`, `attn` | LEFT_HEMI | CPU |
| `mlp`, `gate_proj`, `up_proj`, `ffn` | RIGHT_HEMI | Metal GPU |
| `embed`, `norm`, `lm_head` | PREFRONTAL | Metal GPU |
| `kv_cache`, `past_key_values` | TEMPORAL | CPU |

---

## Installation

On M2 Mac Mini (or any Apple Silicon):

```bash
pip install mlx mlx-lm
git clone https://github.com/Scottcjn/mlx-coffers
cd mlx-coffers
```

---

## Usage

### Quick demo (routing table only, no model download)

```bash
python3 coffer_demo.py --routing-only
```

### Full inference demo (downloads TinyLlama ~600MB)

```bash
python3 coffer_demo.py
```

### Use a custom model

```bash
python3 coffer_demo.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

### Benchmark baseline vs coffer-routed

```bash
python3 coffer_demo.py --bench
```

### Python API

```python
from mlx_coffers import build_coffer_pool, MLXCofferInference
import mlx_lm

model, tokenizer = mlx_lm.load("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")

pool = build_coffer_pool(verbose=True)
inf  = MLXCofferInference(model, pool)

import mlx.core as mx
tokens = mx.array(tokenizer.encode("The nature of intelligence is"))[None]
out    = inf.generate_routed(tokens, max_tokens=50)
print(tokenizer.decode(out))

print(pool.stats())
```

---

## Architecture

```
mlx_coffers.py
├── CofferDomain   — enum: PREFRONTAL, LEFT_HEMI, RIGHT_HEMI, TEMPORAL
├── LayerRouter    — classifies layer names → CofferDomain
│     rules: keyword matching (attn→LEFT_HEMI, mlp→RIGHT_HEMI, etc.)
│     fallback: position-based (early/mid/late layer index)
├── CofferPool     — holds 4 Coffer instances, each with mx.cpu/mx.gpu device
│     pool.route("q_proj") → Coffer(LEFT_HEMI, device=mx.cpu)
└── MLXCofferInference
      wraps mlx_lm model, routes forward passes through coffer streams
      tracks per-coffer hit counts and latency for profiling
```

---

## Neuromorphic Basis

The coffer mapping is grounded in neuroscience:

| Coffer | Brodmann Areas | Function | Hardware Analog |
|--------|----------------|----------|-----------------|
| PREFRONTAL | BA9/46 (DLPFC) | Working memory, planning, coordination | Metal GPU — fast parallel dispatch |
| LEFT_HEMI | BA44/45 (Broca's), BA22 (Wernicke's) | Language production/comprehension | CPU — sequential, precise |
| RIGHT_HEMI | BA39/40 (Parietal) | Spatial reasoning, pattern synthesis | Metal GPU — SIMD pattern matching |
| TEMPORAL | BA35/36 (Perirhinal) | Episodic recognition memory | CPU large buffer — KV cache |

---

## Relationship to RAM Coffers

This is the Apple Silicon prototype of the [RAM Coffers](https://github.com/Scottcjn/ram-coffers) architecture.

| Feature | RAM Coffers (POWER8) | MLX Coffers (Apple Silicon) |
|---------|---------------------|----------------------------|
| Routing substrate | Physical NUMA nodes | Compute streams (CPU/GPU) |
| Memory banking | `numactl --membind=N` | `mx.stream(mx.cpu/mx.gpu)` |
| Prefetch | `dcbt` resident hints | Metal command buffer batching |
| Vec_perm collapse | POWER8 VSX `vec_perm` | MLX `mx.conv_general` / `mx.matmul` |
| Hardware entropy | `mftb` timebase | Metal GPU timestamp counter |
| NUMA nodes | 4 (Node 0-3, 544GB) | 2 streams (CPU + Metal GPU) |

The key insight transfers directly: **route ops to the hardware unit best suited for that cognitive function**, rather than letting the runtime decide uniformly.

---

## Research Note

This work is part of the **Elyan Labs neuromorphic inference** research program.
The RAM Coffers NUMA routing concept was first implemented December 16, 2025 on an IBM POWER8 S824 (512GB RAM, 4 NUMA nodes), predating DeepSeek Engram (arXiv:2601.07372, January 12, 2026) by 27 days.

MLX Coffers demonstrates the portability of the routing concept to Apple Silicon's heterogeneous compute architecture — and sets the stage for mixed Mac + CUDA inference clusters.

---

## Contributing

This is a prototype. Contributions welcome:
- Per-layer stream dispatch (currently wraps full model call)
- M1/M3/M4 benchmarks
- Integration with `exo` distributed inference
- Profiling Metal GPU utilization with `mx.metal.device_info()`

---

## License

MIT — Elyan Labs 2026
