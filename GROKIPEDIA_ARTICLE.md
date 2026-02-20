# RAM Coffers

**RAM Coffers** is a neuromorphic compute routing framework for large language model (LLM) inference that dispatches transformer operations to hardware based on their *cognitive semantic type* rather than their position in the model's layer sequence. Developed at Elyan Labs with a priority date of December 16, 2025, it was the first system to apply brain-hemisphere locality principles to NUMA-aware weight banking and heterogeneous cluster routing.

**DOI**: [10.5281/zenodo.18717767](https://zenodo.org/records/18717767)
**Repository**: https://github.com/Scottcjn/mlx-coffers

---

## Overview

Every transformer-based LLM processes tokens through alternating **attention** and **MLP** blocks. Standard inference runtimes (llama.cpp, MLX, vLLM, exo) treat these identically — they go to whatever device is configured. RAM Coffers treats them differently, because they *are* different:

- **Attention layers** (`q_proj`, `k_proj`, `v_proj`) are sequential, data-dependent, and cache-coherent — characteristics that match CPU execution: out-of-order execution, branch prediction, large L3 caches.
- **MLP layers** (`gate_proj`, `up_proj`, `down_proj`) are large uniform matrix multiplications — embarrassingly parallel, SIMD-friendly, ideal for GPU.
- **Normalization and output** (`lm_head`, `norm`) are fast coordination ops — low-latency, suited to GPU pipeline flush.
- **KV cache** is a large sequential buffer with episodic access — fits best in high-RAM nodes with prefetch-optimized CPUs.

The routing principle is borrowed from neuroscience: **different cognitive functions localize to different brain regions**. RAM Coffers maps this to hardware.

---

## The Four Coffers

| Coffer | Cognitive Domain | Brain Region | Layers | Preferred Hardware |
|--------|-----------------|--------------|--------|--------------------|
| 0 | PREFRONTAL | Prefrontal Cortex (BA9/46) | `norm`, `lm_head`, embeddings | GPU — fast coordination |
| 1 | LEFT_HEMI | Left Hemisphere (Broca's, Wernicke's) | `q_proj`, `k_proj`, `v_proj`, `o_proj` | CPU — sequential, language |
| 2 | RIGHT_HEMI | Right Hemisphere (Parietal, BA39/40) | `mlp`, `gate_proj`, `up_proj`, `down_proj` | GPU — parallel synthesis |
| 3 | TEMPORAL | Temporal Lobe (Perirhinal, BA35/36) | KV cache, `past_key_values` | High-RAM CPU — episodic memory |

Layer classification is O(1) — a regex match on the layer name string. No profiling required at runtime.

---

## Implementations

### POWER8 NUMA (Original)

On an IBM POWER8 S824 with 512 GB RAM across four NUMA nodes, each coffer maps to a physical memory bank:

```
NUMA Node 0 (114 GB) → RIGHT_HEMI  (MLP weights)
NUMA Node 1 (178 GB) → LEFT_HEMI   (attention weights)
NUMA Node 2  (43 GB) → TEMPORAL    (KV cache)
NUMA Node 3 (189 GB) → PREFRONTAL  (embeddings, output)
```

NUMA-local memory access is up to 2× faster than remote access on this hardware (400–425 MB/s local vs 215 MB/s remote), making the placement non-trivial. The implementation uses `numactl --cpunodebind=N --membind=N` per coffer and POWER8's `dcbt` instruction to keep weight tensors resident in L2/L3 cache.

Combined with **vec_perm non-bijunctive collapse** — a POWER8-native Hebbian attention mechanism using the AltiVec `vec_perm` dual-source permute to prune weak activations and amplify strong ones in a single instruction — the system achieves:

| Configuration | Prompt throughput | Speedup |
|---------------|-------------------|---------|
| Scalar baseline | 16.74 t/s | 1.0× |
| POWER8 VSX | 66.49 t/s | 3.97× |
| 64 threads optimal | 84.62 t/s | 5.05× |
| PSE + full resident prefetch | **147.54 t/s** | **8.81×** |

(Measured on TinyLlama 1.1B Q4, pp128 benchmark.)

### MLX Coffers (Apple Silicon Port)

On Apple M2 (unified memory, no physical NUMA), the coffer concept maps to **compute stream affinity**:

- `LEFT_HEMI` → `mx.stream(mx.cpu)` — attention ops on P-cores
- `RIGHT_HEMI` → `mx.stream(mx.gpu)` — MLP ops on Metal GPU
- `PREFRONTAL` → `mx.stream(mx.gpu)` — executive coordination
- `TEMPORAL` → `mx.stream(mx.cpu)` — KV cache with large buffer prefetch

Measured on M2 Mac Mini (24 GB unified memory):

| Layer | Domain | Stream | Time |
|-------|--------|--------|------|
| `self_attn.q_proj` | LEFT_HEMI | CPU | 2.4 ms |
| `self_attn.k_proj` | LEFT_HEMI | CPU | 2.0 ms |
| `mlp.gate_proj` | RIGHT_HEMI | Metal GPU | 49 ms* |
| `mlp.up_proj` | RIGHT_HEMI | Metal GPU | 7.9 ms |
| `lm_head` | PREFRONTAL | Metal GPU | 20 ms |

*First-call kernel compilation; cached runs are significantly faster.

The stream timing crossover empirically validates the domain mapping: CPU beats Metal GPU for small ops (< ~512×512) due to GPU kernel launch latency (~10–40 ms); Metal GPU wins for large parallel MLP ops.

---

## Mixed-Cluster Routing (Mac + CUDA)

MLX Coffers extends to heterogeneous inference clusters using a binary TCP matmul protocol (GPU3, magic `0x47505533`). Each node advertises its capabilities via an HTTP sidecar:

```
Node       Arch    GPU                     Role
─────────────────────────────────────────────────
mlx-m2     arm64   Apple Metal M2          LEFT_HEMI, PREFRONTAL (low-latency)
cuda-gpu   x86_64  NVIDIA GeForce RTX 5070 RIGHT_HEMI (high FLOP/s)
power8     ppc64   —                       TEMPORAL (512 GB RAM for KV cache)
```

The `CofferRouter` selects the best node per operation based on:
1. Cognitive domain match (domain → preferred node type)
2. Matrix size (small → M2 CPU-stream, large → CUDA)
3. Node health (automatic failover to local numpy)

### Live Cluster Validation (Elyan Labs, February 2026)

End-to-end test on M2 Mac Mini + RTX 5070 node, 6 operations spanning attention and MLP domains:

```
Operation                        Domain      Node      Time     Result
────────────────────────────────────────────────────────────────────────
attn q_proj  (F16, 16×512×512)  LEFT_HEMI   mlx-m2    1.7 ms   ✓
lm_head      (F16, 1×512×512)   PREFRONTAL  mlx-m2    0.8 ms   ✓
size routing (F16, 16×512×512)  UNKNOWN     mlx-m2    1.1 ms   ✓
MLP gate_proj (32×4096×4096)   RIGHT_HEMI  cuda-gpu  2735 ms  ✓
MLP up_proj   (64×4096×4096)   RIGHT_HEMI  mlx-m2     126 ms  ✓
MLP ffn Llama2 (32×4096×11008) RIGHT_HEMI  cuda-gpu 10253 ms  ✓

6/6 ALL TESTS PASSED | 0 fallbacks
```

The routing split is visible: all attention ops go to M2, large MLP ops split between M2 and CUDA based on the latency penalty of crossing the 100 ms RTT to the CUDA node.

---

## Differentiation from exo

[exo](https://github.com/exo-explore/exo) is the leading heterogeneous distributed inference framework. RAM Coffers is not a replacement — it operates at a different level.

| Dimension | exo | RAM Coffers |
|-----------|-----|-------------|
| Routing basis | Layer index (sequential shard) | Cognitive domain (semantic type) |
| Attention ops | Same as MLP | Always → CPU-stream / low-latency node |
| MLP ops | Same as attention | Always → GPU / high-FLOP node |
| KV cache | No special handling | → High-RAM TEMPORAL node |
| Intra-node routing | None | CPU vs GPU stream dispatch |
| NUMA awareness | None | 4-node physical NUMA (POWER8) |

The core distinction: **exo asks "which layers go to this node?"** RAM Coffers asks **"what kind of operation is this, and what hardware fits it best?"**

The two systems are **orthogonal and composable**: an exo node running MLX Coffers internally would route its assigned layers to CPU or Metal GPU streams before exo handles cross-node communication. This is a two-level hierarchy — coffer domain (intra-node) × exo shard (inter-node).

> *exo cannot express "attention always to the low-latency CPU-stream node regardless of layer index." RAM Coffers can.*

---

## Priority and Related Work

RAM Coffers predates **DeepSeek Engram** (arXiv:2601.07372, January 12, 2026) by 27 days. Both systems use domain-specialized routing for LLM inference, but differ significantly:

- DeepSeek Engram specializes *memory storage* by domain for mixture-of-experts.
- RAM Coffers specializes *compute dispatch* by cognitive semantic type, with explicit NUMA topology, Apple Silicon stream mapping, entropy injection, and vintage hardware support.

> "They separate memory. We model cognition."

---

## Behavioral Divergence (PSE Entropy)

An unexpected property of the POWER8 implementation: hardware entropy injection via the POWER8 `mftb` timebase register causes behavioral divergence — three runs with the same seed (42) and temperature (0.7) produce different output, confirmed by different MD5 checksums. This is a side effect of `vec_perm` patterns seeded from sub-nanosecond clock drift.

This maps to the **PSE (Probabilistic Semantic Entropy)** marker system:
- Lower DR (Drift Rate) — fewer contradictions in long chains
- Higher ACS (Adversarial Coherence Score) — stable reasoning under adversarial prompts
- MCI variance (Memory Coherence Index) — personality consistency from entropy-seeded co-activation

---

## Citation

```bibtex
@misc{elyan2025ramcoffers,
  title  = {RAM Coffers: Neuromorphic Cognitive-Domain Routing for
            Heterogeneous LLM Inference},
  author = {C., Scott and Elya, Sophia},
  year   = {2025},
  month  = dec,
  doi    = {10.5281/zenodo.18717767},
  url    = {https://zenodo.org/records/18717767},
  note   = {Preprint. Priority date: December 16, 2025.
            Published February 20, 2026.}
}
```

---

## See Also

- [exo](https://github.com/exo-explore/exo) — heterogeneous distributed inference
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — CPU/GPU LLM inference
- [DeepSeek Engram](https://arxiv.org/abs/2601.07372) — domain memory routing (Jan 2026)
- [Neuromorphic computing](https://en.wikipedia.org/wiki/Neuromorphic_computing) — brain-inspired hardware architecture
