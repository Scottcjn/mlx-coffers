#!/usr/bin/env python3
"""
distributed_matmul.py — Drop-in distributed matmul client

Replaces direct connections to port 8096 with smart coffer-routed dispatch.
The caller specifies the op's cognitive domain; the router picks the best node.

Drop-in replacement example:
    # Before (hardcoded to CUDA node):
    conn = MatmulConnection("192.168.0.161", 8096)
    C = conn.matmul(A_bytes, B_bytes, M, N, K, GGML_TYPE_Q4_K)

    # After (coffer-routed):
    conn = DistributedMatmul(router)
    C = conn.matmul(A_bytes, B_bytes, M, N, K, GGML_TYPE_Q4_K,
                    domain=CofferDomain.RIGHT_HEMI)

The domain hint drives routing — if you don't know the domain,
pass CofferDomain.UNKNOWN and the router falls back to size-based heuristics.
"""

from __future__ import annotations

import struct
import socket
import time
from typing import Optional

import numpy as np

from coffer_router import (
    CofferRouter, CofferDomain, CofferNode,
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0,
    MAGIC_V3, get_type_size, build_default_router,
    _remote_matmul, _local_matmul,
)


# ─── Simple direct connection (existing pattern, no routing) ──────────────────

class MatmulConnection:
    """Direct connection to a single matmul node (original behaviour)."""

    def __init__(self, host: str, port: int = 8096):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._lock = __import__("threading").Lock()

    def _get_sock(self) -> socket.socket:
        with self._lock:
            if self._sock is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self.host, self.port))
                self._sock = s
            return self._sock

    def matmul(
        self,
        A_data: bytes, B_data: bytes,
        M: int, N: int, K: int,
        A_type: int = GGML_TYPE_F16,
        B_type: int = GGML_TYPE_F16,
    ) -> np.ndarray:
        hdr  = struct.pack('<IIIIII', MAGIC_V3, M, N, K, A_type, B_type)
        sock = self._get_sock()
        sock.sendall(hdr + A_data + B_data)

        # Receive 16-byte response header
        buf = bytearray()
        while len(buf) < 16:
            buf.extend(sock.recv(16 - len(buf)))
        _, status, rM, rN = struct.unpack('<IIII', buf)

        if status != 0:
            raise RuntimeError(f"Remote error status={status}")

        # Receive FP16 result
        n_bytes = rM * rN * 2
        result  = bytearray()
        while len(result) < n_bytes:
            result.extend(sock.recv(n_bytes - len(result)))

        return np.frombuffer(result, dtype=np.float16).reshape(rM, rN).copy()

    def close(self):
        with self._lock:
            if self._sock:
                self._sock.close()
                self._sock = None


# ─── Distributed client (coffer-routed) ──────────────────────────────────────

class DistributedMatmul:
    """
    Coffer-routed matmul client.

    Routes each request to the best available node based on:
    - Cognitive domain hint (LEFT_HEMI → M2 CPU, RIGHT_HEMI → CUDA GPU)
    - Matrix size (small → M2, large → CUDA V100)
    - Node health (auto-failover)

    Falls back to local numpy if no nodes are reachable.
    """

    def __init__(self, router: CofferRouter):
        self.router = router
        self._stats = {
            "requests":     0,
            "by_node":      {},
            "by_domain":    {},
            "total_ms":     0.0,
            "fallbacks":    0,
        }
        self._lock = __import__("threading").Lock()

    def matmul(
        self,
        A_data: bytes, B_data: bytes,
        M: int, N: int, K: int,
        A_type:  int          = GGML_TYPE_F16,
        B_type:  int          = GGML_TYPE_F16,
        domain:  CofferDomain = CofferDomain.UNKNOWN,
        layer_name: str       = "",
    ) -> np.ndarray:
        """
        Route and execute a matmul. Returns FP16 numpy array [M, N].

        Args:
            A_data:     Raw bytes of A matrix (any GGML quantization)
            B_data:     Raw bytes of B matrix
            M, N, K:    Matrix dimensions (A is M×K, B is K×N, result is M×N)
            A_type:     GGML type of A (GGML_TYPE_F16, GGML_TYPE_Q4_K, etc.)
            B_type:     GGML type of B
            domain:     Cognitive domain hint for routing
            layer_name: Optional layer name (improves domain auto-detection)
        """
        # Auto-detect domain from layer name if not specified
        if domain == CofferDomain.UNKNOWN and layer_name:
            domain = _domain_from_layer(layer_name)

        t0   = time.perf_counter()
        node = self.router.route(M, N, K, A_type, domain)

        if node is None:
            result = _local_matmul(A_data, B_data, M, N, K, A_type, B_type)
            node_name = "local"
            with self._lock:
                self._stats["fallbacks"] += 1
        else:
            try:
                result    = _remote_matmul(node, A_data, B_data, M, N, K, A_type, B_type)
                node_name = node.name
            except Exception as exc:
                print(f"[dist-matmul] Node {node.name} failed: {exc} — falling back")
                node.healthy = False
                node.close_connection()
                result    = _local_matmul(A_data, B_data, M, N, K, A_type, B_type)
                node_name = "local"
                with self._lock:
                    self._stats["fallbacks"] += 1

        elapsed = (time.perf_counter() - t0) * 1000

        with self._lock:
            self._stats["requests"] += 1
            self._stats["total_ms"] += elapsed
            self._stats["by_node"][node_name] = self._stats["by_node"].get(node_name, 0) + 1
            dn = domain.name
            self._stats["by_domain"][dn] = self._stats["by_domain"].get(dn, 0) + 1

        return result

    def stats(self) -> str:
        with self._lock:
            s = dict(self._stats)
        reqs = s["requests"] or 1
        avg  = s["total_ms"] / reqs
        lines = [
            f"DistributedMatmul stats: {reqs} requests | avg {avg:.1f}ms | {s['fallbacks']} fallbacks",
            "  By node:   " + "  ".join(f"{k}={v}" for k, v in s["by_node"].items()),
            "  By domain: " + "  ".join(f"{k}={v}" for k, v in s["by_domain"].items()),
        ]
        return "\n".join(lines)


# ─── Domain auto-detection from layer name ────────────────────────────────────

_ATTN_KEYWORDS  = ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attn")
_MLP_KEYWORDS   = ("mlp", "gate_proj", "up_proj", "down_proj", "ffn", "fc1", "fc2")
_EXEC_KEYWORDS  = ("embed", "norm", "lm_head", "layer_norm", "ln_")
_KV_KEYWORDS    = ("kv_cache", "past_key", "cache")


def _domain_from_layer(name: str) -> CofferDomain:
    lower = name.lower()
    if any(k in lower for k in _KV_KEYWORDS):  return CofferDomain.TEMPORAL
    if any(k in lower for k in _ATTN_KEYWORDS): return CofferDomain.LEFT_HEMI
    if any(k in lower for k in _MLP_KEYWORDS):  return CofferDomain.RIGHT_HEMI
    if any(k in lower for k in _EXEC_KEYWORDS): return CofferDomain.PREFRONTAL
    return CofferDomain.UNKNOWN


# ─── End-to-end test ─────────────────────────────────────────────────────────

def test_end_to_end(router: CofferRouter) -> None:
    """
    Sends real matmuls through the router and verifies results.
    Compares remote FP16 results against local numpy reference.
    """
    print("=" * 60)
    print("  End-to-End Distributed Matmul Test")
    print("=" * 60)
    print()

    client = DistributedMatmul(router)

    test_cases = [
        # (M, K, N,  A_type,        domain,                  desc)
        # ── Small ops: should route to M2 Metal ──
        (16,   512,  512, GGML_TYPE_F16, CofferDomain.LEFT_HEMI,  "attn q_proj (F16, small)"),
        (1,    512,  512, GGML_TYPE_F16, CofferDomain.PREFRONTAL, "lm_head single token"),
        (16,   512,  512, GGML_TYPE_F16, CofferDomain.UNKNOWN,    "unknown domain (size routing)"),
        # ── LLM-realistic MLP ops: should route to CUDA at large size ──
        (32,  4096, 4096, GGML_TYPE_F16, CofferDomain.RIGHT_HEMI, "MLP gate_proj 7B (F16, large)"),
        (64,  4096, 4096, GGML_TYPE_F16, CofferDomain.RIGHT_HEMI, "MLP up_proj 7B (F16, large)"),
        (32,  4096,11008, GGML_TYPE_F16, CofferDomain.RIGHT_HEMI, "MLP ffn Llama2 (F16, XL)"),
    ]

    print(f"  {'Description':<38} {'Domain':<14} {'Node':>14} {'ms':>7} {'Match'}")
    print("  " + "─" * 82)

    all_passed = True

    for M, K, N, A_type, domain, desc in test_cases:
        # Generate random test data matching the declared type
        np_dtype = np.float32 if A_type == GGML_TYPE_F32 else np.float16
        A_np = np.random.randn(M, K).astype(np_dtype)
        B_np = np.random.randn(K, N).astype(np_dtype)
        A_bytes = A_np.tobytes()
        B_bytes = B_np.tobytes()

        # Reference result (local numpy in F32 precision, then cast back)
        C_ref = (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(np_dtype)

        t0 = time.perf_counter()
        try:
            C_dist = client.matmul(A_bytes, B_bytes, M, N, K, A_type, A_type, domain)
            elapsed = (time.perf_counter() - t0) * 1000

            # Get node name from stats
            node = router.route(M, N, K, A_type, domain)
            node_name = node.name if node else "local"

            # Compare (generous threshold — server computes in FP16 regardless of input type)
            max_err = float(np.max(np.abs(C_dist.astype(np.float32) - C_ref.astype(np.float32))))
            passed  = max_err < 1.0
            match   = "✓" if passed else f"✗ err={max_err:.3f}"

            if not passed:
                all_passed = False

            print(f"  {desc:<38} {domain.name:<14} {node_name:>14} {elapsed:>6.1f}ms {match}")

        except Exception as exc:
            print(f"  {desc:<38} {domain.name:<14} {'ERROR':>14}        ✗ {exc}")
            all_passed = False

        # Reset node health after each test so one failure doesn't strand the rest
        with router._lock:
            for n in router._nodes.values():
                n.healthy = True

    print()
    print(client.stats())
    print()
    print("  " + ("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗"))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed matmul test client")
    parser.add_argument("--cuda-host",  default="192.168.0.161", help="CUDA node host")
    parser.add_argument("--mlx-host",   default="192.168.0.134", help="M2 MLX node host")
    parser.add_argument("--no-cuda",    action="store_true")
    parser.add_argument("--no-mlx",     action="store_true")
    args = parser.parse_args()

    router = CofferRouter()

    if not args.no_cuda:
        from coffer_router import CofferNode
        router.add_node(CofferNode(
            name="cuda-gpu", host=args.cuda_host, port=8096, caps_port=8097,
            arch="x86_64", gpu_type="CUDA GPU",   # overwritten by /capabilities fetch
            quant_types=["F32", "F16", "Q8_0", "Q4_K", "Q6_K"],
        ))

    if not args.no_mlx:
        from coffer_router import CofferNode
        router.add_node(CofferNode(
            name="mlx-m2", host=args.mlx_host, port=8096, caps_port=8097,
            arch="arm64", gpu_type="Apple Metal M2",
            quant_types=["F32", "F16", "Q8_0", "Q4_K", "Q6_K"],
        ))

    print("[dist] Starting router...")
    router.start()

    print()
    print(router.status())
    print()

    test_end_to_end(router)
