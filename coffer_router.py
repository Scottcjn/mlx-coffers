#!/usr/bin/env python3
"""
Coffer Router — size-aware + domain-aware matmul node selector

Maintains a registry of matmul nodes (M2 MLX, CUDA V100, CPU fallback),
advertises capabilities, routes each matmul request to the best available
node based on:

  1. Coffer domain  (LEFT_HEMI → M2 CPU, RIGHT_HEMI → CUDA V100)
  2. Matrix size    (small < threshold → M2, large → CUDA)
  3. Node health    (health-checked every 10s, failed nodes sidelined)
  4. Fallback chain (CUDA V100 → M2 Metal → M2 CPU → local numpy)

Usage as library:
    from coffer_router import CofferRouter, CofferDomain
    router = CofferRouter()
    router.add_node(CofferNode("cuda-v100",  "192.168.0.161", 8096, arch="x86_64"))
    router.add_node(CofferNode("mlx-m2",     "192.168.0.134", 8096, arch="arm64"))
    router.start()                          # begin health checks

    node = router.route(M=512, N=512, K=512, domain=CofferDomain.LEFT_HEMI)
    with node.connection() as conn:
        conn.matmul(A_data, B_data, ...)

Usage as HTTP service (for distributed setups):
    python3 coffer_router.py --serve --port 8098
    # Then POST /route  {"M": 512, "N": 512, "K": 512, "domain": "LEFT_HEMI"}
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

import numpy as np


# ─── Domain enum (mirrors mlx_coffers.py) ────────────────────────────────────

class CofferDomain(Enum):
    PREFRONTAL  = 0   # executive, embedding, lm_head    → fast GPU
    LEFT_HEMI   = 1   # attention Q/K/V (small matrices) → CPU preferred
    RIGHT_HEMI  = 2   # MLP gate/up/down (large matmuls) → GPU preferred
    TEMPORAL    = 3   # KV cache, context                → CPU large buf
    UNKNOWN     = -1  # domain not specified


# ─── GGML type constants ──────────────────────────────────────────────────────

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
MAGIC_V3 = 0x47505533
QK8_0 = 32
QK_K  = 256

TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K"}


def get_type_size(tensor_type: int, elements: int) -> int:
    if tensor_type == GGML_TYPE_F32:  return elements * 4
    if tensor_type == GGML_TYPE_F16:  return elements * 2
    if tensor_type == GGML_TYPE_Q8_0: return ((elements + QK8_0 - 1) // QK8_0) * 34
    if tensor_type == GGML_TYPE_Q4_K: return ((elements + QK_K  - 1) // QK_K ) * 148
    if tensor_type == GGML_TYPE_Q6_K: return ((elements + QK_K  - 1) // QK_K ) * 210
    raise ValueError(f"Unknown type: {tensor_type}")


# ─── Node definition ──────────────────────────────────────────────────────────

@dataclass
class CofferNode:
    name:          str
    host:          str
    port:          int           # TCP matmul port
    arch:          str  = "x86_64"   # "arm64" | "x86_64" | "ppc64"
    caps_port:     int  = 8097       # HTTP capabilities port
    gpu_type:      str  = "unknown"
    quant_types:   List[str] = field(default_factory=lambda: ["F32", "F16"])

    # Runtime state (set by health checker)
    healthy:       bool  = True
    latency_ms:    float = 999.0
    last_ping:     float = 0.0
    gpu_threshold: int   = 512 * 512  # elements — when this node prefers GPU stream

    # Preferred domains (populated from /capabilities)
    cpu_domains:   List[CofferDomain] = field(default_factory=list)
    gpu_domains:   List[CofferDomain] = field(default_factory=list)

    # Persistent TCP connection pool
    _conn_lock:    threading.Lock = field(default_factory=threading.Lock, repr=False)
    _conn:         Optional[socket.socket] = field(default=None, repr=False)

    def __post_init__(self):
        self._conn_lock = threading.Lock()
        self._conn = None
        # Defaults based on arch
        if self.arch == "arm64":
            self.cpu_domains = [CofferDomain.LEFT_HEMI, CofferDomain.TEMPORAL]
            self.gpu_domains = [CofferDomain.RIGHT_HEMI, CofferDomain.PREFRONTAL]
        else:  # x86_64 CUDA
            self.gpu_domains = [CofferDomain.RIGHT_HEMI, CofferDomain.PREFRONTAL,
                                 CofferDomain.LEFT_HEMI]

    def capabilities_url(self) -> str:
        return f"http://{self.host}:{self.caps_port}/capabilities"

    def health_url(self) -> str:
        return f"http://{self.host}:{self.caps_port}/health"

    def fetch_capabilities(self) -> bool:
        """Fetch /capabilities and update node properties. Returns True on success.

        Falls back to a raw TCP ping on the matmul port when the HTTP caps server
        (port caps_port) is not present — allows plain CUDA servers without the
        MLX-style HTTP sidecar to still be used as healthy routing targets.
        """
        # Try HTTP caps server first (full metadata)
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(self.health_url(), timeout=2) as r:
                self.latency_ms = (time.perf_counter() - t0) * 1000
            with urllib.request.urlopen(self.capabilities_url(), timeout=3) as r:
                caps = json.loads(r.read())
            self.gpu_type      = caps.get("gpu_type", self.gpu_type)
            self.quant_types   = caps.get("quant_types", self.quant_types)
            self.gpu_threshold = caps.get("gpu_stream_threshold_elements", self.gpu_threshold)
            cr = caps.get("coffer_routing", {})
            if cr:
                cpu_dom = cr.get("cpu_stream", "LEFT_HEMI")
                gpu_dom = cr.get("gpu_stream", "RIGHT_HEMI")
                self.cpu_domains = [CofferDomain[cpu_dom]] if cpu_dom else self.cpu_domains
                self.gpu_domains = [CofferDomain[gpu_dom]] if gpu_dom else self.gpu_domains
            self.healthy   = True
            self.last_ping = time.time()
            return True
        except Exception:
            pass

        # Fallback: TCP check on matmul port (plain CUDA servers have no HTTP sidecar)
        try:
            t0 = time.perf_counter()
            s = socket.create_connection((self.host, self.port), timeout=2)
            self.latency_ms = (time.perf_counter() - t0) * 1000
            s.close()
            self.healthy   = True
            self.last_ping = time.time()
            print(f"[router] {self.name}: HTTP caps unavailable, TCP ping OK "
                  f"({self.latency_ms:.0f}ms) — using defaults")
            return True
        except Exception:
            self.healthy   = False
            self.last_ping = time.time()
            return False

    def get_connection(self) -> socket.socket:
        """Get or create persistent TCP connection to this node's matmul port."""
        with self._conn_lock:
            if self._conn is not None:
                try:
                    # Ping with zero-byte check (will raise if dead)
                    self._conn.setblocking(False)
                    self._conn.recv(1, socket.MSG_PEEK)
                    self._conn.setblocking(True)
                except BlockingIOError:
                    self._conn.setblocking(True)  # No data, connection alive
                except Exception:
                    self._conn = None  # Dead, reconnect

            if self._conn is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                self._conn = s

            return self._conn

    def close_connection(self):
        with self._conn_lock:
            if self._conn:
                try: self._conn.close()
                except Exception: pass
                self._conn = None

    def __repr__(self) -> str:
        status = "✓" if self.healthy else "✗"
        return (f"CofferNode({status} {self.name} {self.host}:{self.port} "
                f"arch={self.arch} lat={self.latency_ms:.0f}ms)")


# ─── Router ───────────────────────────────────────────────────────────────────

class CofferRouter:
    """
    Routes matmul ops to the best available compute node.

    Routing priority order for a given (M, N, K, domain):
    1. Domain match: if a node is preferred for this domain, prefer it
    2. Size match: if M*K >= node's gpu_threshold, prefer GPU-capable node
    3. Health: only route to healthy nodes
    4. Latency: break ties with ping time
    5. Local fallback: always keep a numpy CPU fallback
    """

    HEALTH_INTERVAL = 10.0   # seconds between health checks

    def __init__(self):
        self._nodes:    Dict[str, CofferNode] = {}
        self._lock      = threading.Lock()
        self._running   = False
        self._hc_thread: Optional[threading.Thread] = None

    def add_node(self, node: CofferNode) -> None:
        with self._lock:
            self._nodes[node.name] = node
        print(f"[router] Registered: {node}")

    def remove_node(self, name: str) -> None:
        with self._lock:
            if name in self._nodes:
                self._nodes.pop(name).close_connection()

    def start(self) -> None:
        """Begin background health checking."""
        self._running = True
        # Initial capability fetch for all nodes
        with self._lock:
            nodes = list(self._nodes.values())
        for n in nodes:
            n.fetch_capabilities()
            print(f"[router] Capabilities {n.name}: healthy={n.healthy} "
                  f"lat={n.latency_ms:.0f}ms gpu={n.gpu_type}")

        self._hc_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._hc_thread.start()

    def stop(self) -> None:
        self._running = False

    def _health_loop(self) -> None:
        while self._running:
            time.sleep(self.HEALTH_INTERVAL)
            with self._lock:
                nodes = list(self._nodes.values())
            for n in nodes:
                was_healthy = n.healthy
                n.fetch_capabilities()
                if n.healthy != was_healthy:
                    status = "RECOVERED" if n.healthy else "DOWN"
                    print(f"[router] Node {n.name} {status}")

    def route(
        self,
        M: int, N: int, K: int,
        A_type: int = GGML_TYPE_F16,
        domain:  CofferDomain = CofferDomain.UNKNOWN,
    ) -> Optional[CofferNode]:
        """
        Choose the best node for this matmul. Returns None if no healthy node found
        (caller should fall back to local numpy).
        """
        n_elem_A = M * K
        with self._lock:
            candidates = [n for n in self._nodes.values() if n.healthy]

        if not candidates:
            return None

        def score(node: CofferNode) -> float:
            s = 0.0
            # Domain bonus: strong preference if domain matches node's preferred
            if domain in node.gpu_domains:
                s += 100.0 if n_elem_A >= node.gpu_threshold else 40.0
            elif domain in node.cpu_domains:
                s += 100.0 if n_elem_A < node.gpu_threshold else 40.0
            # Size bonus: GPU nodes for large ops, CPU for small
            if n_elem_A >= node.gpu_threshold and "CUDA" in node.gpu_type.upper():
                s += 60.0
            elif n_elem_A < node.gpu_threshold and node.arch == "arm64":
                s += 50.0
            # Quantization support
            type_name = TYPE_NAMES.get(A_type, "F32")
            if type_name in node.quant_types:
                s += 20.0
            # Latency penalty (subtract 1 point per 10ms)
            s -= node.latency_ms / 10.0
            return s

        best = max(candidates, key=score)
        return best

    def route_and_compute(
        self,
        A_data: bytes, B_data: bytes,
        M: int, N: int, K: int,
        A_type: int = GGML_TYPE_F16,
        B_type: int = GGML_TYPE_F16,
        domain: CofferDomain = CofferDomain.UNKNOWN,
    ) -> np.ndarray:
        """
        Full pipeline: route → send to node → receive result.
        Falls back to local numpy if no node available.
        """
        node = self.route(M, N, K, A_type, domain)

        if node is None:
            print(f"[router] No healthy node, falling back to local numpy")
            return _local_matmul(A_data, B_data, M, N, K, A_type, B_type)

        t0 = time.perf_counter()
        try:
            result = _remote_matmul(node, A_data, B_data, M, N, K, A_type, B_type)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[router] {domain.name} [{M}×{K}]×[{K}×{N}] → {node.name} ({elapsed:.1f}ms)")
            return result
        except Exception as exc:
            print(f"[router] Node {node.name} failed ({exc}), marking unhealthy")
            node.healthy = False
            node.close_connection()
            # Retry with remaining nodes
            return self.route_and_compute(
                A_data, B_data, M, N, K, A_type, B_type, domain
            )

    def status(self) -> str:
        with self._lock:
            nodes = list(self._nodes.values())
        lines = ["─" * 70,
                 f"{'Node':<18} {'Arch':<8} {'GPU':<20} {'Lat':>6} {'Health':<8}",
                 "─" * 70]
        for n in nodes:
            health = "✓ UP" if n.healthy else "✗ DOWN"
            lines.append(
                f"{n.name:<18} {n.arch:<8} {n.gpu_type:<20} "
                f"{n.latency_ms:>5.0f}ms {health:<8}"
            )
        lines.append("─" * 70)
        return "\n".join(lines)


# ─── Remote matmul via v3 protocol ────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Connection closed by remote")
        buf.extend(chunk)
    return bytes(buf)


def _remote_matmul(
    node: CofferNode,
    A_data: bytes, B_data: bytes,
    M: int, N: int, K: int,
    A_type: int, B_type: int,
) -> np.ndarray:
    """Send matmul to a remote node via GPU3 v3 protocol, return FP16 result."""
    hdr = struct.pack('<IIIIII', MAGIC_V3, M, N, K, A_type, B_type)

    sock = node.get_connection()
    sock.sendall(hdr + A_data + B_data)

    # Receive response header (16 bytes)
    resp_hdr = _recv_exact(sock, 16)
    resp_magic, status, resp_M, resp_N = struct.unpack('<IIII', resp_hdr)

    if resp_magic != MAGIC_V3:
        raise ValueError(f"Bad response magic: {hex(resp_magic)}")
    if status != 0:
        raise RuntimeError(f"Remote compute error (status={status})")

    # Receive result (FP16)
    result_bytes = _recv_exact(sock, resp_M * resp_N * 2)
    return np.frombuffer(result_bytes, dtype=np.float16).reshape(resp_M, resp_N).copy()


# ─── Local numpy fallback ─────────────────────────────────────────────────────

def _dequantize_numpy(data: bytes, tensor_type: int, n_elements: int) -> np.ndarray:
    """Minimal dequantization for local fallback (F32/F16 only for speed)."""
    if tensor_type == GGML_TYPE_F32:
        return np.frombuffer(data, dtype=np.float32).astype(np.float16)
    elif tensor_type == GGML_TYPE_F16:
        return np.frombuffer(data, dtype=np.float16).copy()
    else:
        # Import the full dequant from mlx_matmul_server if available
        try:
            from mlx_matmul_server import dequantize
            return dequantize(data, tensor_type, n_elements)
        except ImportError:
            raise ValueError(f"Cannot dequantize type {tensor_type} without mlx_matmul_server")


def _local_matmul(
    A_data: bytes, B_data: bytes,
    M: int, N: int, K: int,
    A_type: int, B_type: int,
) -> np.ndarray:
    A = _dequantize_numpy(A_data, A_type, M * K).reshape(M, K)
    B = _dequantize_numpy(B_data, B_type, K * N).reshape(K, N)
    C = A.astype(np.float32) @ B.astype(np.float32)
    return C.astype(np.float16)


# ─── HTTP service (optional) ──────────────────────────────────────────────────

_global_router: Optional[CofferRouter] = None


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/status":
            body = _global_router.status().encode() if _global_router else b"No router"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/nodes":
            with _global_router._lock:
                nodes = [
                    {"name": n.name, "host": n.host, "port": n.port,
                     "arch": n.arch, "gpu_type": n.gpu_type,
                     "healthy": n.healthy, "latency_ms": round(n.latency_ms, 1)}
                    for n in _global_router._nodes.values()
                ]
            body = json.dumps(nodes, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/route":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            M = body.get("M", 512)
            N = body.get("N", 512)
            K = body.get("K", 512)
            A_type = body.get("A_type", GGML_TYPE_F16)
            domain_str = body.get("domain", "UNKNOWN")
            try:
                domain = CofferDomain[domain_str]
            except KeyError:
                domain = CofferDomain.UNKNOWN

            node = _global_router.route(M, N, K, A_type, domain) if _global_router else None
            if node:
                resp = {"node": node.name, "host": node.host, "port": node.port,
                        "latency_ms": round(node.latency_ms, 1), "arch": node.arch}
            else:
                resp = {"node": "local_fallback", "host": "127.0.0.1", "port": 0}

            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()


def serve(port: int = 8098) -> None:
    global _global_router
    srv = HTTPServer(("0.0.0.0", port), RouterHandler)
    print(f"[router] HTTP service on :{port}  GET /status /nodes  POST /route")
    srv.serve_forever()


# ─── Default cluster config ───────────────────────────────────────────────────

def build_default_router(verbose: bool = True) -> CofferRouter:
    """
    Build a router pre-configured for the Elyan Labs cluster.
    Each node's health is verified before the router starts.
    """
    router = CofferRouter()

    nodes = [
        CofferNode(
            name="cuda-gpu",
            host="192.168.0.161",
            port=8096,
            caps_port=8097,
            arch="x86_64",
            gpu_type="CUDA GPU",   # overwritten by /capabilities on startup
            quant_types=["F32", "F16", "Q8_0", "Q4_K", "Q6_K"],
        ),
        CofferNode(
            name="mlx-m2",
            host="192.168.0.134",
            port=8096,
            caps_port=8097,
            arch="arm64",
            gpu_type="Apple Metal M2",
            quant_types=["F32", "F16", "Q8_0", "Q4_K", "Q6_K"],
        ),
    ]

    for n in nodes:
        router.add_node(n)

    router.start()

    if verbose:
        print()
        print(router.status())

    return router


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Coffer Router")
    parser.add_argument("--serve", action="store_true", help="Run HTTP routing service")
    parser.add_argument("--port",  type=int, default=8098, help="HTTP port")
    parser.add_argument("--test",  action="store_true", help="Run routing test")
    args = parser.parse_args()

    router = build_default_router(verbose=True)

    if args.test:
        print("\n[test] Routing examples:")
        cases = [
            (32, 512, 512,   GGML_TYPE_F16, CofferDomain.LEFT_HEMI,   "attention Q-proj (small)"),
            (32, 2048, 512,  GGML_TYPE_Q4_K, CofferDomain.RIGHT_HEMI, "MLP gate (large Q4_K)"),
            (1,  32000, 512, GGML_TYPE_F16, CofferDomain.PREFRONTAL,   "lm_head projection"),
            (16, 1024, 4096, GGML_TYPE_Q4_K, CofferDomain.RIGHT_HEMI, "MLP up_proj (very large)"),
        ]
        print(f"  {'Description':<35} {'Domain':<14} → {'Node':>12}")
        print("  " + "─" * 68)
        for M, N, K, A_type, domain, desc in cases:
            node = router.route(M, N, K, A_type, domain)
            node_name = node.name if node else "local_fallback"
            print(f"  {desc:<35} {domain.name:<14} → {node_name:>12}")
        print()

    if args.serve:
        _global_router = router
        serve(args.port)
    elif not args.test:
        # Interactive status
        print("\nRouter running. Ctrl-C to exit.")
        try:
            while True:
                time.sleep(30)
                print(router.status())
        except KeyboardInterrupt:
            print("\n[router] Stopped")
