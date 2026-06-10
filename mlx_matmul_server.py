#!/usr/bin/env python3
"""
MLX Matmul Server — Apple Silicon backend for the GPU3 v3 protocol

Speaks the exact same binary TCP protocol as matmul_server_v3_quant.py
(CUDA/CuPy version on the C4130), so POWER8 and other clients can use
M2 as a drop-in matmul node — no client changes needed.

Coffer-aware stream routing:
  - Small matmuls (M*K < THRESHOLD): mx.cpu stream  ← LEFT_HEMI attention
  - Large matmuls (M*K >= THRESHOLD): mx.gpu Metal   ← RIGHT_HEMI MLP

Also exposes HTTP /capabilities on port CAPS_PORT so the router can
auto-discover what this node handles.

Deploy to M2 Mac Mini (.134):
    scp mlx_matmul_server.py sophia@192.168.0.134:~/
    ssh sophia@192.168.0.134 'python3 mlx_matmul_server.py'
"""

import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from typing import Optional

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("[mlx-v3] WARNING: MLX not available — using numpy CPU fallback")

# ─── Protocol constants (must match matmul_server_v3_quant.py) ───────────────
MAGIC_V3   = 0x47505533  # "GPU3"
MAGIC_V2   = 0x47505532  # "GPU2" backwards compat
PORT       = 8096
CAPS_PORT  = 8097         # HTTP capabilities advertisement

# GGML type IDs
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K"}

# K-quant block sizes (must match llama.cpp ggml)
QK8_0 = 32
QK_K  = 256

# Coffer routing threshold: below this, use CPU stream; above, Metal GPU
# M2 GPU wins once matrices are large enough to amortize kernel dispatch
# (empirically: crossover ~512×512 on M2, same as measured in test_coffers_synthetic.py)
GPU_STREAM_THRESHOLD = 512 * 512  # elements in A matrix

stats = {
    "requests": 0, "bytes_in": 0, "bytes_out": 0,
    "cpu_stream": 0, "gpu_stream": 0,
    "dequant_ms": 0.0, "matmul_ms": 0.0,
}
stats_lock = threading.Lock()


# ─── Tensor size helpers ──────────────────────────────────────────────────────

def get_type_size(tensor_type: int, elements: int) -> int:
    if tensor_type == GGML_TYPE_F32:
        return elements * 4
    elif tensor_type == GGML_TYPE_F16:
        return elements * 2
    elif tensor_type == GGML_TYPE_Q8_0:
        n_blocks = (elements + QK8_0 - 1) // QK8_0
        return n_blocks * (2 + QK8_0)        # 34 bytes/block
    elif tensor_type == GGML_TYPE_Q4_K:
        n_blocks = (elements + QK_K - 1) // QK_K
        return n_blocks * 148                 # 2+2+12+4+128
    elif tensor_type == GGML_TYPE_Q6_K:
        n_blocks = (elements + QK_K - 1) // QK_K
        return n_blocks * 210                 # 128+64+16+2
    else:
        raise ValueError(f"Unknown tensor type: {tensor_type}")


# ─── CPU dequantization (numpy) ───────────────────────────────────────────────
# Reused from matmul_server_v3_quant.py — identical logic so results match.

def dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    n_blocks = (n_elements + QK8_0 - 1) // QK8_0
    result = np.zeros(n_elements, dtype=np.float16)
    offset = 0
    for i in range(n_blocks):
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        offset += 2
        qs = np.frombuffer(data[offset:offset+QK8_0], dtype=np.int8)
        offset += QK8_0
        start, end = i * QK8_0, min((i + 1) * QK8_0, n_elements)
        result[start:end] = (float(d) * qs[:end - start]).astype(np.float16)
    return result


def dequant_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    n_blocks = (n_elements + QK_K - 1) // QK_K
    result = np.zeros(n_elements, dtype=np.float16)
    offset = 0
    for i in range(n_blocks):
        d    = float(np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]); offset += 2
        dmin = float(np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]); offset += 2
        scales_bytes = np.frombuffer(data[offset:offset+12], dtype=np.uint8); offset += 12
        mins_bytes   = np.frombuffer(data[offset:offset+4],  dtype=np.uint8); offset += 4
        qs           = np.frombuffer(data[offset:offset+128], dtype=np.uint8); offset += 128

        scales = np.zeros(8, dtype=np.float32)
        mins   = np.zeros(8, dtype=np.float32)
        for j in range(8):
            sc_idx = j // 2
            if j % 2 == 0:
                scales[j] = scales_bytes[sc_idx] & 0x3F
                mins[j]   = mins_bytes[j // 2] & 0x0F if j < 4 else (mins_bytes[(j-4) // 2] >> 4) & 0x0F
            else:
                scales[j] = (scales_bytes[sc_idx] >> 6) | ((scales_bytes[sc_idx + 4] & 0x0F) << 2)
                mins[j]   = (mins_bytes[j // 2] >> 4) & 0x0F if j < 4 else mins_bytes[(j-4) // 2] & 0x0F

        start = i * QK_K
        for j in range(8):
            sc = d * scales[j]
            mn = dmin * mins[j]
            for k in range(32):
                idx = start + j * 32 + k
                if idx >= n_elements:
                    break
                q_idx = j * 16 + k // 2
                q = qs[q_idx] & 0x0F if k % 2 == 0 else (qs[q_idx] >> 4) & 0x0F
                result[idx] = np.float16(sc * q - mn)
    return result


def dequant_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    n_blocks = (n_elements + QK_K - 1) // QK_K
    result = np.zeros(n_elements, dtype=np.float16)
    offset = 0
    for i in range(n_blocks):
        ql     = np.frombuffer(data[offset:offset+128], dtype=np.uint8);  offset += 128
        qh     = np.frombuffer(data[offset:offset+64],  dtype=np.uint8);  offset += 64
        scales = np.frombuffer(data[offset:offset+16],  dtype=np.int8);   offset += 16
        d      = float(np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]); offset += 2
        start  = i * QK_K
        for j in range(QK_K // 16):
            sc = d * float(scales[j])
            for k in range(16):
                idx = start + j * 16 + k
                if idx >= n_elements:
                    break
                q_idx  = j * 8 + k // 2
                h_idx  = j * 4 + k // 4
                q      = ql[q_idx] & 0x0F if k % 2 == 0 else (ql[q_idx] >> 4) & 0x0F
                h_shift = (k % 4) * 2
                q |= ((qh[h_idx] >> h_shift) & 0x03) << 4
                result[idx] = np.float16(sc * (q - 32))
    return result


def dequantize(data: bytes, tensor_type: int, n_elements: int) -> np.ndarray:
    """Dequantize any supported GGML type to float16 numpy array."""
    if tensor_type == GGML_TYPE_F32:
        return np.frombuffer(data, dtype=np.float32).astype(np.float16)
    elif tensor_type == GGML_TYPE_F16:
        return np.frombuffer(data, dtype=np.float16).copy()
    elif tensor_type == GGML_TYPE_Q8_0:
        return dequant_q8_0(data, n_elements)
    elif tensor_type == GGML_TYPE_Q4_K:
        return dequant_q4_k(data, n_elements)
    elif tensor_type == GGML_TYPE_Q6_K:
        return dequant_q6_k(data, n_elements)
    else:
        raise ValueError(f"Unsupported type: {tensor_type}")


# ─── MLX matmul with coffer-aware stream routing ──────────────────────────────

def mlx_matmul(A_np: np.ndarray, B_np: np.ndarray, M: int, K: int, N: int) -> np.ndarray:
    """
    Run matmul on MLX, routing to CPU or Metal GPU based on op size.

    LEFT_HEMI cognitive domain (attention projections):
      Small matrices → CPU stream (sequential, low kernel-launch overhead)

    RIGHT_HEMI cognitive domain (MLP expansions):
      Large matrices → Metal GPU stream (parallel SIMD, amortizes kernel launch)
    """
    # Reshape
    A = A_np.reshape(M, K)
    B = B_np.reshape(K, N)

    n_elements_A = M * K
    use_gpu = (n_elements_A >= GPU_STREAM_THRESHOLD)

    if not HAS_MLX:
        # Pure numpy fallback (POWER8 or non-Mac testing)
        C = A.astype(np.float32) @ B.astype(np.float32)
        return C.astype(np.float16)

    device = mx.gpu if use_gpu else mx.cpu
    stream_name = "Metal GPU" if use_gpu else "CPU"

    A_mlx = mx.array(A)
    B_mlx = mx.array(B)

    with mx.stream(device):
        C_mlx = mx.matmul(A_mlx.astype(mx.float16), B_mlx.astype(mx.float16))
        mx.eval(C_mlx)

    with stats_lock:
        if use_gpu:
            stats["gpu_stream"] += 1
        else:
            stats["cpu_stream"] += 1

    return np.array(C_mlx).astype(np.float16)


# ─── TCP protocol helpers ─────────────────────────────────────────────────────

def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def send_exact(sock: socket.socket, data: bytes) -> bool:
    total = 0
    while total < len(data):
        sent = sock.send(data[total:])
        if sent == 0:
            return False
        total += sent
    return True


# ─── Client handler ───────────────────────────────────────────────────────────

def handle_client(sock: socket.socket, addr: tuple) -> None:
    print(f"[mlx-v3] Client connected: {addr}")
    try:
        while True:
            # Read magic (4 bytes)
            hdr = recv_exact(sock, 4)
            if not hdr:
                break
            magic = struct.unpack('<I', hdr)[0]

            if magic == MAGIC_V3:
                rest = recv_exact(sock, 20)
                if not rest:
                    break
                M, N, K, A_type, B_type = struct.unpack('<IIIII', rest)
            elif magic == MAGIC_V2:
                rest = recv_exact(sock, 12)
                if not rest:
                    break
                M, N, K = struct.unpack('<III', rest)
                A_type = B_type = GGML_TYPE_F16
            else:
                print(f"[mlx-v3] Bad magic: {hex(magic)}")
                break

            A_size = get_type_size(A_type, M * K)
            B_size = get_type_size(B_type, K * N)

            A_data = recv_exact(sock, A_size)
            B_data = recv_exact(sock, B_size)
            if not A_data or not B_data:
                print("[mlx-v3] Failed to receive matrices")
                break

            with stats_lock:
                stats["requests"] += 1
                stats["bytes_in"] += A_size + B_size

            t0 = time.perf_counter()
            try:
                # Dequantize on CPU (numpy)
                t_dq = time.perf_counter()
                A_np = dequantize(A_data, A_type, M * K)
                B_np = dequantize(B_data, B_type, K * N)
                dequant_ms = (time.perf_counter() - t_dq) * 1000

                # Matmul via MLX (coffer-routed stream)
                t_mm = time.perf_counter()
                C_np = mlx_matmul(A_np, B_np, M, K, N)
                matmul_ms = (time.perf_counter() - t_mm) * 1000

                total_ms = (time.perf_counter() - t0) * 1000
                n_elem_A = M * K
                stream = "Metal GPU" if n_elem_A >= GPU_STREAM_THRESHOLD else "CPU"
                print(
                    f"[mlx-v3] [{M}×{K}]×[{K}×{N}] "
                    f"{TYPE_NAMES.get(A_type,'?')}×{TYPE_NAMES.get(B_type,'?')} → "
                    f"{stream} | dequant={dequant_ms:.1f}ms matmul={matmul_ms:.1f}ms"
                )

                with stats_lock:
                    stats["dequant_ms"] += dequant_ms
                    stats["matmul_ms"]  += matmul_ms

                status = 0

            except Exception as exc:
                print(f"[mlx-v3] Compute error: {exc}")
                import traceback; traceback.print_exc()
                C_np   = np.zeros((M, N), dtype=np.float16)
                status = 1

            # Send response (same format as CUDA server)
            resp_magic = MAGIC_V3 if magic == MAGIC_V3 else MAGIC_V2
            resp_hdr   = struct.pack('<IIII', resp_magic, status, M, N)
            payload    = resp_hdr + C_np.tobytes()

            if not send_exact(sock, payload):
                print("[mlx-v3] Failed to send response")
                break

            with stats_lock:
                stats["bytes_out"] += len(payload)

    except Exception as exc:
        print(f"[mlx-v3] Client error: {exc}")
        import traceback; traceback.print_exc()
    finally:
        sock.close()
        print(f"[mlx-v3] Client disconnected: {addr}")


# ─── HTTP capabilities server ─────────────────────────────────────────────────

CAPABILITIES = {
    "node_type":    "mlx_matmul_server",
    "version":      "1.0.0",
    "arch":         "arm64",
    "vendor":       "Apple",
    "protocol":     "GPU3_v3",
    "matmul_port":  PORT,
    "caps_port":    CAPS_PORT,
    "gpu_type":     "Apple Metal",
    "quant_types":  ["F32", "F16", "Q8_0", "Q4_K", "Q6_K"],
    "gpu_stream_threshold_elements": GPU_STREAM_THRESHOLD,
    "coffer_routing": {
        "cpu_stream":     "LEFT_HEMI",   # attention, small matrices
        "gpu_stream":     "RIGHT_HEMI",  # MLP, large matrices
    },
    "mlx_version":  getattr(__import__("mlx.core", fromlist=[""]), "__version__", "unavailable") if HAS_MLX else "unavailable",
    "preferred_dtype": "F16",
}


class CapsHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress per-request logging

    def do_GET(self):
        if self.path == "/capabilities":
            body = json.dumps({**CAPABILITIES, "stats": dict(stats)}, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def run_caps_server():
    srv = HTTPServer(("0.0.0.0", CAPS_PORT), CapsHandler)
    srv.serve_forever()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import platform
    print("[mlx-v3] MLX Matmul Server — Apple Silicon backend")
    print(f"[mlx-v3] Platform: {platform.machine()} / {platform.system()}")
    print(f"[mlx-v3] MLX: {'available' if HAS_MLX else 'NOT AVAILABLE (numpy fallback)'}")
    if HAS_MLX:
        try:
            info = mx.metal.device_info()
            print(f"[mlx-v3] GPU: {info.get('device_name', 'Apple Metal')}")
        except Exception:
            print("[mlx-v3] GPU: Apple Metal")
    print(f"[mlx-v3] Coffer routing: < {GPU_STREAM_THRESHOLD} elements → CPU | >= → Metal GPU")
    print(f"[mlx-v3] Matmul port: {PORT}  |  Capabilities HTTP: {CAPS_PORT}")
    print()

    # Warm up MLX (first matmul compiles Metal kernels)
    if HAS_MLX:
        print("[mlx-v3] Warming up Metal kernels...")
        dummy = mx.random.normal((512, 512))
        with mx.stream(mx.gpu):
            _ = mx.matmul(dummy, dummy)
            mx.eval(_)
        with mx.stream(mx.cpu):
            _ = mx.matmul(dummy, dummy)
            mx.eval(_)
        print("[mlx-v3] Warmup done")

    # Start capabilities HTTP server in background
    caps_thread = threading.Thread(target=run_caps_server, daemon=True)
    caps_thread.start()
    print(f"[mlx-v3] Capabilities server: http://0.0.0.0:{CAPS_PORT}/capabilities")

    # Start TCP matmul server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(16)
    print(f"[mlx-v3] Listening on TCP port {PORT} (GPU3 v3 protocol)")
    print()

    while True:
        client_sock, addr = server.accept()
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t = threading.Thread(target=handle_client, args=(client_sock, addr), daemon=True)
        t.start()


if __name__ == "__main__":
    main()
