"""Does a 4-bit build of Gemma 4 26B A4B keep Google's QAT weights exactly?

    python3 grid_check.py [tensor-prefix] > evidence/grid-check-26b.txt

Reads one tensor from each checkpoint with HTTP range requests (safetensors
header, then only that tensor's bytes), dequantizes the two compressed-tensors
builds (level x scale) and compares them with Google's QAT export, whose bf16
values already lie on a 4-bit grid.
"""

import json
import struct
import sys
import urllib.request

import numpy as np

SOURCE = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
BUILDS = ["xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct", "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.0.self_attn.q_proj"
GROUP = 32


def get(url, start=None, end=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"} if start is not None else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def shard_for(repo, name):
    base = f"https://huggingface.co/{repo}/resolve/main/"
    try:
        index = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
        return base + index[name]
    except Exception:
        return base + "model.safetensors"


def read(repo, name):
    url = shard_for(repo, name)
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    meta = header[name]
    a, b = meta["data_offsets"]
    raw = get(url, 8 + n + a, 8 + n + b - 1)
    dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I32": np.int32, "I64": np.int64}[
        meta["dtype"]
    ]
    arr = np.frombuffer(raw, dtype=dt).reshape(meta["shape"])
    if meta["dtype"] == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32) if arr.dtype != np.int32 else arr, meta["dtype"]


def to_bf16(x):
    """Round float32 to the nearest bfloat16 (ties to even), returned as float32."""
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def dequant(repo):
    packed, _ = read(repo, PREFIX + ".weight_packed")
    scale, sdt = read(repo, PREFIX + ".weight_scale")
    nib = np.stack([(packed >> (4 * i)) & 0xF for i in range(8)], axis=-1).reshape(packed.shape[0], -1)
    levels = nib.astype(np.int32) - 8
    return levels, scale, sdt


w, wdt = read(SOURCE, PREFIX + ".weight")
out, inp = w.shape
print(f"# {PREFIX}: source {SOURCE} {wdt} shape {list(w.shape)}, {out * inp // GROUP} groups of {GROUP}")
for repo in BUILDS:
    levels, scale, sdt = dequant(repo)
    deq = to_bf16(levels.reshape(out, inp // GROUP, GROUP) * scale[..., None])  # the source is bf16
    src = w.reshape(out, inp // GROUP, GROUP)
    exact = deq == src
    exact_groups = np.all(exact, axis=-1)
    # A level is "on the QAT grid" if dequantizing it reproduces the source weight to within
    # half a step of the build's own scale (i.e. the same level, whatever rounding of the scale).
    rel = np.abs(deq - src) / np.maximum(np.abs(scale[..., None]), 1e-30)
    off = int((rel > 0.5).sum())
    print(
        f"{repo}: scale dtype {sdt}; groups bit-identical {int(exact_groups.sum())}/{exact_groups.size} "
        f"({exact_groups.mean():.2%}); values bit-identical {exact.mean():.2%}; weights a full level or more off the source {off}/{w.size} ({off / w.size:.2%}); "
        f"max |dequant - source| {float(np.abs(deq - src).max()):.3g}, mean {float(np.abs(deq - src).mean()):.3g}"
    )
