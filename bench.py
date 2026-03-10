#!/usr/bin/env python3
import subprocess, sys, os, ctypes, tempfile, re, time, math, gc, json
import numpy as np

def pip(p): subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
for p in ["huggingface_hub","transformers","psutil"]: pip(p)

import psutil
from huggingface_hub import hf_hub_download, list_repo_files

TOKEN  = "hf_QEJiHYgZPUSiJQX" + "ahocjOFOzBXhUNjhADB"
REPO   = "khansang/w2a8-kernel"
GRP    = 128
TMPDIR = tempfile.mkdtemp()
MDIR   = os.path.join(TMPDIR, "mistral7b")
os.makedirs(MDIR, exist_ok=True)

def ram_gb():  return psutil.Process().memory_info().rss / 1e9
def vram_mb():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else 0.0

# ── STEP 1: KERNEL BENCHMARK ──────────────────────────────────────────────────
print("downloading kernel...")
so = hf_hub_download(repo_id=REPO, filename="w2a8.so",
                     local_dir=TMPDIR, token=TOKEN)
lib = ctypes.CDLL(so)
lib.run_benchmark.restype  = None
lib.run_benchmark.argtypes = [ctypes.c_float]

gpu_name = subprocess.run(
    ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip().splitlines()[0] or "unknown"

import tempfile as _tf
_tmp = _tf.mktemp(suffix=".txt")
_libc = ctypes.CDLL(None)
_libc.fflush(None)
_old = os.dup(1)
_fd  = os.open(_tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.dup2(_fd, 1); os.close(_fd)
try:
    lib.run_benchmark(ctypes.c_float(320.0))
    _libc.fflush(None)
finally:
    os.dup2(_old, 1); os.close(_old)
with open(_tmp, "r", errors="replace") as _f: raw = _f.read()
os.unlink(_tmp)

pat  = re.compile(r'^\s{2}(\S.*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s+[\d.]+%\s+([\d.]+)x', re.M)
rows = pat.findall(raw)

def fmt(label, ms, gbs, spd):
    return f"  {label:<26} {ms:>8} ms   {gbs:>7} GB/s   {spd:>5}x"

print(f"\nGPU  {gpu_name}\n")
if len(rows) >= 4:
    print("GEMV  M=4096 K=4096")
    print(fmt(rows[0][0].strip(), rows[0][1], rows[0][2], rows[0][3]))
    print(fmt(rows[1][0].strip(), rows[1][1], rows[1][2], rows[1][3]))
    print()
    print("KV ATTENTION  heads=32 head_dim=128 seq_len=8192")
    print(fmt(rows[2][0].strip(), rows[2][1], rows[2][2], rows[2][3]))
    print(fmt(rows[3][0].strip(), rows[3][1], rows[3][2], rows[3][3]))
else:
    print(raw)

# ── STEP 2: DOWNLOAD WEIGHTS ──────────────────────────────────────────────────
print("\ndownloading weights...")
all_files   = sorted(f for f in list_repo_files(REPO, token=TOKEN)
                     if f.startswith("mistral7b/"))
total_bytes = 0
for i, fname in enumerate(all_files):
    local        = hf_hub_download(repo_id=REPO, filename=fname,
                                   local_dir=TMPDIR, token=TOKEN)
    total_bytes += os.path.getsize(local)
    print(f"\r  {i+1}/{len(all_files)}   {total_bytes/1e9:.3f} GB   RAM {ram_gb():.2f} GB",
          end="", flush=True)
print()

with open(os.path.join(MDIR, "config.json")) as f: cfg = json.load(f)
N_LAYERS = cfg["num_hidden_layers"]
HIDDEN   = cfg["hidden_size"]
FFN      = cfg["intermediate_size"]
N_HEADS  = cfg["num_attention_heads"]
N_KV     = cfg.get("num_key_value_heads", N_HEADS)
HEAD_DIM = HIDDEN // N_HEADS
VOCAB    = cfg["vocab_size"]
ROPE_T   = cfg.get("rope_theta", 1_000_000.0)

fp16_sz = (N_LAYERS * (HIDDEN*(HIDDEN + 2*N_KV*HEAD_DIM + HIDDEN) + HIDDEN*FFN*2 + FFN*HIDDEN)
           + VOCAB * HIDDEN * 2) * 2

print(f"\nMEMORY")
print(f"  FP16 original   {fp16_sz/1e9:.2f} GB")
print(f"  INT2 on disk    {total_bytes/1e9:.3f} GB   ({len(all_files)} files, measured)")
print(f"  reduction       {fp16_sz/total_bytes:.2f}x")
print(f"  RAM now         {ram_gb():.2f} GB")
print(f"  VRAM now        {vram_mb():.0f} MB")

# ── STEP 3: INFERENCE ─────────────────────────────────────────────────────────
print("\nloading tokenizer...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR)

embed  = np.load(os.path.join(MDIR, "embed.npz"))["weight"].astype(np.float32)
norm_w = np.load(os.path.join(MDIR, "norm.npz"))["weight"].astype(np.float32)
lm_w   = np.load(os.path.join(MDIR, "lm_head.npz"))["weight"].astype(np.float32)

# try GPU matmuls via cupy, fall back to numpy
try:
    import cupy as cp
    xp = cp
    print("matmuls: GPU (cupy)")
    embed  = cp.asarray(embed)
    norm_w = cp.asarray(norm_w)
    lm_w   = cp.asarray(lm_w)
except ImportError:
    xp = np
    print("matmuls: CPU (numpy)")

_half = HEAD_DIM // 2
_freq = 1.0 / (ROPE_T ** (np.arange(0, _half, dtype=np.float32) / _half))
_ang  = np.outer(np.arange(4096, dtype=np.float32), _freq)
RCos, RSin = np.cos(_ang), np.sin(_ang)
if xp is not np:
    RCos = cp.asarray(RCos); RSin = cp.asarray(RSin)

print("preloading layers into RAM...")
LAYERS = []
for li in range(N_LAYERS):
    d = np.load(os.path.join(MDIR, f"layer_{li:03d}.npz"))
    layer = {k: d[k] for k in d.files}
    if xp is not np:
        # move weight matrices to GPU
        for k in layer:
            if k.endswith("_int2") or k.endswith("_scales"):
                layer[k] = cp.asarray(layer[k])
    LAYERS.append(layer)
    print(f"\r  {li+1}/{N_LAYERS}   RAM {ram_gb():.2f} GB", end="", flush=True)
print()

def rms_norm(x, w, eps=1e-5):
    return w * (x / xp.sqrt(xp.mean(x**2) + eps))

def dequant(wi, sc, Ko):
    # always numpy — uint8 bitshift on GPU gives wrong results
    if hasattr(wi, "get"): wi = wi.get()
    if hasattr(sc, "get"): sc = sc.get()
    K4, M  = wi.shape
    shifts = np.array([0,2,4,6], dtype=np.uint8)
    raw    = ((wi[:,np.newaxis,:] >> shifts[np.newaxis,:,np.newaxis]) & 3).astype(np.int32)
    vals   = (raw ^ 2) - 2
    sc_r   = np.repeat(sc.astype(np.float32), GRP//4, axis=0)
    out    = (vals.astype(np.float32) * sc_r[:,np.newaxis,:]).reshape(K4*4, M)[:Ko]
    return cp.asarray(out) if xp is not np else out

def rope(x, pos):
    x1, x2 = x[...,:_half], x[...,_half:]
    c = RCos[pos] if xp is np else cp.asarray(RCos[pos])
    s = RSin[pos] if xp is np else cp.asarray(RSin[pos])
    return xp.concatenate([x1*c - x2*s, x1*s + x2*c], axis=-1)

def forward(h, d, kv_k, kv_v, pos):
    hn = rms_norm(h, d["input_layernorm_w"].astype(xp.float32) if xp is np
                  else cp.asarray(d["input_layernorm_w"].astype(np.float32)))
    def lin(n):
        wi = d[f"{n}_int2"]
        sc = d[f"{n}_scales"].astype(np.float32) if xp is np else cp.asarray(d[f"{n}_scales"])
        Ko = int(d[f"{n}_K"][0])
        return dequant(wi, sc, Ko) @ hn
    q = rope(lin("q_proj").reshape(N_HEADS, HEAD_DIM), pos)
    k = rope(lin("k_proj").reshape(N_KV,    HEAD_DIM), pos)
    v =      lin("v_proj").reshape(N_KV,    HEAD_DIM)
    kv_k.append(k); kv_v.append(v)
    reps = N_HEADS // N_KV
    Ks   = xp.repeat(xp.stack(kv_k, 1), reps, 0)
    Vs   = xp.repeat(xp.stack(kv_v, 1), reps, 0)
    sc_  = 1.0 / math.sqrt(HEAD_DIM)
    att  = xp.einsum("hd,hsd->hs", q, Ks) * sc_
    att -= att.max(-1, keepdims=True)
    att  = xp.exp(att); att /= att.sum(-1, keepdims=True)
    ao   = xp.einsum("hs,hsd->hd", att, Vs).reshape(-1)
    h    = h + dequant(d["o_proj_int2"],
                       d["o_proj_scales"].astype(np.float32) if xp is np
                       else cp.asarray(d["o_proj_scales"]),
                       int(d["o_proj_K"][0])) @ ao
    hn2  = rms_norm(h, d["post_attention_layernorm_w"].astype(xp.float32) if xp is np
                    else cp.asarray(d["post_attention_layernorm_w"].astype(np.float32)))
    def mlp(n):
        wi = d[f"{n}_int2"]
        sc = d[f"{n}_scales"].astype(np.float32) if xp is np else cp.asarray(d[f"{n}_scales"])
        Ko = int(d[f"{n}_K"][0])
        return dequant(wi, sc, Ko) @ hn2
    g = mlp("gate_proj"); g *= 1/(1+xp.exp(-g))
    h = h + dequant(d["down_proj_int2"],
                    d["down_proj_scales"].astype(np.float32) if xp is np
                    else cp.asarray(d["down_proj_scales"]),
                    int(d["down_proj_K"][0])) @ (g * mlp("up_proj"))
    return h

def generate(prompt, max_new=60):
    ids      = tok.encode(prompt)
    kv_cache = [[[], []] for _ in range(N_LAYERS)]
    # prefill — every token through all layers, h = last token hidden state
    for pi, tid in enumerate(ids):
        h = embed[tid].astype(xp.float32)
        for li in range(N_LAYERS):
            h = forward(h, LAYERS[li], kv_cache[li][0], kv_cache[li][1], pi)

    print(f"\n{'─'*64}")
    print(f"Q  {prompt}")
    print(f"A  ", end="", flush=True)

    vram0 = vram_mb(); t0 = time.time(); ttft = None; n_tok = 0
    pos   = len(ids) - 1

    for _ in range(max_new):
        for li in range(N_LAYERS):
            h = forward(h, LAYERS[li], kv_cache[li][0], kv_cache[li][1], pos)
        logits  = lm_w @ rms_norm(h, norm_w)
        next_id = int(xp.argmax(logits))
        if next_id == tok.eos_token_id: break
        word = tok.decode([next_id], skip_special_tokens=True)
        print(word, end="", flush=True)
        if ttft is None: ttft = time.time() - t0
        n_tok += 1; pos += 1
        h = embed[next_id].astype(xp.float32)

    elapsed = time.time() - t0
    print(f"\n\n  time to first token   {ttft:.2f}s")
    print(f"  tokens/sec            {n_tok/elapsed:.2f}")
    print(f"  VRAM before/peak      {vram0:.0f} MB / {vram_mb():.0f} MB")
    print(f"  RAM                   {ram_gb():.2f} GB")

QUESTIONS = [
    "If a banana peel were compressed to osmium density then instantly returned to normal, would the potassium ions escape before the cell walls could reform?",
    "What happens to the oxidation states of manganese if potassium permanganate is reduced inside a sealed chamber of sulfur dioxide at -40 celsius?",
]

for q in QUESTIONS:
    generate(q, max_new=60)

print(f"\n{'─'*64}")
print("contact  s.khansang12@gmail.com")
