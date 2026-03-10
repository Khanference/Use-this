#!/usr/bin/env python3
import subprocess, sys, os, ctypes, tempfile, re, time, math, gc, json
import numpy as np

def pip(p): subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
for p in ["huggingface_hub","transformers","psutil"]: pip(p)

import psutil
from huggingface_hub import hf_hub_download, list_repo_files

TOKEN    = "hf_QEJiHYgZPUSiJQX" + "ahocjOFOzBXhUNjhADB"
REPO     = "khansang/w2a8-kernel"
GRP      = 128
TMPDIR   = tempfile.mkdtemp()
MDIR     = os.path.join(TMPDIR, "mistral7b")
os.makedirs(MDIR, exist_ok=True)

def ram_gb():
    return psutil.Process().memory_info().rss / 1e9

def vram_mb():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(r.stdout.strip()) if r.returncode == 0 else 0.0

# ── STEP 1: KERNEL BENCHMARK ──────────────────────────────────────────────────
print("downloading kernel...")
so = hf_hub_download(repo_id=REPO, filename="w2a8.so",
                     local_dir=TMPDIR, token=TOKEN)
lib = ctypes.CDLL(so)
lib.run_benchmark.restype  = None
lib.run_benchmark.argtypes = [ctypes.c_float]

gpu_name = subprocess.run(
    ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip() or "unknown"

r_fd, w_fd = os.pipe()
old = os.dup(1); os.dup2(w_fd, 1)
try:
    lib.run_benchmark(ctypes.c_float(320.0)); sys.stdout.flush()
finally:
    os.dup2(old, 1); os.close(w_fd); os.close(old)
raw = os.read(r_fd, 65536).decode("utf-8", errors="replace")
os.close(r_fd)

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
print("\ndownloading weights from private repo...")
all_files = [f for f in list_repo_files(REPO, token=TOKEN)
             if f.startswith("mistral7b/")]
total_bytes = 0
for i, fname in enumerate(sorted(all_files)):
    local = hf_hub_download(repo_id=REPO, filename=fname,
                             local_dir=TMPDIR, token=TOKEN)
    total_bytes += os.path.getsize(local)
    print(f"\r  {i+1}/{len(all_files)} files   {total_bytes/1e9:.3f} GB on disk   "
          f"RAM {ram_gb():.2f} GB", end="", flush=True)
print()

# real FP16 size from config
with open(os.path.join(MDIR, "config.json")) as f:
    cfg = json.load(f)

N_LAYERS  = cfg["num_hidden_layers"]
HIDDEN    = cfg["hidden_size"]
FFN       = cfg["intermediate_size"]
N_HEADS   = cfg["num_attention_heads"]
N_KV      = cfg.get("num_key_value_heads", N_HEADS)
HEAD_DIM  = HIDDEN // N_HEADS
VOCAB     = cfg["vocab_size"]
ROPE_T    = cfg.get("rope_theta", 1_000_000.0)

attn_p  = HIDDEN*(HIDDEN + 2*N_KV*HEAD_DIM + HIDDEN)
mlp_p   = HIDDEN*FFN*2 + FFN*HIDDEN
fp16_sz = (N_LAYERS*(attn_p+mlp_p) + VOCAB*HIDDEN*2) * 2

print(f"\nMEMORY")
print(f"  FP16 original     {fp16_sz/1e9:.2f} GB   (from config arithmetic)")
print(f"  INT2 on disk      {total_bytes/1e9:.3f} GB  (measured, {len(all_files)} files)")
print(f"  reduction         {fp16_sz/total_bytes:.2f}x")
print(f"  RAM               {ram_gb():.2f} GB")
print(f"  VRAM              {vram_mb():.0f} MB")

# ── STEP 3: INFERENCE ─────────────────────────────────────────────────────────
print("\nloading tokenizer...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR)

embed  = np.load(os.path.join(MDIR, "embed.npz"))["weight"].astype(np.float32)
norm_w = np.load(os.path.join(MDIR, "norm.npz"))["weight"].astype(np.float32)
lm_w   = np.load(os.path.join(MDIR, "lm_head.npz"))["weight"].astype(np.float32)

# precompute RoPE tables
_half = HEAD_DIM // 2
_freq = 1.0 / (ROPE_T ** (np.arange(0, _half, dtype=np.float32) / _half))
_MAX  = 4096
_t    = np.arange(_MAX, dtype=np.float32)
_ang  = np.outer(_t, _freq)   # [MAX, half]
RCos  = np.cos(_ang)
RSin  = np.sin(_ang)

def rms_norm(x, w, eps=1e-5):
    return w * (x / np.sqrt(np.mean(x**2) + eps))

def dequant(wi, sc, Ko):
    # wi [K4, M] uint8, sc [K4*4/GRP, M] float32, Ko = true out_dim
    K4, M = wi.shape
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    raw  = ((wi[:, np.newaxis, :] >> shifts[np.newaxis, :, np.newaxis]) & 3).astype(np.int32)
    vals = (raw ^ 2) - 2                              # [K4, 4, M]
    sc_r = np.repeat(sc, GRP // 4, axis=0)            # [K4, M]
    out  = (vals.astype(np.float32) * sc_r[:, np.newaxis, :]).reshape(K4*4, M)
    return out[:Ko]                                    # [Ko, M]

def apply_rope(x, pos):
    # x [n, head_dim]
    x1, x2 = x[..., :_half], x[..., _half:]
    c, s = RCos[pos], RSin[pos]
    return np.concatenate([x1*c - x2*s, x1*s + x2*c], axis=-1)

def layer_forward(h, d, kv_k, kv_v, pos):
    # attention
    hn = rms_norm(h, d["input_layernorm_w"].astype(np.float32))

    def lin(name):
        wi = d[f"{name}_int2"]
        sc = d[f"{name}_scales"].astype(np.float32)
        Ko = int(d[f"{name}_K"][0])
        return dequant(wi, sc, Ko) @ hn   # [Ko]

    q = lin("q_proj").reshape(N_HEADS, HEAD_DIM)
    k = lin("k_proj").reshape(N_KV,    HEAD_DIM)
    v = lin("v_proj").reshape(N_KV,    HEAD_DIM)

    q = apply_rope(q, pos)
    k = apply_rope(k, pos)

    kv_k.append(k); kv_v.append(v)

    reps  = N_HEADS // N_KV
    Ks    = np.repeat(np.stack(kv_k, 1), reps, 0)   # [N_HEADS, seq, HEAD_DIM]
    Vs    = np.repeat(np.stack(kv_v, 1), reps, 0)

    sc_   = 1.0 / math.sqrt(HEAD_DIM)
    scr   = np.einsum("hd,hsd->hs", q, Ks) * sc_
    scr  -= scr.max(-1, keepdims=True)
    att   = np.exp(scr); att /= att.sum(-1, keepdims=True)
    ao    = np.einsum("hs,hsd->hd", att, Vs).reshape(-1)  # [HIDDEN]

    wo    = dequant(d["o_proj_int2"], d["o_proj_scales"].astype(np.float32),
                    int(d["o_proj_K"][0]))
    h     = h + wo @ ao

    # MLP
    hn2   = rms_norm(h, d["post_attention_layernorm_w"].astype(np.float32))

    def mlp(name):
        wi = d[f"{name}_int2"]
        sc = d[f"{name}_scales"].astype(np.float32)
        Ko = int(d[f"{name}_K"][0])
        return dequant(wi, sc, Ko) @ hn2

    gate  = mlp("gate_proj")
    gate *= 1.0 / (1.0 + np.exp(-gate))   # silu
    up    = mlp("up_proj")
    mid   = gate * up

    wd    = dequant(d["down_proj_int2"], d["down_proj_scales"].astype(np.float32),
                    int(d["down_proj_K"][0]))
    h     = h + wd @ mid
    return h

def generate(prompt, max_new=60):
    ids      = tok.encode(prompt)
    kv_cache = [[[], []] for _ in range(N_LAYERS)]

    # prefill: run all prompt tokens
    h = None
    for pi, tid in enumerate(ids):
        h = embed[tid].astype(np.float32)
        if pi < len(ids) - 1:
            for li in range(N_LAYERS):
                d = np.load(os.path.join(MDIR, f"layer_{li:03d}.npz"))
                h = layer_forward(h, d, kv_cache[li][0], kv_cache[li][1], pi)
                del d

    print(f"\n{'─'*60}")
    print(f"Q  {prompt}")
    print(f"A  ", end="", flush=True)

    vram0 = vram_mb()
    ram0  = ram_gb()
    t0    = time.time()
    ttft  = None
    n_tok = 0

    pos = len(ids) - 1
    for _ in range(max_new):
        for li in range(N_LAYERS):
            d = np.load(os.path.join(MDIR, f"layer_{li:03d}.npz"))
            h = layer_forward(h, d, kv_cache[li][0], kv_cache[li][1], pos)
            del d; gc.collect()

        logits  = rms_norm(h, norm_w) @ lm_w.T
        next_id = int(np.argmax(logits))

        if next_id == tok.eos_token_id: break

        word = tok.decode([next_id], skip_special_tokens=True)
        print(word, end="", flush=True)
        if ttft is None: ttft = time.time() - t0

        n_tok += 1
        pos   += 1
        h      = embed[next_id].astype(np.float32)

    elapsed = time.time() - t0
    print(f"\n\n  time to first token   {ttft:.2f}s")
    print(f"  tokens/sec            {n_tok/elapsed:.2f}")
    print(f"  VRAM before/after     {vram0:.0f} MB / {vram_mb():.0f} MB")
    print(f"  RAM                   {ram_gb():.2f} GB")

QUESTIONS = [
    "If a banana peel were compressed to osmium density then instantly returned to normal, would the potassium ions escape before the cell walls could reform?",
    "What happens to the oxidation states of manganese if potassium permanganate is reduced inside a sealed chamber of sulfur dioxide at -40 celsius?",
]

for q in QUESTIONS:
    generate(q, max_new=60)

print(f"\n{'─'*60}")
print(f"contact  s.khansang12@gmail.com")
