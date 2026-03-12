# w2a8

INT2 weight quantization with fused INT2 KV attention for LLM inference on CUDA. No Tensor Cores required, runs on sm_75 (T4) and newer.

Built in ~3 days on free Colab/Kaggle hardware. Not finished. Kernel benchmarks are real. End-to-end inference is broken in known ways. I'm not incentivized to fix it further right now, leaving everything here in case it's useful to someone.

## Benchmark

```
GEMV  M=4096 K=4096
  W2A8 INT2       0.1247 ms   143.2 GB/s   4.73x cuBLAS
  cuBLAS FP16     0.5898 ms   227.6 GB/s   1.00x

KV ATTENTION  heads=32 head_dim=128 seq_len=8192
  INT2 fused      0.0540 ms   174.8 GB/s   5.61x FP16
  FP16 baseline   0.3028 ms   223.4 GB/s   1.00x
```

Tested on Tesla T4, sm_75, 320 GB/s peak bandwidth.

## Memory

| | Before | After | Reduction |
|---|---|---|---|
| Weights | 13.96 GB | 1.96 GB | 7.11x |
| KV cache | FP16 | INT2 (never materialized) | 8x |

## What works

- GEMV kernel: INT2 weights, INT8 activations, dp4a on sm_75+
- Fused INT2 KV attention kernel — FP16 K never exists in VRAM
- Quantized Qwen2.5-32B: 64 GB FP16 → 10.3 GB, fits on a single free T4
- One-line reproducible benchmark (see below)

## What's broken and why

**Perplexity is ~15M (should be ~7-8 for a healthy 2-bit model).** Root cause: `launch_kv_int2_fused_attn` is a bare `__global__` kernel with no host C wrapper in the `.so` — ctypes launches it but the output buffer stays zero, softmax becomes uniform, model outputs garbage. The numpy fallback in bench.py works around this but PPL is still measured over only 80 tokens with a non-standard sliding window, so the number isn't comparable to published benchmarks anyway.

**Generation outputs repetitive tokens** ("mädchen" repeating) — same root cause, uniform attention.

**0.36 tok/s** — PCIe bottleneck from numpy CPU einsum on V buffer every layer. Fix is known.

All of these are one-line or one-day fixes. I just don't have the compute credits or time right now.

## Prior work

- **bitsandbytes INT8**: 2x weight reduction, no KV quantization
- **AWQ / GPTQ INT4**: 3.9x weight reduction, no fused KV kernel
- **KIVI**: INT2 KV cache, dequantizes to FP16 before compute
- **BitDecoding**: fused KV but requires Tensor Cores (A100+)

This kernel fuses INT2 dequantization directly into dp4a. FP16 K is never materialized in VRAM.

## Run

!curl -sSL https://raw.githubusercontent.com/Khanference/Use-this/main/bench.py | python3

Requires CUDA GPU (sm_75+), Linux, Python 3.8+.

## Contact

s.khansang12@gmail.com
