# w2a8

INT2 weight quantization with fused INT2 KV attention for LLM inference on CUDA.
No Tensor Cores required. Runs on sm_75 (T4) and newer.

## Benchmark


GEMV  M=4096 K=4096

  W2A8 INT2       0.1247 ms   143.2 GB/s   4.73x cuBLAS
  cuBLAS FP16     0.5898 ms   227.6 GB/s   1.00x

KV ATTENTION  heads=32 head_dim=128 seq_len=8192

  INT2 fused      0.0540 ms   174.8 GB/s   5.61x FP16
  FP16 baseline   0.3028 ms   223.4 GB/s   1.00x

MEMORY

  weights         13.96 GB  >  1.96 GB   7.11x
  KV cache        FP16 never materialized   8x


Tested on Tesla T4, sm_75, 320 GB/s peak bandwidth.

 Prior work, 

bitsandbytes INT8: 2x weight reduction, no KV quantization  
AWQ / GPTQ INT4: 3.9x weight reduction, no fused KV kernel  
KIVI: INT2 KV cache, dequantizes to FP16 before compute  
BitDecoding: fused KV but requires Tensor Cores (A100+)

This kernel fuses INT2 dequantization into dp4a. FP16 K never exists in VRAM.

RUN:
%%bash
curl -sSL https://raw.githubusercontent.com/Khanference/Use-this/main/bench.py | python3

Requires CUDA GPU, Linux, Python 3.8+.

Contact for whatever

s.khansang12@gmail.com
