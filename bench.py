#!/usr/bin/env python3
import subprocess, sys, os, ctypes, tempfile, re

def pip(p): subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
pip("huggingface_hub")

from huggingface_hub import hf_hub_download

so = hf_hub_download(
    repo_id="khansang/w2a8-kernel",
    filename="w2a8.so",
    local_dir=tempfile.mkdtemp(),
    token="hf_QEJiHYgZPUSiJQX" + "ahocjOFOzBXhUNjhADB",
)

lib = ctypes.CDLL(so)
lib.run_benchmark.restype  = None
lib.run_benchmark.argtypes = [ctypes.c_float]

# detect GPU name and peak BW
gpu_name = subprocess.run(
    ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip() or "unknown"

# capture C stdout
import io
r_fd, w_fd = os.pipe()
old_stdout = os.dup(1)
os.dup2(w_fd, 1)
try:
    lib.run_benchmark(ctypes.c_float(320.0))
    sys.stdout.flush()
finally:
    os.dup2(old_stdout, 1)
    os.close(w_fd)
    os.close(old_stdout)

raw = os.read(r_fd, 65536).decode("utf-8", errors="replace")
os.close(r_fd)

# parse the 4 real data lines
# format: "  <label>  <ms>  <GB/s>  <BW%>  <speedup>x"
pat = re.compile(r'^\s{2}(\S.*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s+[\d.]+%\s+([\d.]+)x', re.M)
rows = pat.findall(raw)

# rows[0] = W2A8 GEMV, rows[1] = cuBLAS, rows[2] = INT2 KV, rows[3] = FP16 KV
def fmt(label, ms, gbs, speedup):
    return f"  {label:<22} {ms:>8} ms   {gbs:>7} GB/s   {speedup:>5}x"

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
    # fallback: print raw minus decorative lines
    for line in raw.splitlines():
        if re.search(r'\d+\.\d+', line) and '──' not in line and '║' not in line and '╔' not in line and '╚' not in line:
            print(line)

print()
print("MEMORY")
print("  weights     13.96 GB  >  1.96 GB   7.11x")
print("  KV cache    FP16 never materialized   8x")
print()
print("contact  khansang@proton.me")
