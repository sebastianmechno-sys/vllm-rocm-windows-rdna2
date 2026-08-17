import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODE = os.environ.get("BENCH_MODE", "eager")
MAXLEN = int(os.environ.get("BENCH_MAXLEN", "4096"))
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "512"))
MODEL = os.environ.get("BENCH_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")

import torch
from vllm import LLM, SamplingParams

GIB = 2**30
print(f"== model={MODEL} mode={MODE} dev={torch.cuda.get_device_name(0)}")
free0, total = torch.cuda.mem_get_info()
print(f"== VRAM total {total / GIB:.1f} GiB free {free0 / GIB:.1f} GiB")

kwargs = dict(
    model=MODEL,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.93,
    max_model_len=MAXLEN,
    attention_backend="TRITON_ATTN",
    kv_cache_dtype=(os.environ.get("BENCH_KVFP8", "0") == "1" and "fp8_e4m3" or "auto"),
    skip_mm_profiling=True,
    limit_mm_per_prompt={"image": 0, "video": 0},
)

MTP = os.environ.get("BENCH_MTP", "0") == "1"
NGRAM = os.environ.get("BENCH_NGRAM", "0") == "1"
if MTP:
    kwargs["speculative_config"] = {
        "method": "mtp",
        "num_speculative_tokens": int(os.environ.get("BENCH_NSPEC", "4")),
    }
elif NGRAM:
    kwargs["speculative_config"] = {
        "method": "ngram",
        "num_speculative_tokens": int(os.environ.get("BENCH_NSPEC", "5")),
        "prompt_lookup_min": int(os.environ.get("BENCH_PLMIN", "2")),
        "prompt_lookup_max": int(os.environ.get("BENCH_PLMAX", "7")),
    }

if MODE == "eager":
    kwargs["enforce_eager"] = True
elif MODE == "graph":
    kwargs["enforce_eager"] = False
    kwargs["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
elif MODE == "compile":
    kwargs["enforce_eager"] = False
    kwargs["compilation_config"] = {"mode": 3}
elif MODE == "graph+compile":
    kwargs["enforce_eager"] = False
    kwargs["compilation_config"] = {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}
else:
    raise SystemExit(f"unknown mode {MODE}")

import time

t0 = time.perf_counter()
llm = LLM(**kwargs)
print(f"== engine init {time.perf_counter() - t0:.0f}s")
free1, _ = torch.cuda.mem_get_info()
print(f"== VRAM used after init: {(total - free1) / GIB:.2f} GiB")

tok = llm.get_tokenizer()
question = (
    "Un contadino deve portare una volpe, una gallina e un sacco di grano "
    "dall'altra parte di un fiume con una barca che puo' portare solo lui e "
    "una cosa alla volta. Se lasciati soli, la volpe mangia la gallina e la "
    "gallina mangia il grano. Risolvi passo passo."
)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
)

sp_warm = SamplingParams(temperature=0.0, max_tokens=16)
llm.generate([prompt], sp_warm)
torch.cuda.synchronize()

sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)
t0 = time.perf_counter()
out = llm.generate([prompt], sp)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

o = out[0]
n_out = len(o.outputs[0].token_ids)
n_in = len(o.prompt_token_ids)
m = o.metrics
print("=" * 60)
print("OUTPUT (first 1200 chars):")
print(o.outputs[0].text[:1200])
print("=" * 60)
print(f"prompt_tokens={n_in} generated_tokens={n_out} wall={dt:.2f}s")
print(f"overall_tok_s={n_out / dt:.2f}")
if m is not None and m.first_scheduled_time and m.finished_time:
    dec = n_out / (m.finished_time - m.first_scheduled_time)
    print(f"engine_tok_s={dec:.2f}")
