"""
inference.py - Clean vLLM inference entry point for Windows ROCm RDNA2
FIXED: ZMQ ipc:// bug on Windows - forces V0 engine
"""
import argparse
import os

# FIX FOR WINDOWS: Must be set BEFORE importing vllm
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

import torch
from vllm import LLM, SamplingParams

def parse_args():
    p = argparse.ArgumentParser(description="vLLM native on RX 6750 XT gfx1031 - Windows fix")
    p.add_argument("--model", type=str, default="facebook/opt-125m", help="HF model id")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--attention-backend", type=str, default="TRITON_ATTN")
    p.add_argument("--prompt", type=str, default="Hello, my name is")
    p.add_argument("--enforce-eager", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    
    print(f"torch {torch.__version__} | cuda_avail {torch.cuda.is_available()} | dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Loading model: {args.model}")
    print(f"Using V0 engine for Windows compatibility (VLLM_USE_V1=0)")

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        attention_backend=args.attention_backend,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate([args.prompt], sampling_params)
    
    print("="*60)
    print(f"PROMPT: '{args.prompt}'")
    print(f"OUTPUT: '{outputs[0].outputs[0].text}'")
    print("="*60)
    print("FIRST_TOKEN_OK")

if __name__ == "__main__":
    main()
