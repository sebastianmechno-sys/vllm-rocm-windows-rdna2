import os
import subprocess
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("VLLM_WIN_BF16_GEMV", "1")
os.environ.setdefault("VLLM_WIN_HIPGEMV", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get("SERVED_MODEL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()
PORT = os.environ.get("SERVE_PORT", "8000")
THINKING = os.environ.get("THINKING", "1").strip() == "1"

if not MODEL:
    sys.exit(
        "ERROR: SERVED_MODEL is not set.\n"
        "Run INSTALL.bat first (it writes config.bat), or edit config.bat\n"
        "next to this repo and set SERVED_MODEL + MODEL_NAME, then retry."
    )
if not MODEL_NAME:
    MODEL_NAME = os.path.basename(MODEL)

print(f"== {MODEL_NAME} OpenAI-compatible server")
print(f"== model: {MODEL}")
print(f"== url:   http://127.0.0.1:{PORT}/v1  (chat: run CHAT.bat)")

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--served-model-name", MODEL_NAME,
    "--host", "127.0.0.1",
    "--port", PORT,
    "--dtype", "float16",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.93",
    "--attention-backend", "TRITON_ATTN",
    "--skip-mm-profiling",
    "--limit-mm-per-prompt", '{"image":0,"video":0}',
    "--compilation-config", '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}',
    "--default-chat-template-kwargs", '{"enable_thinking": %s}' % ("true" if THINKING else "false"),
]
sys.exit(subprocess.call(cmd))