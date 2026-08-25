import os
import subprocess
import sys

def _load_config_bat():
    """Read config.bat (written by INSTALL.bat next to this script) so serve.py
    works even when launched directly instead of through SERVE.bat."""
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(os.path.dirname(here), "config.bat")
    if not os.path.isfile(cfg):
        return
    with open(cfg, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('set "') and line.endswith('"'):
                try:
                    key, val = line[5:-1].split("=", 1)
                    os.environ.setdefault(key.strip(), val)
                except ValueError:
                    pass

_load_config_bat()

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("VLLM_WIN_BF16_GEMV", "1")
os.environ.setdefault("VLLM_WIN_HIPGEMV", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get("SERVED_MODEL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()
PORT = os.environ.get("SERVE_PORT", "8000")
THINKING = os.environ.get("THINKING", "0").strip() == "1"

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
    # allow the local chat page (chat.html, opened via file:// or http://localhost)
    # to call this API from the browser - without these the UI stays on "connecting..."
    "--allowed-origins", '["*"]',
    "--allowed-methods", '["*"]',
    "--allowed-headers", '["*"]',
]
sys.exit(subprocess.call(cmd))