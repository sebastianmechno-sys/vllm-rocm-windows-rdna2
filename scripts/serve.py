import os
import subprocess
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("VLLM_WIN_BF16_GEMV", "1")
os.environ.setdefault("VLLM_WIN_HIPGEMV", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get(
    "SERVED_MODEL",
    r"C:\Users\sebas\.cache\huggingface\hub\models--cyankiwi--Qwen3.5-4B-AWQ-4bit\snapshots\ef85d23bebaba87b3c4672ba11c449c79dbdb23e",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3.5-4B")
PORT = os.environ.get("SERVE_PORT", "8000")

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
    "--default-chat-template-kwargs", '{"enable_thinking": true}',
    "--attention-backend", "TRITON_ATTN",
    "--skip-mm-profiling",
    "--limit-mm-per-prompt", '{"image":0,"video":0}',
    "--compilation-config", '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}',
]
sys.exit(subprocess.call(cmd))
