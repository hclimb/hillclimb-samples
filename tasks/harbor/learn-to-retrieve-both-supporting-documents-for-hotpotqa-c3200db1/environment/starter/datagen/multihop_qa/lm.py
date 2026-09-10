"""Shared vLLM client for retrieval and generation modules."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "persistent_tpu"))
from vllm_inference import VLLMInference

VLLM_MODEL    = os.getenv("VLLM_MODEL",    "meta-llama/Llama-3.3-70B-Instruct")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")

VLLM_TP_SIZE      = int(os.getenv("VLLM_TP_SIZE",      "8"))
VLLM_TPU_DEVICES  = os.getenv("VLLM_TPU_DEVICES",      "0,1,2,3,4,5,6,7")
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "256"))

_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not VLLMInference.is_server_ready(VLLM_BASE_URL):
            print(f"Starting vLLM server for {VLLM_MODEL}...")
            VLLMInference.start_server(
                model=VLLM_MODEL,
                base_url=VLLM_BASE_URL,
                tensor_parallel_size=VLLM_TP_SIZE,
                tpu_visible_devices=VLLM_TPU_DEVICES,
                max_num_seqs=VLLM_MAX_NUM_SEQS,
                max_model_len=32768,
                download_dir="/dev/shm",
            )
            VLLMInference.wait_for_server(VLLM_BASE_URL, timeout=1200)
        else:
            print(f"vLLM server already running at {VLLM_BASE_URL}")
        _client = VLLMInference(model=VLLM_MODEL, base_url=VLLM_BASE_URL)
    return _client


def chat(system, prompt, max_new_tokens=64, temperature=0.6, retries=3):
    for attempt in range(retries):
        try:
            client = get_client()
            return client.chat(
                prompt=prompt,
                system=system,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                thinking=False,
            ).strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
