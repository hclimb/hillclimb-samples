import os
import re
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

from openai import AsyncOpenAI, OpenAI


class VLLMInference:
    def __init__(self, model: str = "Qwen/Qwen3-32B", base_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.async_client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    def chat(self, prompt: str, system: str | None = None, max_completion_tokens: int = 1024, temperature: float = 0.7, thinking: bool = False, top_k: int | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": thinking}
        if top_k is not None:
            kwargs["extra_body"]["top_k"] = top_k

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    async def async_chat(self, prompt: str, system: str | None = None, max_completion_tokens: int = 1024, temperature: float = 0.7, thinking: bool = False, top_k: int | None = None, **kwargs) -> tuple[str, str | None]:
        """Returns (content, thinking_trace). thinking_trace is None if enable_thinking=False."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": thinking}
        if top_k is not None:
            kwargs["extra_body"]["top_k"] = top_k

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **kwargs,
        )
        message = response.choices[0].message
        content = message.content or ""
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        think = think_match.group(1).strip() if think_match else None
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content, think

    @staticmethod
    def start_server(
        model: str,
        base_url: str,
        tensor_parallel_size: int = 8,
        tpu_visible_devices: str | None = None,
        download_dir: str = "/dev/shm",
        max_model_len: int = 8192,
        max_num_seqs: int | None = None,
    ) -> subprocess.Popen:
        """Launch a vLLM server subprocess. Returns the Popen handle."""
        port = urlparse(base_url).port or 8000
        cmd = [
            "uv", "run", "vllm", "serve", model,
            "--port", str(port),
            "--download_dir", download_dir,
            "--disable-log-requests",
            "--disable-uvicorn-access-log",
            "--no-enable-log-requests",
            "--disable-log-stats",
            "--tensor_parallel_size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
        ]
        if max_num_seqs is not None:
            cmd += ["--max-num-seqs", str(max_num_seqs)]
        env = os.environ.copy()
        env["VLLM_CONFIGURE_LOGGING"] = "0"
        env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
        if tpu_visible_devices is not None:
            env["TPU_VISIBLE_DEVICES"] = tpu_visible_devices
        return subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)

    @staticmethod
    def is_server_ready(base_url: str) -> bool:
        parsed = urlparse(base_url)
        health_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/health"
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return True
        except Exception:
            return False

    @staticmethod
    def wait_for_server(base_url: str, timeout: int = 1800, interval: float = 5.0) -> None:
        """Block until the server's /health endpoint responds or timeout is reached."""
        print(f"Waiting for server at {base_url}...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if VLLMInference.is_server_ready(base_url):
                print(f"Server at {base_url} is ready.")
                return
            time.sleep(interval)
        raise TimeoutError(f"Server at {base_url} did not become ready within {timeout}s")
