import glob
import os
import signal
import subprocess
import time
import urllib.request
from urllib.parse import urlparse
import logging

from openai import AsyncOpenAI, OpenAI

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class VLLMInference:
    def __init__(self, model: str = "Qwen/Qwen3-32B", base_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.async_client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    def completion(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs) -> str:
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].text

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
            extra_body={"enable_thinking": thinking},
            **kwargs,
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content

        return content, reasoning_content

    async def async_chat(self, prompt: str, system: str | None = None, max_completion_tokens: int = 1024, temperature: float = 0.7, thinking: bool = False, top_k: int | None = None, **kwargs) -> tuple[str, str | None]:
        """Returns (content, thinking). thinking is None if enable_thinking=False."""
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
        reasoning = getattr(message, "reasoning_content", None)
        return content, reasoning

    # Device paths JAX/libtpu hold on TPU VMs
    _TPU_DEV_PATTERNS = ["/dev/accel*", "/dev/vfio/*"]

    @staticmethod
    def free_tpu_devices() -> int:
        """SIGKILL any processes holding TPU VFIO devices. Returns count killed."""
        my_pid = os.getpid()
        tpu_devs = set()
        for pattern in VLLMInference._TPU_DEV_PATTERNS:
            tpu_devs.update(glob.glob(pattern))

        killed = 0
        for dev in sorted(tpu_devs):
            result = subprocess.run(["fuser", dev], capture_output=True, text=True)
            for token in result.stdout.split():
                if not token.isdigit():
                    continue
                pid = int(token)
                if pid == my_pid:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except ProcessLookupError:
                    pass
        if killed:
            time.sleep(2)
        return killed

    @staticmethod
    def start_server(
        model: str,
        base_url: str,
        tensor_parallel_size: int = 8,
        tpu_visible_devices: str | None = None,
        download_dir: str = "/dev/shm",
        max_model_len: int = 8192,
        max_num_seqs: int | None = None,
        free_tpu: bool = True,
    ) -> subprocess.Popen:
        """Launch a vLLM server subprocess. Returns the Popen handle.

        If free_tpu=True (default), any process holding /dev/accel* devices is
        killed first so vLLM can acquire the TPU.
        """
        if free_tpu:
            VLLMInference.free_tpu_devices()
        port = urlparse(base_url).port or 8000
        cmd = [
            "uv", "run", "--no-sync", "vllm", "serve", model,
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
        if os.environ.get("VLLM_TPU_LOCAL_ONLY") == "1":
            # Multi-host slice host (runbook §2.3): confine the judge to this host's chips.
            # Without this, libtpu reads the slice topology from instance metadata and the
            # vLLM engine proc dies waiting for its peer host ("Engine core initialization
            # failed"). Bounds assume a ct6e-standard-4t host (2x2 chips).
            env["TPU_SKIP_MDS_QUERY"] = "1"
            env["TPU_PROCESS_BOUNDS"] = "1,1,1"
            env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,2,1"
        if tpu_visible_devices is not None:
            env["TPU_VISIBLE_DEVICES"] = tpu_visible_devices
        return subprocess.Popen(cmd, env=env)

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