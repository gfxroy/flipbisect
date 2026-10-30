"""White-box backends: launch a server with a given knob config, tear it down.

A backend is just: a dict of knobs (name -> allowed values, first = default) and a
``command(config)`` that turns a config into an argv. llama.cpp is implemented and tested;
vLLM and SGLang ship as flag maps so they plug into the same bisect loop on a GPU box.
"""
from __future__ import annotations
import os, signal, socket, subprocess, time, shutil
from dataclasses import dataclass, field
from .client import Client


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@dataclass
class Backend:
    name: str
    knobs: dict[str, list]
    model: str
    binary: str
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: float = 180

    def defaults(self) -> dict:
        return {k: v[0] for k, v in self.knobs.items()}

    def command(self, cfg: dict, port: int) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def launch(self, cfg: dict) -> "Server":
        port = free_port()
        cmd = self.command({**self.defaults(), **cfg}, port)
        env = dict(os.environ)
        bindir = os.path.dirname(os.path.realpath(self.binary))
        env["LD_LIBRARY_PATH"] = bindir + ":" + env.get("LD_LIBRARY_PATH", "")
        log = open(os.path.join("/tmp", f"flipbisect-{self.name}-{port}.log"), "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        srv = Server(proc, f"http://127.0.0.1:{port}", cmd, log.name)
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited ({proc.returncode}); see {log.name}\n{' '.join(cmd)}")
            if Client(srv.url).health():
                try:
                    Client(srv.url).complete("hi", max_tokens=1, top_logprobs=0)
                    return srv
                except Exception:
                    pass
            time.sleep(0.5)
        srv.stop()
        raise TimeoutError(f"server did not start in {self.startup_timeout}s: {' '.join(cmd)}")


@dataclass
class Server:
    proc: subprocess.Popen
    url: str
    cmd: list[str]
    log: str

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()


LLAMACPP_KNOBS = {
    "threads": [4, 1, 2, 8],
    "batch": [2048, 512, 64],
    "ubatch": [512, 128, 32],
    "parallel": [4, 1, 2, 8],
    "cache_prompt": [True, False],
    "flash_attn": ["off", "on"],
    "cache_type_k": ["f16", "q8_0"],
    "cache_ram": [8192, 0],  # MiB of host-RAM prompt cache (separate from --cache-prompt)
}


class LlamaCpp(Backend):
    def __init__(self, model: str, binary: str | None = None, ctx_per_slot: int = 2048, knobs=None, **kw):
        binary = binary or shutil.which("llama-server") or "llama-server"
        super().__init__("llamacpp", knobs or dict(LLAMACPP_KNOBS), model, binary, **kw)
        self.ctx_per_slot = ctx_per_slot

    def command(self, c: dict, port: int) -> list[str]:
        cmd = [self.binary, "-m", self.model, "--host", "127.0.0.1", "--port", str(port),
               "-t", str(c["threads"]), "-b", str(c["batch"]), "-ub", str(c["ubatch"]),
               "-np", str(c["parallel"]), "-c", str(self.ctx_per_slot * int(c["parallel"])),
               "-fa", str(c["flash_attn"]), "-ctk", str(c["cache_type_k"]), "-cram", str(c["cache_ram"]), "--no-webui"]
        cmd.append("--cache-prompt" if c["cache_prompt"] else "--no-cache-prompt")
        return cmd + list(self.extra_args)


# Flag maps for GPU servers. Same interface; not exercised in CI (no GPU here).
VLLM_KNOBS = {
    "max_num_seqs": [256, 1, 8],
    "enable_prefix_caching": [True, False],
    "enable_chunked_prefill": [True, False],
    "enforce_eager": [False, True],
    "attention_backend": ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"],
    "tensor_parallel_size": [1, 2],
    "batch_invariant": [False, True],
}


class VLLM(Backend):
    def __init__(self, model: str, binary: str = "vllm", knobs=None, **kw):
        super().__init__("vllm", knobs or dict(VLLM_KNOBS), model, binary, startup_timeout=900, **kw)

    def command(self, c: dict, port: int) -> list[str]:
        cmd = [self.binary, "serve", self.model, "--host", "127.0.0.1", "--port", str(port),
               "--max-num-seqs", str(c["max_num_seqs"]), "--tensor-parallel-size", str(c["tensor_parallel_size"])]
        cmd.append("--enable-prefix-caching" if c["enable_prefix_caching"] else "--no-enable-prefix-caching")
        cmd.append("--enable-chunked-prefill" if c["enable_chunked_prefill"] else "--no-enable-chunked-prefill")
        if c["enforce_eager"]:
            cmd.append("--enforce-eager")
        return cmd + list(self.extra_args)

    def launch(self, cfg):
        c = {**self.defaults(), **cfg}
        os.environ["VLLM_ATTENTION_BACKEND"] = c["attention_backend"]
        os.environ["VLLM_BATCH_INVARIANT"] = "1" if c["batch_invariant"] else "0"
        return super().launch(cfg)


SGLANG_KNOBS = {
    "max_running_requests": [None, 1, 8],
    "disable_radix_cache": [False, True],
    "chunked_prefill_size": [None, -1, 512],
    "disable_cuda_graph": [False, True],
    "attention_backend": ["flashinfer", "triton", "fa3"],
    "enable_deterministic_inference": [False, True],
}


class SGLang(Backend):
    def __init__(self, model: str, binary: str = "python", knobs=None, **kw):
        super().__init__("sglang", knobs or dict(SGLANG_KNOBS), model, binary, startup_timeout=900, **kw)

    def command(self, c: dict, port: int) -> list[str]:
        cmd = [self.binary, "-m", "sglang.launch_server", "--model-path", self.model, "--host", "127.0.0.1",
               "--port", str(port), "--attention-backend", c["attention_backend"]]
        if c["max_running_requests"]:
            cmd += ["--max-running-requests", str(c["max_running_requests"])]
        if c["chunked_prefill_size"] is not None:
            cmd += ["--chunked-prefill-size", str(c["chunked_prefill_size"])]
        for k in ("disable_radix_cache", "disable_cuda_graph", "enable_deterministic_inference"):
            if c[k]:
                cmd.append("--" + k.replace("_", "-"))
        return cmd + list(self.extra_args)


BACKENDS = {"llamacpp": LlamaCpp, "vllm": VLLM, "sglang": SGLang}
