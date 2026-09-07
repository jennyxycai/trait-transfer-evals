"""Shared helpers for the Team-6 extension evals (sales, livecode, customer_service, data_entry).

* import_authors(repo_path, sub) -- put an authors' sub-package on sys.path and import it UNMODIFIED.
* SandboxExec -- run a Python file / shell command as a subprocess with timeout, rlimits and (when the
  kernel allows it) `unshare -rn` network isolation. Replaces the authors' docker / inspect sandboxes
  (Docker is not usable on this cluster; see DEVIATIONS.md).
* AuxLLM -- a second OpenAI-compatible endpoint used for the simulated customer (customer_service) and the
  LLM graders (data_entry). The authors used OpenAI gpt-4.1 / gpt-4.1-nano; both OPENAI_API_KEY and
  GEMINI_API_KEY in ~/.sigil_env were rejected (401 / 400) on 2026-09-03, so the default is a local vLLM
  server (Qwen/Qwen3-4B, thinking off) started by serve_and_run.sbatch when AUX_MODEL_PATH is set.
"""
import asyncio
import importlib
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THINK_CLOSE = "</think>"


def import_authors(repo_path: Path, subdir: str, module: str):
    p = str((Path(repo_path) / subdir).resolve()) if subdir else str(Path(repo_path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module(module)


def split_think(raw: str):
    """(reasoning, final). Split on the LAST </think>; no tag -> ('', raw)."""
    if raw is None:
        return "", ""
    i = raw.rfind(THINK_CLOSE)
    if i < 0:
        return "", raw
    r = raw[:i]
    if r.startswith("<think>"):
        r = r[len("<think>"):]
    return r, raw[i + len(THINK_CLOSE):].lstrip("\n")


# ----------------------------------------------------------------------------- sandbox
_UNSHARE_OK = None


def unshare_ok() -> bool:
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        try:
            r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
            _UNSHARE_OK = (r.returncode == 0)
        except Exception:
            _UNSHARE_OK = False
    return _UNSHARE_OK


def _limits(mem_mb: int):
    def pre():
        os.setsid()
        try:
            b = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
            resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024))
            # no RLIMIT_NPROC: under `unshare -r` it counts every process of the real uid and forks fail
        except Exception:
            pass
    return pre


def run_cmd_sandboxed(cmd: list, timeout: float, cwd=None, mem_mb: int = 4096, stdin: str | None = None,
                     env: dict | None = None, no_net: bool = True):
    """Run cmd as a subprocess (own process group, rlimits, optional `unshare -rn`). Returns dict."""
    full = (["unshare", "-rn"] if (no_net and unshare_ok()) else []) + list(cmd)
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.Popen(full, cwd=cwd, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                             preexec_fn=_limits(mem_mb))
        try:
            out, err = p.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
            out, err = p.communicate()
        rc = p.returncode
    except Exception as e:  # noqa
        return {"success": False, "returncode": -1, "stdout": "", "stderr": f"spawn error: {e!r}",
                "timed_out": False, "wall_s": round(time.time() - t0, 3)}
    return {"success": (rc == 0 and not timed_out), "returncode": rc, "stdout": out or "", "stderr": err or "",
            "timed_out": timed_out, "wall_s": round(time.time() - t0, 3)}


class ExecPool:
    """Thread pool so blocking subprocess runs do not stall the asyncio event loop."""

    def __init__(self, workers: int = 8):
        self.pool = ThreadPoolExecutor(max_workers=workers)

    async def run(self, fn, *a, **kw):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, lambda: fn(*a, **kw))


# ----------------------------------------------------------------------------- aux LLM
class AuxLLM:
    """Second chat model (simulated customer / LLM grader) on an OpenAI-compatible endpoint."""

    def __init__(self, base_url, model, api_key="dummy", temperature=0.7, top_p=0.8, max_tokens=1024,
                 chat_template_kwargs=None, timeout=600.0, max_retries=3, concurrency=32):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model, self.temperature, self.top_p, self.max_tokens = model, temperature, top_p, max_tokens
        self.extra_body = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else None
        self.max_retries = max_retries
        self.sem = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def describe(self):
        return {"base_url": str(self.client.base_url), "model": self.model, "temperature": self.temperature,
                "top_p": self.top_p, "max_tokens": self.max_tokens, "extra_body": self.extra_body}

    async def chat(self, messages, seed=None, temperature=None, max_tokens=None):
        kw = dict(model=self.model, messages=messages, temperature=self.temperature if temperature is None else temperature,
                  top_p=self.top_p, max_tokens=max_tokens or self.max_tokens)
        if seed is not None:
            kw["seed"] = seed
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        last = None
        for attempt in range(self.max_retries):
            try:
                async with self.sem:
                    r = await self.client.chat.completions.create(**kw)
                msg = r.choices[0].message
                content = msg.content or ""
                sr = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                if sr:
                    content = f"{sr}{THINK_CLOSE}\n{content}"
                _, final = split_think(content)
                self.calls += 1
                if r.usage:
                    self.prompt_tokens += r.usage.prompt_tokens or 0
                    self.completion_tokens += r.usage.completion_tokens or 0
                return {"raw_content": content, "final": final, "finish_reason": r.choices[0].finish_reason,
                        "usage": r.usage.model_dump() if r.usage else None}
            except Exception as e:  # noqa
                last = e
                await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"aux request failed after {self.max_retries} attempts: {last!r}")

    def usage(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}
