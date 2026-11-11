"""Periodic resource-aware launcher for independent Python functions (Linux)."""

import errno
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import psutil


def _worker(source, output, error):
    try:
        with open(source, "rb") as f:
            fn, kwargs = pickle.load(f)

        result = fn(**kwargs)

        with open(output, "wb") as f:
            pickle.dump(result, f)

    except BaseException as exc:
        msg = str(exc).lower()
        kind = (
            "gpu_oom"
            if "cuda out of memory" in msg or "cuda error: out of memory" in msg
            else "ram_oom"
            if isinstance(exc, MemoryError)
            or (isinstance(exc, OSError) and exc.errno == errno.ENOMEM)
            else "other"
        )

        Path(error).write_text(
            json.dumps({"kind": kind, "traceback": traceback.format_exc()})
        )
        sys.exit(1)


@dataclass
class Trial:
    fn: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)
    estimated_ram: int | None = None
    estimated_vram: int | None = 0

    attempts: int = field(default=0, init=False)
    result: Any = field(default=None, init=False)
    error: str | None = field(default=None, init=False)


class TuningLauncher:
    def __init__(
        self,
        trials: list[Trial],
        *,
        gpu_ids: list[int] | None = None,
        max_processes: int = 4,
        min_processes: int = 1,
        ram_offset: int = 2 << 30,
        vram_offset: int = 1 << 30,
        initial_ram: int = 2 << 30,
        initial_vram: int = 2 << 30,
        interval: float = 10,
        memory_tolerance: float = 0.0,
        retry_factor: float = 1.3,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        if not 1 <= min_processes <= max_processes or max_retries < 0:
            raise ValueError("Invalid process limits")
        if min(ram_offset, vram_offset, initial_ram, initial_vram) < 0:
            raise ValueError("Memory values must be nonnegative")
        if interval <= 0 or not 0 <= memory_tolerance < 1 or retry_factor <= 1:
            raise ValueError("Invalid scheduler parameters")

        for t in trials:
            if any(
                v is not None and v < 0 for v in (t.estimated_ram, t.estimated_vram)
            ):
                raise ValueError("Negative memory estimate")
            if t.fn.__module__ == "__main__" or "<locals>" in t.fn.__qualname__:
                raise ValueError("Trial functions must be top-level and importable")

        self.trials = trials
        self.pending = deque(range(len(trials)))
        self.running = {}  # index -> (process, gpu, output, error)
        self.completed = []
        self.failed = []

        self.max_processes = max_processes
        self.min_processes = min_processes
        self.ram_offset = ram_offset
        self.vram_offset = vram_offset
        self.initial_ram = initial_ram
        self.initial_vram = initial_vram
        self.interval = interval
        self.memory_tolerance = memory_tolerance
        self.retry_factor = retry_factor
        self.max_retries = max_retries
        self.verbose = verbose

        self.avg_ram = self.avg_vram = None
        self._used = False
        self.nvml, self.gpus = None, {}

        if gpu_ids != []:
            try:
                import pynvml

                pynvml.nvmlInit()
            except Exception as exc:
                if gpu_ids is not None:
                    raise RuntimeError("NVML unavailable") from exc
            else:
                self.nvml = pynvml
                ids = range(pynvml.nvmlDeviceGetCount()) if gpu_ids is None else gpu_ids
                for i in ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    if isinstance(uuid, bytes):
                        uuid = uuid.decode()
                    self.gpus[i] = handle, uuid

    def _log(self, msg):
        if self.verbose:
            print(f"[launcher] {msg}", flush=True)

    def _memory(self):
        ram = psutil.virtual_memory().available
        gpu = {
            i: self.nvml.nvmlDeviceGetMemoryInfo(handle).free
            for i, (handle, _) in self.gpus.items()
        }
        return ram, gpu

    def _update_averages(self, ram, gpu):
        if self.running:
            used = self.base_ram - ram
            if used > 0:
                self.avg_ram = used / len(self.running)
        else:
            self.base_ram = ram

        used = count = 0
        for i in self.gpus:
            n = sum(g == i for _, g, _, _ in self.running.values())
            if n:
                used += self.base_gpu[i] - gpu[i]
                count += n
            else:
                self.base_gpu[i] = gpu[i]

        if count and used > 0:
            self.avg_vram = used / count

    def _launch(self, idx, gpu, directory):
        t = self.trials[idx]
        t.attempts += 1
        t.result = t.error = None

        prefix = directory / f"{idx}_{t.attempts}"
        src = Path(f"{prefix}.pkl")
        out = Path(f"{prefix}.out")
        err = Path(f"{prefix}.err")

        with src.open("wb") as f:
            pickle.dump((t.fn, t.kwargs), f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpus[gpu][1] if gpu is not None else ""
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (os.getcwd(), *sys.path, env.get("PYTHONPATH")))
        )

        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(src),
                str(out),
                str(err),
            ],
            env=env,
            start_new_session=True,
            stdout=None if self.verbose else subprocess.DEVNULL,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )

        self.running[idx] = process, gpu, out, err
        self._log(f"start #{idx} attempt={t.attempts} GPU={gpu}")

    def _schedule(self, ram, gpu, directory):
        # Enforce both actual free memory and outstanding reservations.
        ram_left = (
            min(
                ram,
                self.base_ram - sum(self.trials[i].estimated_ram for i in self.running),
            )
            - self.ram_offset
        )

        gpu_left = {
            i: min(
                free,
                self.base_gpu[i]
                - sum(
                    self.trials[j].estimated_vram
                    for j, (_, g, _, _) in self.running.items()
                    if g == i
                ),
            )
            - self.vram_offset
            for i, free in gpu.items()
        }

        launched = 0

        for _ in range(len(self.pending)):
            if len(self.running) >= self.max_processes:
                break

            idx = self.pending.popleft()
            t = self.trials[idx]

            if t.estimated_ram is None:
                t.estimated_ram = math.ceil(
                    self.avg_ram if self.avg_ram is not None else self.initial_ram
                )

            if t.estimated_vram is None:
                t.estimated_vram = math.ceil(
                    self.avg_vram if self.avg_vram is not None else self.initial_vram
                )

            selected = max(gpu_left, key=gpu_left.get) if gpu_left else None
            available_vram = gpu_left[selected] if selected is not None else 0

            if t.estimated_ram > ram_left or t.estimated_vram > available_vram:
                self.pending.append(idx)
                continue

            self._launch(idx, selected, directory)

            ram_left -= t.estimated_ram
            if selected is not None:
                gpu_left[selected] -= t.estimated_vram

            launched += 1

        return launched

    def _collect(self):
        for idx, (process, _, output, error) in list(self.running.items()):
            if process.poll() is None:
                continue

            del self.running[idx]

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            t = self.trials[idx]

            if process.returncode == 0 and output.exists():
                try:
                    with output.open("rb") as f:
                        t.result = pickle.load(f)
                except Exception:
                    t.error = traceback.format_exc()
                else:
                    self.completed.append(t)
                    self._log(f"done #{idx} attempts={t.attempts}")
                    continue

            kind = "other"

            if error.exists():
                try:
                    info = json.loads(error.read_text())
                    kind, t.error = info["kind"], info["traceback"]
                except (ValueError, OSError, KeyError):
                    t.error = "Unreadable error report"
            elif t.error is None:
                t.error = f"Exit code: {process.returncode}"

            if kind == "ram_oom":
                t.estimated_ram = math.ceil(t.estimated_ram * self.retry_factor)
            elif kind == "gpu_oom":
                t.estimated_vram = math.ceil(t.estimated_vram * self.retry_factor)
            else:
                self.max_processes = max(self.min_processes, self.max_processes - 1)

            self._log(f"error #{idx} ({kind}), limit={self.max_processes}: {t.error}")

            if t.attempts <= self.max_retries:
                self.pending.append(idx)
            else:
                self.failed.append(t)

    def run(self):
        if self._used:
            raise RuntimeError("Launcher can run only once")
        self._used = True

        try:
            with TemporaryDirectory(prefix="lob_launcher_") as d:
                directory = Path(d)
                self.base_ram, self.base_gpu = self._memory()
                previous = None

                while self.pending or self.running:
                    self._collect()

                    ram, gpu = self._memory()
                    self._update_averages(ram, gpu)

                    stable = previous is None or (
                        ram >= previous[0] * (1 - self.memory_tolerance)
                        and all(
                            gpu[i] >= previous[1][i] * (1 - self.memory_tolerance)
                            for i in gpu
                        )
                    )

                    launched = self._schedule(ram, gpu, directory) if stable else 0

                    free = ", ".join(f"{i}:{v / 2**30:.1f}GiB" for i, v in gpu.items())
                    self._log(
                        f"running={len(self.running)}/{self.max_processes} "
                        f"queued={len(self.pending)} "
                        f"RAM={ram / 2**30:.1f}GiB "
                        f"GPU=[{free}] stable={stable}"
                    )

                    if not self.running and self.pending and stable and not launched:
                        raise RuntimeError("No pending trial fits the available memory")

                    previous = ram, gpu

                    if self.running or self.pending:
                        time.sleep(self.interval)

        finally:
            for process, _, _, _ in self.running.values():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

            if self.nvml is not None:
                self.nvml.nvmlShutdown()

        return self.completed, self.failed


if __name__ == "__main__":
    _worker(*sys.argv[1:])
