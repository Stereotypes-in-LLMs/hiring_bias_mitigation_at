"""What each pipeline step costs in GPU time and wall-clock time, one JSON record per step.

The final analysis compares arms by what they cost -- the extra GPU-hours an arm takes, how
much faster one path is than another, how many parameters a method changes -- and each such
number has to come from a measurement taken with its hardware and workload on record. This
module takes the measurement; `scripts/cost_report.py` turns the records into `reports/cost/`.

Two ways in. Wrap a command, which is what the GPU steps of `scripts/cscs/*.sbatch` do through
the `cost` function in env.sh, and which works for any program, code outside this repo too:

    cost <stage> <label> <command...>
    python -m hiring_bias_mitigation.utils.cost run --stage eval --label audit -- <command...>

or, in a Python process that no wrapper started:

    from hiring_bias_mitigation.utils import cost
    with cost.track("patching", "layer sweep"):
        ...

Stages are `STAGES`. Code running inside a tracked step adds what only it knows to the step's
record, and does nothing when no step is tracked:

    cost.add(requests=n, output_tokens=k)   # counters, summed over calls and processes
    cost.note(params_trainable=p)           # facts, the last value wins
    with cost.phase("load"): ...            # seconds per named phase, summed

A record holds the stage, label, arm, command and config; start, end, wall seconds and status
(`ok`, `failed`, `terminated`, or `running` while the step is alive -- rewritten every minute,
so a step killed outright still leaves all but its last minute on disk); the GPUs the step's
processes actually ran on (NVML's process list, sampled), their model, energy (NVML's counter),
mean utilisation and peak memory; the Slurm job, node and partition; the commit and whether
src/, scripts/ or configs/ had uncommitted changes; and the notes above. GPU-hours are wall
time times the GPUs used. Records go to $HBM_COST_DIR, by default $HBM_OUTPUT_ROOT/cost/records.

Nothing here may cost a run: a failure of the bookkeeping is reported on stderr and the
command runs regardless. HBM_COST=0 switches recording off.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

SCHEMA = 1
STAGES = ("datagen", "train", "eval", "patching", "other")
#: Where a wrapper tells its command's processes to leave their notes: each process writes
#: `<prefix>.<pid>.json`, so concurrent processes (DDP ranks) never overwrite each other.
NOTES_ENV = "HBM_COST_NOTES"
#: Seconds between GPU samples; the record on disk is rewritten every `HEARTBEAT_SECONDS`.
SAMPLE_SECONDS = float(os.environ.get("HBM_COST_INTERVAL", "10"))
HEARTBEAT_SECONDS = 60.0
#: Signals that end a step. SIGINT is recorded but not forwarded: a terminal's Ctrl-C already
#: reaches the whole process group, the command included.
_ENDING = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
_FORWARDED = (signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2)


def enabled() -> bool:
    return os.environ.get("HBM_COST", "1").strip() not in {"0", "false", "no"}


def records_dir() -> Path:
    explicit = os.environ.get("HBM_COST_DIR", "").strip()
    if explicit:
        return Path(explicit)
    root = os.environ.get("HBM_OUTPUT_ROOT", "").strip()
    # parents[3] is the repo root, as in utils/config.py::output_root.
    return (Path(root) if root else Path(__file__).resolve().parents[3]) / "cost" / "records"


def _warn(message: str) -> None:
    print(f"cost: {message}", file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now().astimezone()


def _write_json(path: Path, data: dict) -> None:
    """Atomic, so a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):  # numpy and torch scalars
        with contextlib.suppress(Exception):
            return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# --------------------------------------------------------------------------- notes


def _empty_notes() -> dict:
    return {"counters": {}, "facts": {}, "phases": {}}


def merge_notes(parts: list[dict]) -> dict:
    """Counters and phases add up across processes; for a fact, the last part wins."""
    out = _empty_notes()
    for part in parts:
        for key, value in (part.get("counters") or {}).items():
            out["counters"][key] = out["counters"].get(key, 0) + value
        out["facts"].update(part.get("facts") or {})
        for key, value in (part.get("phases") or {}).items():
            out["phases"][key] = round(out["phases"].get(key, 0.0) + value, 3)
    return out


_lock = threading.RLock()
_local = _empty_notes()  # what this process has added
_active: _Step | None = None  # a `track()` running in this process


def tracked() -> bool:
    """Whether a step is being recorded: by a wrapper around this process, or by `track()`."""
    return _active is not None or bool(os.environ.get(NOTES_ENV))


def _flush() -> None:
    prefix = os.environ.get(NOTES_ENV)
    if _active is not None or not prefix:
        return  # an in-process step reads `_local` itself; untracked, nothing to write
    try:
        _write_json(Path(f"{prefix}.{os.getpid()}.json"), _local)
    except Exception as e:
        _warn(f"cannot write notes to {prefix}: {e!r}")


def add(**counters: float) -> None:
    """Adds to counters of the current step (requests, tokens, rows...)."""
    with _lock:
        for key, value in counters.items():
            if value is not None:
                value = _jsonable(value)
                _local["counters"][key] = _local["counters"].get(key, 0) + value
        _flush()


def note(**facts) -> None:
    """Records facts about the current step (parameter counts, run name...)."""
    with _lock:
        _local["facts"].update({k: _jsonable(v) for k, v in facts.items()})
        _flush()


@contextlib.contextmanager
def phase(name: str):
    """Times the enclosed block into the current step's `phases[name]`, in seconds."""
    start = time.monotonic()
    try:
        yield
    finally:
        with _lock:
            phases = _local["phases"]
            phases[name] = round(phases.get(name, 0.0) + time.monotonic() - start, 3)
            _flush()


# --------------------------------------------------------------------------- GPUs


class _GpuMonitor:
    """NVML's view of the GPUs a step may use (CUDA_VISIBLE_DEVICES, else all of them).

    A device counts as used once a process of the step's tree -- the command and everything it
    started: torchrun ranks, vLLM workers -- appears in that device's compute process list.
    Utilisation and memory are averaged and maxed over the samples in which it does. Without
    psutil the tree is unknown and every visible device counts; without NVML (or with
    HBM_COST_NVML=0) the record falls back to the number of visible devices.

    CUDA_VISIBLE_DEVICES indices are taken to be NVML indices. On a node of identical GPUs
    both orders are the PCI bus order; UUIDs are matched exactly.
    """

    def __init__(self, root_pid: int):
        self.root_pid = root_pid
        self.nv = None
        self.devices: list[dict] = []
        self.model: str | None = None
        self.samples = 0
        self.tree_known = True
        if os.environ.get("HBM_COST_NVML", "1").strip() == "0":
            return
        try:
            import pynvml as nv

            nv.nvmlInit()
        except Exception:
            return
        self.nv = nv
        self.driver: dict = {}
        with contextlib.suppress(Exception):
            version = nv.nvmlSystemGetDriverVersion()
            self.driver["driver"] = version.decode() if isinstance(version, bytes) else version
        with contextlib.suppress(Exception):
            cuda = nv.nvmlSystemGetCudaDriverVersion()  # e.g. 13000
            self.driver["cuda_driver"] = f"{cuda // 1000}.{cuda % 1000 // 10}"
        for handle in self._visible_handles():
            with contextlib.suppress(Exception):
                self.devices.append({
                    "handle": handle,
                    "index": nv.nvmlDeviceGetIndex(handle),
                    "energy0": self._energy(handle),
                    "used": False, "util_sum": 0.0, "util_n": 0, "mem_peak": 0,
                })
        if self.devices:
            with contextlib.suppress(Exception):
                name = nv.nvmlDeviceGetName(self.devices[0]["handle"])
                self.model = name.decode() if isinstance(name, bytes) else str(name)

    def _visible_handles(self) -> list:
        nv = self.nv
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            return [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        handles = []
        for item in (s.strip() for s in visible.split(",")):
            try:
                if item.startswith(("GPU-", "MIG-")):
                    handles.append(nv.nvmlDeviceGetHandleByUUID(item))
                elif int(item) >= 0:
                    handles.append(nv.nvmlDeviceGetHandleByIndex(int(item)))
                else:
                    break  # CUDA stops at the first invalid entry, e.g. -1
            except Exception:
                break
        return handles

    def _energy(self, handle) -> float | None:
        try:
            return float(self.nv.nvmlDeviceGetTotalEnergyConsumption(handle))  # mJ
        except Exception:
            return None

    def _tree(self) -> set[int] | None:
        """The step's process ids; None when psutil is missing and the tree is unknown."""
        try:
            import psutil

            root = psutil.Process(self.root_pid)
            return {root.pid, *(p.pid for p in root.children(recursive=True))}
        except ImportError:
            return None
        except Exception:
            return set()

    @staticmethod
    def _foreign(pid: int) -> bool:
        """A GPU process this host's process table does not know: a container's PID
        namespace, where the step's own processes cannot be recognised by pid."""
        try:
            import psutil

            return not psutil.pid_exists(pid)
        except ImportError:
            return False

    def sample(self) -> None:
        if not self.devices:
            return
        tree = self._tree()
        self.tree_known = tree is not None
        nv = self.nv
        for device in self.devices:
            handle = device["handle"]
            try:
                procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                if tree is not None and not any(p.pid in tree for p in procs):
                    if any(self._foreign(p.pid) for p in procs):
                        device["foreign"] = True
                    continue
                device["used"] = True
                device["util_sum"] += nv.nvmlDeviceGetUtilizationRates(handle).gpu
                device["util_n"] += 1
                device["mem_peak"] = max(device["mem_peak"],
                                         nv.nvmlDeviceGetMemoryInfo(handle).used)
            except Exception:
                continue
        self.samples += 1

    def summary(self) -> dict:
        if not self.devices:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            declared = os.environ.get("HBM_COST_GPUS")
            if visible is not None:
                count = len([s for s in visible.split(",") if s.strip() not in {"", "-1"}])
                return {"count": count, "basis": "visible"}
            if declared:
                return {"count": int(declared), "basis": "declared"}
            return {"count": None, "basis": "unknown"}
        used = [d for d in self.devices if d["used"]]
        basis = "nvml-processes" if self.tree_known else "visible"
        if not used and any(d.get("foreign") for d in self.devices):
            # The step's processes are invisible by pid: count what it could use.
            used, basis = list(self.devices), "visible"
        energy = 0.0
        for device in used:
            now = self._energy(device["handle"])
            if now is not None and device["energy0"] is not None:
                energy += (now - device["energy0"]) / 1000.0
        util = [d["util_sum"] / d["util_n"] for d in used if d["util_n"]]
        return {
            "count": len(used),
            "basis": basis,
            "model": self.model,
            **self.driver,
            "visible": [d["index"] for d in self.devices],
            "used": [d["index"] for d in used],
            "energy_j": round(energy, 1),
            "util_mean_pct": round(sum(util) / len(util), 1) if util else None,
            "mem_peak_gib": round(max((d["mem_peak"] for d in used), default=0) / 2**30, 2),
            "samples": self.samples,
        }


# --------------------------------------------------------------------------- the record


def _slurm() -> dict:
    keys = {
        "job_id": "SLURM_JOB_ID", "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION", "node": "SLURMD_NODENAME",
        "nodes": "SLURM_JOB_NUM_NODES", "gpus_on_node": "SLURM_GPUS_ON_NODE",
        "array_task": "SLURM_ARRAY_TASK_ID",
    }
    return {name: os.environ[var] for name, var in keys.items() if os.environ.get(var)}


def _code() -> dict:
    """The checkout the step ran from, read before the command can change anything."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=20)
        except Exception:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if commit is None:
        return {}
    status = git("status", "--porcelain", "--", "src", "scripts", "configs")
    return {
        "repo": git("rev-parse", "--show-toplevel"),
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _software() -> dict:
    """Versions of the stack, from package metadata: nothing is imported, so no CUDA context
    is created and the record costs no GPU memory."""
    from importlib import metadata

    packages = {}
    for name in ("torch", "vllm", "transformers", "trl", "peft", "accelerate"):
        with contextlib.suppress(Exception):
            packages[name] = metadata.version(name)
    return {"python": platform.python_version(), **packages}


def config_of(command: list[str]) -> str | None:
    """The first `--config` argument of a command line, if it has one."""
    for i, arg in enumerate(command):
        if arg == "--config" and i + 1 < len(command):
            return command[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


class _Step:
    """One step's record, kept current on disk while the step runs."""

    def __init__(self, stage: str, label: str, *, arm: str | None = None,
                 config: str | None = None, command: list[str] | None = None,
                 in_process: bool = False):
        if stage not in STAGES:
            _warn(f"stage {stage!r} is not one of {', '.join(STAGES)}; recorded as given")
        self.in_process = in_process
        self.started = _now()
        self.t0 = time.monotonic()
        stamp = self.started.strftime("%Y%m%dT%H%M%S")
        job = os.environ.get("SLURM_JOB_ID", "local")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{stage}_{label}")[:60]
        self.path = records_dir() / f"{stamp}_{name}_{job}_{os.getpid()}.json"
        self.notes_prefix = str(self.path.parent / ".notes" / self.path.stem)
        self.data = {
            "schema": SCHEMA,
            "stage": stage,
            "label": label,
            "arm": arm or os.environ.get("HBM_COST_ARM") or None,
            "config": config,
            "command": shlex.join(command)[:2000] if command else None,
            "cwd": os.getcwd(),
            "host": socket.gethostname(),
            "start": self.started.isoformat(timespec="seconds"),
            "end": None,
            "wall_s": None,
            "status": "running",
            "exit_code": None,
            "gpu_h": None,
            "gpu": {},
            "slurm": _slurm(),
            "code": _code(),
            "software": _software(),
            "notes": _empty_notes(),
        }
        self.monitor: _GpuMonitor | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def create(cls, *args, **kwargs) -> _Step | None:
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            _warn(f"not recording this step: {e!r}")
            return None

    # The sampler thread and the main thread both call _refresh; the lock keeps them apart.
    def start(self, root_pid: int) -> None:
        try:
            self.monitor = _GpuMonitor(root_pid)
            self._refresh()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True,
                                            name="cost-sampler")
            self._thread.start()
        except Exception as e:
            _warn(f"cannot start recording: {e!r}")

    def _sample_loop(self) -> None:
        last_write = time.monotonic()
        wait = min(2.0, SAMPLE_SECONDS)  # an early first sample catches short steps
        while not self._stop.wait(wait):
            wait = SAMPLE_SECONDS
            try:
                with _lock:
                    self.monitor.sample()
                if time.monotonic() - last_write >= HEARTBEAT_SECONDS:
                    self._refresh()
                    last_write = time.monotonic()
            except Exception as e:
                _warn(f"sampling failed: {e!r}")

    def _notes(self) -> dict:
        if self.in_process:
            return merge_notes([_local])
        folder = Path(self.notes_prefix).parent
        stem = Path(self.notes_prefix).name
        parts = [_read_json(p) for p in sorted(folder.glob(f"{stem}.*.json"))]
        return merge_notes([p for p in parts if p])

    def _refresh(self, status: str | None = None, exit_code: int | None = None) -> None:
        with _lock:
            wall = time.monotonic() - self.t0
            gpu = self.monitor.summary() if self.monitor else {}
            count = gpu.get("count")
            self.data.update(
                end=_now().isoformat(timespec="seconds"),
                wall_s=round(wall, 1),
                gpu=gpu,
                gpu_h=None if count is None else round(wall * count / 3600, 4),
                notes=self._notes(),
            )
            if status is not None:
                self.data.update(status=status, exit_code=exit_code)
            _write_json(self.path, self.data)

    def mark_ending(self) -> None:
        """A signal is ending the step: on disk at once, in case SIGKILL follows."""
        with contextlib.suppress(Exception):
            self._refresh(status="terminated")

    def finish(self, status: str, exit_code: int | None) -> None:
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=10)
            self._refresh(status=status, exit_code=exit_code)
            if not self.in_process:
                folder = Path(self.notes_prefix).parent
                for part in folder.glob(f"{Path(self.notes_prefix).name}.*.json"):
                    part.unlink(missing_ok=True)
            self._report()
        except Exception as e:
            _warn(f"cannot finish the record {self.path}: {e!r}")

    def _report(self) -> None:
        d, gpu = self.data, self.data["gpu"]
        where = (f"{gpu['count']} x {gpu.get('model') or 'GPU'}" if gpu.get("count") is not None
                 else "unknown GPUs")
        joules = gpu.get("energy_j") or 0
        energy = (f", {joules / 1e6:.2f} MJ" if joules >= 1e6
                  else f", {joules / 1e3:.1f} kJ" if joules else "")
        gpu_h = "?" if d["gpu_h"] is None else f"{d['gpu_h']:.3f}"
        _warn(f"{d['stage']}/{d['label']} {d['status']} in {d['wall_s']:.0f} s on {where} "
              f"= {gpu_h} GPU-h{energy} -> {self.path}")


# --------------------------------------------------------------------------- entry points


def run(stage: str, label: str, command: list[str], *, arm: str | None = None,
        config: str | None = None) -> int:
    """Runs `command`, records it as one step, and returns its exit status the way a shell
    reports it (128 + N when signal N ended it)."""
    if not enabled():
        os.execvp(command[0], command)
    step = _Step.create(stage, label, arm=arm, config=config or config_of(command),
                        command=command)
    env = dict(os.environ)
    if step is not None:
        env[NOTES_ENV] = step.notes_prefix
    try:
        proc = subprocess.Popen(command, env=env)
    except OSError as e:
        _warn(f"cannot start {command[0]!r}: {e}")
        return 127

    ending: list[int] = []

    def on_signal(signum, _frame):
        if signum in _FORWARDED:
            with contextlib.suppress(OSError):
                proc.send_signal(signum)
        if signum in _ENDING:
            ending.append(signum)

    handled = (*_ENDING, *_FORWARDED)
    previous = {s: signal.signal(s, on_signal) for s in dict.fromkeys(handled)}
    if step is not None:
        step.start(proc.pid)
    marked = False
    try:
        while True:
            try:
                rc = proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if ending and not marked and step is not None:
                    step.mark_ending()
                    marked = True
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    code = 128 - rc if rc < 0 else rc
    if step is not None:
        status = "ok" if rc == 0 else "terminated" if (ending or rc < 0) else "failed"
        step.finish(status, code)
    return code


@contextlib.contextmanager
def track(stage: str, label: str, *, arm: str | None = None, config: str | None = None):
    """Records the enclosed block as one step, for code that no wrapper started.

    Inside a step that is already tracked, it times the block as a phase of that step
    instead, so nothing is counted twice. Use it in a single process; a multi-process program
    (torchrun, several workers) is recorded whole by the wrapper.
    """
    global _active
    if not enabled() or tracked():
        with phase(label):
            yield None
        return
    with _lock:
        _local.update(_empty_notes())
    step = _Step.create(stage, label, arm=arm, config=config, command=sys.argv,
                        in_process=True)
    if step is not None:
        _active = step
        step.start(os.getpid())
    status, code = "failed", 1
    try:
        yield step
        status, code = "ok", 0
    except KeyboardInterrupt:
        status, code = "terminated", 130
        raise
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        status = "ok" if code == 0 else "failed"
        raise
    finally:
        if step is not None:
            step.finish(status, code)
        _active = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hiring_bias_mitigation.utils.cost",
        description="Run a command and record its GPU and wall-clock cost as one step.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    runner = sub.add_parser("run", help="run a command as one recorded step")
    runner.add_argument("--stage", required=True, help=" | ".join(STAGES))
    runner.add_argument("--label", required=True, help="what the step does: sft, audit, judge...")
    runner.add_argument("--arm", help="the arm the cost belongs to (default: $HBM_COST_ARM)")
    runner.add_argument("--config", help="default: the command's own --config argument")
    runner.add_argument("command", nargs=argparse.REMAINDER, help="-- <command...>")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command to run")
    return run(args.stage, args.label, command, arm=args.arm, config=args.config)


if __name__ == "__main__":
    sys.exit(main())
