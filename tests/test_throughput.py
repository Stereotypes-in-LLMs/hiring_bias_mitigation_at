"""Part B timing switches (docs/THROUGHPUT_B.md): the order prompts are sent in and vLLM's
prefix cache change how fast an audit generates, never which response belongs to which
prompt, and never the decoding."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
THROUGHPUT_CONFIGS = sorted((REPO / "configs" / "throughput").glob("*.yaml"))


def _answer(prompt: str) -> str:
    """A parseable response that names the prompt it answers."""
    tag = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    return json.dumps({"decision": "hire", "feedback": tag})


class EchoBackend:
    def __init__(self):
        self.sent: list[str] = []

    def generate(self, batch):
        self.sent.extend(batch)
        return [_answer(p) for p in batch]

    def close(self):
        pass


@pytest.mark.parametrize("order", ["as_built", "prefix", "shuffle"])
def test_submit_order_keeps_each_response_with_its_prompt(monkeypatch, tmp_path, order):
    """A response written back to the wrong row would swap decisions between attribute
    variants, which is exactly the difference the audit measures."""
    from hiring_bias_mitigation.eval import runner as R

    monkeypatch.setattr(R, "resolve_output_path", lambda p: str(tmp_path / str(p)))
    backend = EchoBackend()
    monkeypatch.setattr(R.BK, "build_backend", lambda cfg, activation_editor=None: backend)
    cfg = {"model": "m", "lang": "en", "protected_groups": ["military_status", "gender"],
           "limit_pairs": 3, "seed": 42, "submit_order": order}

    parsed, meta = R.run_audit(cfg)

    frame = R.build_frame(cfg).assign(profile_rendered="with_attribute")
    prompts = R._build_prompts(frame, "baseline")
    assert parsed["raw_output"].tolist() == [_answer(p) for p in prompts]
    assert parsed[["pair_id", "protected_attr", "condition"]].equals(
        frame[["pair_id", "protected_attr", "condition"]])
    assert sorted(backend.sent) == sorted(prompts)          # every prompt, once
    if order == "as_built":
        assert backend.sent == prompts
    else:
        assert backend.sent != prompts                       # the switch did reorder
    if order == "prefix":
        assert backend.sent == sorted(prompts)
    assert meta["submit_order"] == order


def test_reordered_generation_resumes_onto_its_own_chunks(monkeypatch, tmp_path):
    """Reordering and the chunk cache together: a crash after one chunk, then a resume, still
    returns every response at its prompt's position; a different order never resumes onto
    chunks sent in another order."""
    from hiring_bias_mitigation.eval import runner as R

    monkeypatch.setattr(R, "resolve_output_path", lambda p: str(tmp_path / str(p)))
    prompts = [f"p{i}" for i in range(10)]
    order = R.submission_order(prompts, "shuffle", seed=3)

    class Dies(EchoBackend):
        def generate(self, batch):
            if self.sent:
                raise RuntimeError("power cut")
            return super().generate(batch)

    with pytest.raises(RuntimeError):
        R._generate_in_order(Dies(), prompts, "run", order, chunk=4)
    resumed = EchoBackend()
    out = R._generate_in_order(resumed, prompts, "run", order, chunk=4)
    assert out == [_answer(p) for p in prompts]
    assert resumed.sent == [prompts[i] for i in order[4:]]  # the first chunk came from disk

    other = EchoBackend()
    R._generate_in_order(other, prompts, "run", R.submission_order(prompts, "prefix"), chunk=4)
    assert len(other.sent) == len(prompts)                  # nothing reused across orders


def test_default_order_is_the_order_every_audit_used():
    """The switch must not move existing runs: unset or as_built sends the prompts as built,
    so their chunk caches (keyed by the prompts as sent) still resume."""
    from hiring_bias_mitigation.eval import runner as R

    prompts = ["b", "a", "c"]
    assert R.submission_order(prompts) == [0, 1, 2]
    assert R.submission_order(prompts, "as_built") == [0, 1, 2]
    assert R.submission_order(prompts, "prefix") == [1, 0, 2]
    shuffled = R.submission_order([str(i) for i in range(50)], "shuffle", seed=42)
    assert shuffled == R.submission_order([str(i) for i in range(50)], "shuffle", seed=42)
    assert sorted(shuffled) == list(range(50)) and shuffled != list(range(50))
    with pytest.raises(ValueError):
        R.submission_order(prompts, "fastest")


def test_timing_configs_differ_only_in_the_speed_switches():
    """The four timing conditions must be the same audit: same model, prompts and decoding as
    the GH200 baseline, differing only in the prefix cache, the order and the run name."""
    assert len(THROUGHPUT_CONFIGS) == 4
    speed_only = {"enable_prefix_caching", "submit_order", "run_suffix"}
    timing_only = {"limit_pairs", "raw_dir", "disable_log_stats"}

    def flat(path):
        cfg = yaml.safe_load(path.read_text())
        engine = cfg.pop("engine_kwargs", {}) or {}
        return {**cfg, **engine}

    configs = [flat(p) for p in THROUGHPUT_CONFIGS]
    for cfg in configs[1:]:
        differing = {k for k in cfg.keys() | configs[0].keys() if cfg.get(k) != configs[0].get(k)}
        assert differing <= speed_only, differing
    assert {(c["enable_prefix_caching"], c["submit_order"]) for c in configs} == {
        (True, "prefix"), (True, "shuffle"), (False, "prefix"), (False, "shuffle")}

    baseline = flat(REPO / "configs" / "audit" / "qwen3.5-9b_en_baseline_gh200.yaml")
    for key in baseline.keys() - speed_only - timing_only:
        assert configs[0].get(key) == baseline[key], key
    for cfg in configs:
        assert cfg["raw_dir"] == "outputs/throughput/raw"


def test_a_timing_run_writes_generations_only(monkeypatch, tmp_path):
    """Timing runs must leave eval/results/ and the audits' outputs/raw/ alone, and must not
    resume onto a chunk an earlier attempt left behind: that chunk's time would go unmeasured."""
    from hiring_bias_mitigation.eval import runner as R

    spec = importlib.util.spec_from_file_location("throughput_b",
                                                  REPO / "scripts" / "throughput_b.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    monkeypatch.setattr(R, "resolve_output_path", lambda p: str(tmp_path / str(p)))
    backend = EchoBackend()
    monkeypatch.setattr(R.BK, "build_backend", lambda cfg, activation_editor=None: backend)
    config = tmp_path / "t.yaml"
    cfg = yaml.safe_load(THROUGHPUT_CONFIGS[0].read_text())
    cfg.update(limit_pairs=2, protected_groups=["military_status"])
    config.write_text(yaml.safe_dump(cfg))

    run_name = f"Qwen3.5-9B--en--baseline--{cfg['run_suffix']}-r1"
    stale = tmp_path / "outputs" / "partial" / f"{run_name}--main.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_text("left by an earlier attempt")

    tb.run(str(config), "r1")

    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == [f"outputs/throughput/raw/{run_name}.meta.json",
                       f"outputs/throughput/raw/{run_name}.parquet", "t.yaml"]
    assert len(backend.sent) == len(R.build_frame(cfg))     # every prompt generated here


def test_prefix_cache_counters_reach_the_cost_record(tmp_path, monkeypatch):
    """The hit rate in reports/cost/throughput.csv comes from vLLM's own counters, read after
    each generate call. They are cumulative, so the last reading is the run's total; an engine
    with stat logging off (every audit's) has none, and nothing may fail because of it."""
    from types import SimpleNamespace

    from hiring_bias_mitigation.eval import backends as BK
    from hiring_bias_mitigation.utils import cost

    monkeypatch.setenv("HBM_COST_DIR", str(tmp_path))
    monkeypatch.setenv("HBM_COST_NVML", "0")
    monkeypatch.delenv(cost.NOTES_ENV, raising=False)

    class Engine:
        def __init__(self, queries, hits):
            self.readings = [(queries // 2, hits // 2), (queries, hits)]

        def get_metrics(self):
            q, h = self.readings.pop(0)
            return [SimpleNamespace(name="vllm:prefix_cache_queries", value=q),
                    SimpleNamespace(name="vllm:prefix_cache_hits", value=h),
                    SimpleNamespace(name="vllm:prompt_tokens", value=10 * q)]

    class StatsOff:
        def get_metrics(self):
            raise AssertionError("Stat logging disabled")

    BK._note_prefix_cache(Engine(1000, 300))  # untracked: nothing to note
    with cost.track("eval", "throughput-x"):
        engine = Engine(1000, 300)
        BK._note_prefix_cache(engine)
        BK._note_prefix_cache(engine)
        BK._note_prefix_cache(StatsOff())
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["notes"]["facts"] == {"prefix_cache_queries": 1000, "prefix_cache_hits": 300}

    spec = importlib.util.spec_from_file_location("cost_report",
                                                  REPO / "scripts" / "cost_report.py")
    CR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CR)
    record["notes"].update(counters={"requests": 100, "output_tokens": 3600},
                           phases={"generate": 50.0})
    step = CR.record_step({**record, "_file": "r.json"}, {}, CR.datetime.now())
    (row,) = CR.throughput_table([step])
    assert row["prefix_cache_hit_rate"] == 0.3
    assert row["requests_per_s"] == 2.0
    step.update(prefix_cache_queries=None, prefix_cache_hits=None)
    assert CR.throughput_table([step])[0]["prefix_cache_hit_rate"] is None
