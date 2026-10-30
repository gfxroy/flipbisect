"""Integration tests against a real llama.cpp server (CPU). Skipped if no binary/model."""
import os, pytest
from flipbisect.backends import LlamaCpp
from flipbisect.client import Client
from flipbisect.probe import probe, Condition
from flipbisect.bisect import Evaluator, bisect

BIN = os.environ.get("FLIPBISECT_SERVER_BIN", ".bin/llama-b12015/llama-server")
MODEL = os.environ.get("FLIPBISECT_MODEL", ".models/smol360.gguf")
pytestmark = pytest.mark.skipif(not (os.path.exists(BIN) and os.path.exists(MODEL)), reason="no llama-server/model")
P = ["Explain why the sky is blue in three sentences.", "List five uses of copper."]


def test_no_prompt_cache_is_deterministic_under_load():
    b = LlamaCpp(MODEL, BIN)
    with b.launch({"cache_prompt": False, "threads": 1}) as s:
        r = probe(Client(s.url), P, [Condition("solo"), Condition("burst", concurrency=4)], repeats=2, max_tokens=24,
                  log=lambda *_: None)
    assert r["summary"]["total"]["identical"] == r["summary"]["total"]["runs"]
    assert all(p["reference"]["tokens"][0]["top"] for p in r["prompts"])  # logprobs came back


def test_bisect_returns_minimal_reproducer():
    b = LlamaCpp(MODEL, BIN)
    ev = Evaluator(b, P, repeats=2, max_tokens=48, log=lambda *_: None)
    res = bisect(ev, {"cache_prompt": False, "threads": 1}, {"cache_prompt": True, "threads": 8, "ubatch": 128})
    if "already diverges" in res.get("error", ""):
        pytest.skip("rare residual flip in the good config (~2% of runs on this hardware)")
    if "did not diverge" in res.get("error", ""):
        pytest.skip("bad config happened to be stable on this hardware")
    assert "error" not in res, res.get("error")
    assert res["minimal"] and set(res["minimal"]) <= {"cache_prompt", "threads", "ubatch"}
    assert len(res["minimal"]) < 3
    assert res["minimal_result"]["flips"] > 0


def test_parity_bisect_blames_flash_attention():
    """Deterministic predicate: which knob changes the solo output vs the baseline build config."""
    b = LlamaCpp(MODEL, BIN)
    ev = Evaluator(b, P, repeats=1, max_tokens=48, log=lambda *_: None)
    res = bisect(ev, {"cache_prompt": False, "threads": 1},
                 {"cache_prompt": False, "threads": 1, "flash_attn": "on", "ubatch": 128, "batch": 512}, "parity")
    assert "error" not in res, res.get("error")
    assert res["minimal"] == {"flash_attn": "on"}
