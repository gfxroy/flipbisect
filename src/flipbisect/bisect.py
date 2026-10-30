"""White-box mode: toggle server knobs and bisect which ones cause divergence.

Two questions, two predicates:

* ``stability``: does this config give different temperature-0 outputs for the *same*
  request depending on load (concurrency, neighbours, cache state)?
* ``parity``: does this config give a different output than the baseline config at all
  (e.g. threads=8 vs threads=4) even when every run is solo?

``bisect`` takes a good and a bad config and runs delta debugging (ddmin) over the knobs
that differ, returning a 1-minimal set of changes that still reproduces the divergence.
"""
from __future__ import annotations
import itertools, json, time
from .backends import Backend
from .client import Client
from .compare import first_divergence, IDENTICAL, TIE
from .probe import Condition, probe

STRESS = [Condition("solo-warm"), Condition("solo-cold", evict_first=True),
          Condition("concurrent-4", concurrency=4), Condition("concurrent-8-shuffled", concurrency=8, shuffle=True)]


def key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


class Evaluator:
    def __init__(self, backend: Backend, prompts: list[str], repeats=2, max_tokens=48, top_k=5,
                 tie_eps=0.1, count_ties=True, log=print):
        self.b, self.prompts, self.repeats, self.max_tokens, self.top_k = backend, prompts, repeats, max_tokens, top_k
        self.tie_eps, self.count_ties, self.log = tie_eps, count_ties, log
        self.cache: dict[str, dict] = {}
        self.baseline_refs: list | None = None

    def run(self, cfg: dict) -> dict:
        full = {**self.b.defaults(), **cfg}
        k = key(full)
        if k in self.cache:
            return self.cache[k]
        t0 = time.time()
        self.log(f"· launching {self.b.name} {short(full, self.b.defaults())}")
        with self.b.launch(full) as srv:
            res = probe(Client(srv.url), self.prompts, STRESS, self.repeats, self.max_tokens, self.top_k,
                        self.tie_eps, log=lambda *_: None)
        refs = [p["reference"] for p in res["prompts"]]
        t = res["summary"]["total"]
        flips = t["divergence"] + (t["near-tie"] if self.count_ties else 0)
        out = {"config": full, "summary": res["summary"], "flips": flips, "runs": t["runs"],
               "refs": refs, "probe": res, "seconds": round(time.time() - t0, 1)}
        if self.baseline_refs is not None:
            out["parity"] = parity(self.baseline_refs, refs, self.tie_eps)
        self.cache[k] = out
        self.log(f"  -> {flips}/{t['runs']} unstable runs"
                 + (f", {out['parity']['differs']}/{len(refs)} prompts differ from baseline" if "parity" in out else "")
                 + f" ({out['seconds']}s)")
        return out

    def set_baseline(self, cfg: dict):
        self.baseline_refs = None
        r = self.run(cfg)
        self.baseline_refs = r["refs"]
        for v in self.cache.values():
            v["parity"] = parity(self.baseline_refs, v["refs"], self.tie_eps)
        return r


def parity(base_refs, refs, tie_eps):
    from .client import Completion, Tok
    def comp(d):
        return Completion(d["text"], [Tok(t["token"], t["logprob"], [tuple(x) for x in t["top"]]) for t in d["tokens"]])
    diffs = [first_divergence(comp(a), comp(b), tie_eps).to_dict() for a, b in zip(base_refs, refs)]
    return {"differs": sum(d["verdict"] != IDENTICAL for d in diffs), "diffs": diffs}


def short(cfg, defaults):
    d = {k: v for k, v in cfg.items() if defaults.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "(defaults)"


def is_bad(r: dict, predicate: str) -> bool:
    if predicate == "stability":
        return r["flips"] > 0
    return r.get("parity", {}).get("differs", 0) > 0


def sweep(ev: Evaluator, base: dict | None = None, knobs: list[str] | None = None) -> list[dict]:
    """One-factor-at-a-time: change each knob away from the baseline, everything else fixed."""
    base = {**ev.b.defaults(), **(base or {})}
    ev.set_baseline(base)
    rows = [row(ev, base, base, None)]
    for k in knobs or ev.b.knobs:
        for v in ev.b.knobs[k]:
            if v == base[k]:
                continue
            cfg = {**base, k: v}
            try:
                rows.append(row(ev, cfg, base, k))
            except Exception as e:  # a knob value the build does not support
                ev.log(f"  !! {k}={v}: {str(e).splitlines()[0]}")
                rows.append({"knob": k, "value": v, "error": str(e).splitlines()[0]})
    return rows


def row(ev, cfg, base, k):
    r = ev.run(cfg)
    return {"knob": k, "value": None if k is None else cfg[k], "change": short(cfg, base),
            "flips": r["flips"], "runs": r["runs"], "parity_differs": r.get("parity", {}).get("differs"),
            "first_split": first_split(r), "seconds": r["seconds"]}


def first_split(r):
    for p in r["probe"]["prompts"]:
        for x in p["runs"]:
            if x["verdict"] != IDENTICAL:
                return {k: x[k] for k in ("condition", "verdict", "position", "ref_token", "got_token", "margin", "swing")}
    return None


def ddmin(changes: list[tuple[str, object]], bad) -> list[tuple[str, object]]:
    """Zeller's delta debugging: smallest subset of ``changes`` for which ``bad(subset)`` holds."""
    n = 2
    while len(changes) >= 2:
        size = max(1, len(changes) // n)
        chunks = [changes[i:i + size] for i in range(0, len(changes), size)]
        for c in chunks:
            if bad(c):
                changes, n = c, 2
                break
        else:
            for c in chunks:
                comp = [x for x in changes if x not in c]
                if comp and bad(comp):
                    changes, n = comp, max(n - 1, 2)
                    break
            else:
                if n >= len(changes):
                    break
                n = min(len(changes), n * 2)
    return changes


def bisect(ev: Evaluator, good: dict, bad_cfg: dict, predicate="stability") -> dict:
    good = {**ev.b.defaults(), **good}
    bad_cfg = {**ev.b.defaults(), **bad_cfg}
    ev.set_baseline(good)
    g, b = ev.run(good), ev.run(bad_cfg)
    res = {"predicate": predicate, "good": good, "bad": bad_cfg,
           "good_result": strip(g), "bad_result": strip(b), "steps": []}
    if is_bad(g, predicate):
        res["error"] = "the 'good' config already diverges; pick a stricter good config"
        return res
    if not is_bad(b, predicate):
        res["error"] = "the 'bad' config did not diverge in this many runs; raise --repeats or add prompts"
        return res
    delta = [(k, bad_cfg[k]) for k in bad_cfg if bad_cfg[k] != good[k]]

    def test(subset):
        cfg = {**good, **dict(subset)}
        r = ev.run(cfg)
        bad = is_bad(r, predicate)
        res["steps"].append({"apply": dict(subset), "bad": bad, "flips": r["flips"],
                             "parity_differs": r.get("parity", {}).get("differs")})
        return bad

    minimal = ddmin(delta, test) if len(delta) > 1 else delta
    res["minimal"] = dict(minimal)
    res["minimal_config"] = {**good, **dict(minimal)}
    res["minimal_result"] = strip(ev.run(res["minimal_config"]))
    return res


def strip(r):
    return {k: r[k] for k in ("config", "flips", "runs", "summary", "seconds")} | (
        {"parity_differs": r["parity"]["differs"]} if "parity" in r else {}) | {"first_split": first_split(r)}
