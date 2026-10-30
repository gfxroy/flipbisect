"""Black-box probe: replay identical temperature-0 requests under varied serving conditions."""
from __future__ import annotations
import random, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from .client import Client, Completion
from .compare import first_divergence, IDENTICAL, TIE

NEIGHBOR_TEXT = ("Write a detailed history of the printing press, covering Gutenberg, the spread of movable type "
                 "across Europe, and its effects on literacy, religion and science. ")


def neighbor_prompt(i: int, rng: random.Random) -> str:
    """Padding neighbours of deliberately different lengths so the batch shape changes."""
    return f"[{i}] " + NEIGHBOR_TEXT * rng.randint(1, 6) + "\nSummary:"


@dataclass
class Condition:
    name: str
    concurrency: int = 1      # target + (concurrency-1) neighbours in flight together
    evict_first: bool = False  # send an unrelated long prompt first (cold prefix cache)
    shuffle: bool = False     # randomise submission order / jitter inside the burst
    extra: dict = field(default_factory=dict)  # vendor body fields, e.g. {"cache_prompt": False}


DEFAULT_CONDITIONS = [
    Condition("solo-warm"),
    Condition("solo-cold", evict_first=True),
    Condition("concurrent-2", concurrency=2),
    Condition("concurrent-4", concurrency=4),
    Condition("concurrent-8-shuffled", concurrency=8, shuffle=True),
]


def run_condition(cl: Client, prompt: str, cond: Condition, max_tokens: int, top_k: int,
                  rng: random.Random) -> Completion:
    if cond.evict_first:
        cl.complete(neighbor_prompt(999, rng), max_tokens=4, top_logprobs=0)
    if cond.concurrency <= 1:
        return cl.complete(prompt, max_tokens, top_k, cond.extra)
    jobs = [("T", prompt)] + [("N", neighbor_prompt(i, rng)) for i in range(cond.concurrency - 1)]
    if cond.shuffle:
        rng.shuffle(jobs)

    def go(job):
        kind, p = job
        if cond.shuffle:
            time.sleep(rng.random() * 0.05)
        return kind, cl.complete(p, max_tokens if kind == "T" else max_tokens * 2, top_k if kind == "T" else 0,
                                 cond.extra)

    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(go, jobs))
    return next(c for k, c in res if k == "T")


def probe(cl: Client, prompts: list[str], conditions=None, repeats: int = 3, max_tokens: int = 64,
          top_k: int = 5, tie_eps: float = 0.1, seed: int = 0, log=print) -> dict:
    conditions = conditions or DEFAULT_CONDITIONS
    rng = random.Random(seed)
    out = {"prompts": [], "conditions": [c.name for c in conditions], "repeats": repeats,
           "max_tokens": max_tokens, "tie_eps": tie_eps}
    for pi, p in enumerate(prompts):
        ref = cl.complete(p, max_tokens, top_k)
        rows = []
        for c in conditions:
            for r in range(repeats):
                got = run_condition(cl, p, c, max_tokens, top_k, rng)
                d = first_divergence(ref, got, tie_eps)
                rows.append({"condition": c.name, "repeat": r, **d.to_dict(), "text": got.text})
                if d.verdict != IDENTICAL:
                    log(f"  prompt {pi} {c.name}#{r}: {d.verdict} at token {d.position} "
                        f"{d.ref_token!r}->{d.got_token!r} margin={d.margin}")
        out["prompts"].append({"prompt": p, "reference": ref.to_dict(), "runs": rows})
    out["summary"] = summarize(out)
    return out


def summarize(res: dict) -> dict:
    by: dict = {}
    for pr in res["prompts"]:
        for r in pr["runs"]:
            s = by.setdefault(r["condition"], {"runs": 0, "identical": 0, "near-tie": 0, "divergence": 0})
            s["runs"] += 1
            key = r["verdict"] if r["verdict"] in s else "divergence"
            s[key] += 1
    total = {k: sum(v[k] for v in by.values()) for k in ("runs", "identical", "near-tie", "divergence")}
    return {"by_condition": by, "total": total, "unstable": total["runs"] != total["identical"]}
