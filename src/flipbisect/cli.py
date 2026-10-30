"""flipbisect command line."""
from __future__ import annotations
import argparse, json, sys, time
from . import __version__
from .client import Client
from .probe import probe, DEFAULT_CONDITIONS
from .backends import BACKENDS
from .bisect import Evaluator, sweep, bisect
from . import report

DEFAULT_PROMPTS = [
    "Explain why the sky is blue in three sentences.",
    "Write a Python function that checks if a number is prime.",
    "List five uses of copper.",
    "Summarise the plot of Romeo and Juliet in two sentences.",
]


def kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = {"true": True, "false": False}.get(v.lower(), int(v) if v.lstrip("-").isdigit() else v)
    return out


def prompts_of(a):
    if a.prompts:
        with open(a.prompts) as f:
            return [l.strip() for l in f if l.strip()]
    return DEFAULT_PROMPTS


def main(argv=None):
    ap = argparse.ArgumentParser("flipbisect", description="git bisect for LLM nondeterminism")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--prompts", help="file with one prompt per line")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--max-tokens", type=int, default=64)
        p.add_argument("--top-k", type=int, default=5, help="top logprobs to request")
        p.add_argument("--tie-eps", type=float, default=0.1, help="nats; splits under this margin are near-ties")
        p.add_argument("--json", default="flipbisect.json")
        p.add_argument("--html", default="flipbisect.html")

    p = sub.add_parser("probe", help="black-box: replay identical requests against a running server")
    p.add_argument("--url", required=True)
    p.add_argument("--model")
    p.add_argument("--api-key-env", help="name of env var holding the API key")
    p.add_argument("--chat", action="store_true", help="use /v1/chat/completions")
    common(p)

    for name in ("sweep", "bisect"):
        q = sub.add_parser(name, help="white-box: launch the server with different settings" if name == "sweep"
                           else "white-box: find the minimal setting change that causes divergence")
        q.add_argument("--backend", choices=list(BACKENDS), default="llamacpp")
        q.add_argument("--model", required=True, help="GGUF path (llama.cpp) or HF id (vLLM/SGLang)")
        q.add_argument("--server-bin", help="path to llama-server / vllm / python")
        q.add_argument("--set", nargs="*", default=[], help="base overrides, knob=value")
        q.add_argument("--ignore-ties", action="store_true", help="do not count near-tie flips as divergence")
        if name == "sweep":
            q.add_argument("--knobs", nargs="*")
        else:
            q.add_argument("--good", nargs="*", default=[], help="knob=value for the known-good config")
            q.add_argument("--bad", nargs="*", default=[], help="knob=value for the diverging config")
            q.add_argument("--predicate", choices=["stability", "parity"], default="stability")
        common(q)
    r = sub.add_parser("report", help="re-render HTML from a JSON result")
    r.add_argument("json_in")
    r.add_argument("-o", "--out", default="flipbisect.html")
    a = ap.parse_args(argv)

    if a.cmd == "report":
        doc = json.load(open(a.json_in))
        report.write(doc, None, a.out)
        print(a.out)
        return 0
    prompts = prompts_of(a)
    t0 = time.time()
    if a.cmd == "probe":
        import os
        key = os.environ.get(a.api_key_env) if a.api_key_env else None
        cl = Client(a.url, a.model, key, "chat" if a.chat else "completions")
        res = probe(cl, prompts, DEFAULT_CONDITIONS, a.repeats, a.max_tokens, a.top_k, a.tie_eps)
        meta = {"server": a.url, "model": a.model or "default", "prompts": len(prompts), "repeats": a.repeats}
        doc = {"kind": "probe", "meta": meta, "result": res}
        t = res["summary"]["total"]
        print(f"{t['runs']} runs: {t['identical']} identical, {t['near-tie']} near-tie flips, {t['divergence']} real divergences")
    else:
        kw = {"binary": a.server_bin} if a.server_bin else {}
        b = BACKENDS[a.backend](a.model, **kw)
        ev = Evaluator(b, prompts, a.repeats, a.max_tokens, a.top_k, a.tie_eps, not a.ignore_ties)
        meta = {"backend": a.backend, "model": a.model.split("/")[-1], "prompts": len(prompts), "repeats": a.repeats}
        if a.cmd == "sweep":
            res = sweep(ev, kv(a.set), a.knobs)
            doc = {"kind": "sweep", "meta": meta, "result": res}
            for r in res:
                if "error" not in r:
                    print(f"{r['change']:<28} unstable {r['flips']}/{r['runs']}  differs-from-baseline {r['parity_differs']}")
        else:
            res = bisect(ev, {**kv(a.set), **kv(a.good)}, {**kv(a.set), **kv(a.bad)}, a.predicate)
            doc = {"kind": "bisect", "meta": meta, "result": res}
            print(res.get("error") or f"minimal: {res['minimal']}")
    doc["meta"]["seconds"] = round(time.time() - t0, 1)
    report.write(doc, a.json, a.html)
    print(f"wrote {a.json} {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
