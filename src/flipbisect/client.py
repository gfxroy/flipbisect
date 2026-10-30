"""Minimal OpenAI-compatible client (stdlib only) that returns per-token logprobs."""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from dataclasses import dataclass, field, asdict


@dataclass
class Tok:
    token: str
    logprob: float | None = None
    top: list[tuple[str, float]] = field(default_factory=list)  # sorted desc


@dataclass
class Completion:
    text: str
    tokens: list[Tok]
    latency_s: float = 0.0
    finish_reason: str | None = None

    def to_dict(self):
        return {"text": self.text, "finish_reason": self.finish_reason, "latency_s": round(self.latency_s, 4),
                "tokens": [asdict(t) for t in self.tokens]}


def _parse_logprobs(lp) -> list[Tok]:
    if not lp:
        return []
    if isinstance(lp, dict) and "content" in lp:  # chat style (also llama.cpp /v1/completions)
        out = []
        for c in lp["content"] or []:
            top = [(t["token"], float(t["logprob"])) for t in c.get("top_logprobs") or []]
            top.sort(key=lambda x: -x[1])
            out.append(Tok(c["token"], c.get("logprob"), top))
        return out
    # legacy completions style: tokens / token_logprobs / top_logprobs (list of dicts)
    toks = lp.get("tokens") or []
    tl = lp.get("token_logprobs") or [None] * len(toks)
    tops = lp.get("top_logprobs") or [None] * len(toks)
    out = []
    for t, l, d in zip(toks, tl, tops):
        top = sorted(((k, float(v)) for k, v in (d or {}).items()), key=lambda x: -x[1])
        out.append(Tok(t, l, top))
    return out


class Client:
    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 mode: str = "completions", timeout: float = 300):
        self.base = base_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        self.model, self.api_key, self.mode, self.timeout = model, api_key, mode, timeout

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def complete(self, prompt: str, max_tokens: int = 64, top_logprobs: int = 5, extra: dict | None = None,
                 seed: int | None = 0) -> Completion:
        body: dict = {"max_tokens": max_tokens, "temperature": 0}
        if seed is not None:
            body["seed"] = seed
        if self.model:
            body["model"] = self.model
        if self.mode == "chat":
            body.update(messages=[{"role": "user", "content": prompt}], logprobs=True, top_logprobs=top_logprobs)
            path = "/chat/completions"
        else:
            body.update(prompt=prompt, logprobs=top_logprobs)
            path = "/completions"
        body.update(extra or {})
        t0 = time.perf_counter()
        d = self._post(path, body)
        dt = time.perf_counter() - t0
        ch = d["choices"][0]
        text = ch["message"]["content"] if self.mode == "chat" else ch.get("text", "")
        toks = _parse_logprobs(ch.get("logprobs"))
        return Completion(text or "", toks, dt, ch.get("finish_reason"))

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/models", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False
