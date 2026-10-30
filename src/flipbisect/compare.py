"""Find the first diverging token and classify it as a tie-flip or a real divergence."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .client import Completion

IDENTICAL, TIE, REAL, UNKNOWN = "identical", "near-tie", "divergence", "divergence-unscored"


@dataclass
class Diff:
    verdict: str
    position: int | None = None      # index of first differing token
    ref_token: str | None = None
    got_token: str | None = None
    margin: float | None = None      # ref top1 - top2 logprob at the split, in nats (from the reference run)
    got_rank_in_ref: int | None = None  # rank of the other token in the reference's top-k (0 = top1)
    got_margin: float | None = None  # in the other run: its pick minus the reference token's logprob
    swing: float | None = None       # margin + got_margin = how far the two candidates moved relative to each other
    ref_len: int = 0
    got_len: int = 0
    prefix_text: str = ""

    def to_dict(self):
        return asdict(self)


def first_divergence(ref: Completion, got: Completion, tie_eps: float = 0.1) -> Diff:
    """Compare token sequences; if tokens are unavailable fall back to text comparison.

    A split is a *near-tie* when the reference's top-1 and top-2 logprobs at the split are
    within ``tie_eps`` nats AND the token the other run picked is the reference's runner-up.
    That is the expected footprint of float non-associativity: tiny numeric drift that tips
    an argmax between two almost-equal candidates. Anything else is a real divergence.
    """
    a, b = ref.tokens, got.tokens
    if not a or not b:
        if ref.text == got.text:
            return Diff(IDENTICAL, ref_len=len(ref.text), got_len=len(got.text))
        i = next((k for k, (x, y) in enumerate(zip(ref.text, got.text)) if x != y), min(len(ref.text), len(got.text)))
        return Diff(UNKNOWN, position=i, ref_len=len(ref.text), got_len=len(got.text), prefix_text=ref.text[:i])
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k].token != b[k].token), None)
    if i is None:
        if len(a) == len(b):
            return Diff(IDENTICAL, ref_len=len(a), got_len=len(b))
        i = n  # one stopped earlier
    prefix = "".join(t.token for t in a[:i])
    rt = a[i].token if i < len(a) else "<eos>"
    gt = b[i].token if i < len(b) else "<eos>"
    d = Diff(REAL, i, rt, gt, ref_len=len(a), got_len=len(b), prefix_text=prefix[-120:])
    if i < len(a) and len(a[i].top) >= 2:
        top = a[i].top
        d.margin = round(top[0][1] - top[1][1], 6)
        names = [t for t, _ in top]
        d.got_rank_in_ref = names.index(gt) if gt in names else None
        if i < len(b) and b[i].top:
            gm = dict(b[i].top)
            if gt in gm and rt in gm:
                d.got_margin = round(gm[gt] - gm[rt], 6)
                d.swing = round(d.margin + d.got_margin, 6)
        if d.margin <= tie_eps and d.got_rank_in_ref == 1:
            d.verdict = TIE
    elif i < len(a):
        d.verdict = UNKNOWN
    return d
