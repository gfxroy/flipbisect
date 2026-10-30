from flipbisect.client import Completion, Tok, _parse_logprobs
from flipbisect.compare import first_divergence, IDENTICAL, TIE, REAL
from flipbisect.bisect import ddmin
from flipbisect.backends import LlamaCpp, VLLM, SGLang
from flipbisect import report


def c(*toks):
    return Completion("".join(t[0] for t in toks), [Tok(t[0], t[1], list(t[2])) for t in toks])


def test_identical():
    a = c(("a", -.1, [("a", -.1), ("b", -3)]))
    assert first_divergence(a, a).verdict == IDENTICAL


def test_near_tie_flip():
    ref = c(("x", -.1, [("x", -.1)]), ("A", -.70, [("A", -.70), ("B", -.72)]))
    got = c(("x", -.1, [("x", -.1)]), ("B", -.69, [("B", -.69), ("A", -.71)]))
    d = first_divergence(ref, got)
    assert (d.verdict, d.position, d.ref_token, d.got_token) == (TIE, 1, "A", "B")
    assert abs(d.margin - 0.02) < 1e-9 and abs(d.swing - 0.04) < 1e-9


def test_real_divergence_large_margin():
    ref = c(("A", -.01, [("A", -.01), ("B", -5)]))
    got = c(("B", -.01, [("B", -.01), ("A", -5)]))
    assert first_divergence(ref, got).verdict == REAL


def test_non_runner_up_is_real():
    ref = c(("A", -.5, [("A", -.5), ("B", -.51), ("C", -.52)]))
    got = c(("C", -.5, [("C", -.5)]))
    assert first_divergence(ref, got).verdict == REAL


def test_early_stop():
    ref = c(("A", -.1, [("A", -.1), ("B", -2)]), ("B", -.1, [("B", -.1), ("<eos>", -3)]))
    got = c(("A", -.1, [("A", -.1)]))
    d = first_divergence(ref, got)
    assert d.position == 1 and d.got_token == "<eos>"


def test_parse_both_formats():
    chat = {"content": [{"token": "a", "logprob": -1, "top_logprobs": [{"token": "b", "logprob": -2}, {"token": "a", "logprob": -1}]}]}
    t = _parse_logprobs(chat)[0]
    assert t.top[0] == ("a", -1)
    legacy = {"tokens": ["a"], "token_logprobs": [-1], "top_logprobs": [{"b": -2, "a": -1}]}
    assert _parse_logprobs(legacy)[0].top == [("a", -1), ("b", -2)]


def test_ddmin_finds_single_and_pair():
    ch = [(k, 1) for k in "abcdefgh"]
    assert ddmin(ch, lambda s: ("c", 1) in s) == [("c", 1)]
    got = ddmin(ch, lambda s: ("b", 1) in s and ("g", 1) in s)
    assert sorted(got) == [("b", 1), ("g", 1)]


def test_command_builders():
    l = LlamaCpp("m.gguf", binary="llama-server")
    cmd = l.command({**l.defaults(), "parallel": 2, "cache_prompt": False}, 9)
    assert "-np" in cmd and cmd[cmd.index("-np") + 1] == "2" and "--no-cache-prompt" in cmd
    assert cmd[cmd.index("-c") + 1] == str(2048 * 2)
    v = VLLM("Qwen/Qwen3-0.6B")
    assert "--no-enable-prefix-caching" in v.command({**v.defaults(), "enable_prefix_caching": False}, 1)
    s = SGLang("Qwen/Qwen3-0.6B")
    assert "--disable-radix-cache" in s.command({**s.defaults(), "disable_radix_cache": True}, 1)


def test_report_renders_and_escapes():
    doc = {"kind": "probe", "meta": {"m": "<x>"}, "result": {"summary": {"total": {"runs": 1, "identical": 0, "near-tie": 1, "divergence": 0},
           "by_condition": {"c": {"runs": 1, "identical": 0, "near-tie": 1, "divergence": 0}}},
           "prompts": [{"prompt": "<p>", "runs": [{"verdict": "near-tie", "position": 1, "ref_token": "<a>", "got_token": "b",
                                                    "margin": .01, "condition": "c", "prefix_text": "x"}]}]}}
    h = report.render(doc)
    assert "&lt;a&gt;" in h and "<a>" not in h and "near-tie" in h
