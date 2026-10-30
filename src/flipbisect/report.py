"""Self-contained monochrome HTML report from a flipbisect JSON result."""
from __future__ import annotations
import html, json

CSS = """
:root{--bg:#000;--fg:#ededed;--mute:#8a8a8a;--line:#222;--card:#0a0a0a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:980px;margin:0 auto;padding:64px 24px 96px}
h1{font-size:40px;letter-spacing:-.03em;line-height:1.1;margin:0 0 12px;font-weight:600}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--mute);font-weight:500;margin:56px 0 16px}
p.lead{color:var(--mute);font-size:17px;margin:0 0 32px;max-width:640px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.kpi{background:var(--card);padding:20px}.kpi b{display:block;font-size:28px;font-weight:600;letter-spacing:-.02em}.kpi span{color:var(--mute);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.08em}
code,.mono{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:13px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;border:1px solid #333;font-size:12px}
.pill.bad{background:#fff;color:#000;border-color:#fff}.pill.tie{border-style:dashed}
.split{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:10px 0;font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:13px;white-space:pre-wrap;word-break:break-word}
.split .a{background:#fff;color:#000;padding:0 3px;border-radius:3px}.split .b{border:1px solid #fff;padding:0 3px;border-radius:3px}
.bar{height:6px;background:#1a1a1a;border-radius:3px;overflow:hidden;min-width:80px}.bar i{display:block;height:100%;background:#fff}
.verdict{border:1px solid var(--line);border-radius:14px;padding:24px;background:var(--card);font-size:18px}
footer{color:var(--mute);font-size:12px;margin-top:64px}
"""


def e(x):
    return html.escape(str(x))


def pill(v):
    cls = "bad" if v == "divergence" else "tie" if v == "near-tie" else ""
    return f'<span class="pill {cls}">{e(v)}</span>'


def split_block(prefix, a, b, margin, cond):
    m = "" if margin is None else f"  margin {margin:.4f} nats"
    return (f'<div class="split"><span style="color:#8a8a8a">{e(cond)}{m}</span>\n…{e(prefix[-90:])}'
            f'<span class="a">{e(a)}</span> → <span class="b">{e(b)}</span></div>')


def probe_html(res):
    s = res["summary"]
    t = s["total"]
    out = ['<div class="kpis">',
           *(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in
             [("runs", t["runs"]), ("identical", t["identical"]), ("near-tie flips", t["near-tie"]),
              ("real divergences", t["divergence"])]), "</div>",
           "<h2>By condition</h2><table><tr><th>condition</th><th>runs</th><th>identical</th><th>near-tie</th><th>divergence</th><th></th></tr>"]
    for c, v in s["by_condition"].items():
        bad = (v["runs"] - v["identical"]) / max(v["runs"], 1)
        out.append(f"<tr><td class=mono>{e(c)}</td><td>{v['runs']}</td><td>{v['identical']}</td><td>{v['near-tie']}</td>"
                   f"<td>{v['divergence']}</td><td><div class=bar><i style='width:{bad*100:.0f}%'></i></div></td></tr>")
    out.append("</table><h2>First diverging token</h2>")
    seen = set()
    for p in res["prompts"]:
        for r in p["runs"]:
            if r["verdict"] == "identical":
                continue
            k = (p["prompt"], r["position"], r["got_token"])
            if k in seen:
                continue
            seen.add(k)
            out.append(f"<p class=mono style='color:#8a8a8a;margin:24px 0 4px'>{e(p['prompt'][:100])} · token {r['position']} {pill(r['verdict'])}</p>")
            out.append(split_block(r["prefix_text"], r["ref_token"], r["got_token"], r["margin"], r["condition"]))
    if not seen:
        out.append("<p>No run diverged.</p>")
    return "".join(out)


def sweep_html(rows):
    out = ["<table><tr><th>change</th><th>unstable runs</th><th>differs from baseline</th><th>first split</th></tr>"]
    for r in rows:
        if "error" in r:
            out.append(f"<tr><td class=mono>{e(r['knob'])}={e(r['value'])}</td><td colspan=3>unsupported: {e(r['error'][:80])}</td></tr>")
            continue
        fs = r.get("first_split")
        fs_s = "—" if not fs else f"{pill(fs['verdict'])} <span class=mono>token {fs['position']} {e(repr(fs['ref_token']))}→{e(repr(fs['got_token']))}</span>"
        pd = r.get("parity_differs")
        out.append(f"<tr><td class=mono>{e(r['change'])}</td><td>{r['flips']}/{r['runs']}</td><td>{'—' if pd is None else pd}</td><td>{fs_s}</td></tr>")
    return "".join(out) + "</table>"


def bisect_html(res):
    if "error" in res:
        return f"<div class=verdict>{e(res['error'])}</div>"
    m = ", ".join(f"<code>{e(k)}={e(v)}</code>" for k, v in res["minimal"].items())
    out = [f"<div class=verdict>Minimal change that reproduces the divergence: {m}</div>",
           "<h2>Steps</h2><table><tr><th>applied to good config</th><th>result</th><th>unstable runs</th></tr>"]
    for s in res["steps"]:
        out.append(f"<tr><td class=mono>{e(', '.join(f'{k}={v}' for k, v in s['apply'].items()))}</td>"
                   f"<td>{pill('divergence' if s['bad'] else 'identical')}</td><td>{s['flips']}</td></tr>")
    out.append("</table>")
    fs = res["minimal_result"].get("first_split")
    if fs:
        out.append("<h2>First split under the minimal config</h2>" +
                   split_block("", fs["ref_token"], fs["got_token"], fs["margin"], f"{fs['condition']} · token {fs['position']}"))
    return "".join(out)


def render(doc: dict) -> str:
    kind = doc.get("kind", "probe")
    meta = doc.get("meta", {})
    title = {"probe": "Same request. Different answers.", "sweep": "Which setting breaks determinism.",
             "bisect": "The setting that flips your output."}[kind]
    lead = e(meta.get("lead", "Identical temperature-0 requests replayed under different serving conditions."))
    body = []
    if kind == "probe":
        body.append(probe_html(doc["result"]))
    elif kind == "sweep":
        body.append("<h2>One setting at a time</h2>" + sweep_html(doc["result"]))
    else:
        body.append(bisect_html(doc["result"]))
    m = " · ".join(f"{e(k)}: {e(v)}" for k, v in meta.items() if k != "lead")
    return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>flipbisect report</title><style>{CSS}</style><main><h1>{title}</h1><p class=lead>{lead}</p>"
            f"{''.join(body)}<footer>flipbisect · {m}</footer></main></html>")


def write(doc: dict, json_path: str | None, html_path: str | None):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=1, default=str)
    if html_path:
        with open(html_path, "w") as f:
            f.write(render(doc))
