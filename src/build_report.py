"""Build the static results page report/index.html from results/report_data.json (run seeds.py first).
Also writes report/fragment.html, the same page without the document wrapper, for hosts that add their own."""
import json
import sys

from config import RES_DIR, ROOT

HEAD = r"""<title>Manglish DAE Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column, a headline finding up top, then evidence blocks in the order a reviewer asks for them. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --fg: #13171d; --muted: #566070; --line: #dde1e8;
  --mbart: #2c58d6; --byt5: #0b8a74; --mt5: #bf7410; --base: #8791a1; --bad: #c0392b; --soft: #eef1f6;
  --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1218; --surface: #171b23; --fg: #e8ebf1; --muted: #9aa5b6; --line: #2a303b;
  --mbart: #7b9cff; --byt5: #3cc9ad; --mt5: #e3a23c; --base: #7f8a9c; --bad: #ef7a6d; --soft: #1f242e; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f1218; --surface: #171b23; --fg: #e8ebf1; --muted: #9aa5b6; --line: #2a303b;
  --mbart: #7b9cff; --byt5: #3cc9ad; --mt5: #e3a23c; --base: #7f8a9c; --bad: #ef7a6d; --soft: #1f242e; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 16px; line-height: 1.55; margin: 0; }
.wrap { max-width: 980px; margin-inline: auto; padding-inline: 20px; padding-block: 40px 72px; }
h1 { font-size: clamp(1.9rem, 4.5vw, 2.7rem); line-height: 1.12; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 14px; text-wrap: balance; }
h2 { font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance; }
h3 { font-size: 0.95rem; font-weight: 600; margin: 0 0 4px; }
p { margin: 0 0 12px; max-width: 68ch; }
.eyebrow { font-family: var(--mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
.lede { font-size: 1.15rem; color: var(--fg); max-width: 62ch; }
.muted { color: var(--muted); }
section { margin-top: 56px; }
.sub { color: var(--muted); max-width: 64ch; margin-bottom: 18px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 20px; }
.panel { overflow-x: auto; }
.scroll { overflow-x: auto; }
.panel svg { min-width: 640px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 10px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { font-size: 0.72rem; font-family: var(--mono); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
td.num { font-family: var(--mono); font-size: 0.9rem; }
td .sd { color: var(--muted); font-size: 0.8rem; }
tr.best td.num.win { font-weight: 500; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: baseline; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
.tabs button { font: inherit; font-size: 0.85rem; font-family: var(--mono); color: var(--fg); background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: 5px 12px; cursor: pointer; }
.tabs button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
button:focus-visible { outline: 2px solid var(--mbart); outline-offset: 2px; }
svg { display: block; width: 100%; height: auto; }
svg text { font-family: var(--mono); font-size: 11.5px; fill: var(--muted); }
svg .lab { font-family: var(--sans); font-size: 13px; fill: var(--fg); }
svg .val { fill: var(--fg); }
.grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.grid3 > * { min-width: 0; }
.fail { border-top: 3px solid var(--line); padding-top: 12px; }
.stat { font-family: var(--mono); font-size: 0.8rem; color: var(--muted); margin-top: 8px; }
code, .mono { font-family: var(--mono); font-size: 0.86rem; }
.ex td { text-align: left; white-space: normal; vertical-align: top; min-width: 150px; font-family: var(--mono); font-size: 0.82rem; }
.ex td.ok { color: var(--byt5); }
.ex td.no { color: var(--bad); }
ul.plain { padding-left: 1.15em; margin: 0 0 12px; max-width: 68ch; }
ul.plain li { margin-bottom: 6px; }
.note { border-left: 3px solid var(--mt5); padding: 4px 0 4px 14px; margin: 14px 0; max-width: 68ch; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.85rem; margin-bottom: 10px; }
.themebtn { float: right; font: inherit; font-family: var(--mono); font-size: 0.75rem; color: var(--muted); background: none; border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
footer { margin-top: 56px; color: var(--muted); font-size: 0.85rem; }
</style>
"""

BODY = r"""<div class="wrap">
  <button class="themebtn" id="themebtn" type="button" aria-label="Switch colour theme">theme</button>
  <div class="eyebrow">Denoising adaptation · mBART-50 / mT5 / ByT5 · LoRA</div>
  <h1 id="headline">Which backbone cleans up noisy Manglish best?</h1>
  <p class="lede" id="lede"></p>

  <section id="results">
    <h2>Ranked results</h2>
    <p class="sub" id="results-sub"></p>
    <div class="panel scroll"><table id="rtable"></table></div>
  </section>

  <section>
    <h2>Error by metric</h2>
    <p class="sub">Bars are the mean over training seeds, dots are individual seeds, and the dashed line is the baseline that copies the noisy input unchanged.</p>
    <div class="panel">
      <div class="tabs" id="mtabs" role="group" aria-label="Metric"></div>
      <div class="legend" id="legend"></div>
      <div id="barchart"></div>
    </div>
  </section>

  <section>
    <h2>Are the top two actually different?</h2>
    <p class="sub" id="pair-sub"></p>
    <div class="panel"><div id="forest"></div></div>
  </section>

  <section>
    <h2>Training loss</h2>
    <p class="sub">Per-token loss at each logged step, first seed. The mT5 run at the specified learning rate is still falling when training stops, which is why it fails.</p>
    <div class="panel"><div id="loss"></div></div>
  </section>

  <section>
    <h2>How each model fails</h2>
    <div class="grid3" id="fails"></div>
  </section>

  <section>
    <h2>Same sentences, every model</h2>
    <p class="sub">The first eight validation sentences. Green means the output equals the clean target, red means it does not.</p>
    <div class="panel scroll"><table class="ex" id="examples"></table></div>
  </section>

  <section>
    <h2>Recommendation</h2>
    <div id="reco"></div>
  </section>

  <section>
    <h2>Limits and deviations</h2>
    <div class="grid3">
      <div><h3>Limits</h3><ul class="plain" id="limits"></ul></div>
      <div><h3>Deviations from the proposal</h3><ul class="plain" id="devs"></ul></div>
    </div>
  </section>

  <footer>Metrics are computed on the held-out validation split (seed 42 split, fixed validation noise). Lower CER and WER are better; higher BLEU and chrF are better. This study makes no translation-quality claim.</footer>
</div>
<script>
const D = __DATA__;
const COLORS = {mbart_large:'var(--mbart)', byt5_base:'var(--byt5)', 'mt5_base_lr1e-3':'var(--mt5)', mt5_base:'var(--mt5)', identity:'var(--base)'};
const LOWER = {CER:true, WER:true, BLEU:false, chrF:false};
const fmt = (x, d=2) => x.toFixed(d);
const el = (id) => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
function S(tag, attrs, text) { const n = document.createElementNS(ns, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text !== undefined) n.textContent = text; return n; }
const sys = Object.fromEntries(D.systems.map(s => [s.key, s]));
const ranked = [...D.systems].sort((a, b) => a.agg.CER.mean - b.agg.CER.mean);
const lbl = k => (sys[k] ? sys[k].label : k);
const cell = (s, m) => { const a = s.agg[m]; return fmt(a.mean) + (a.sd !== null ? ` <span class="sd">± ${fmt(a.sd)}</span>` : ''); };

function renderTable() {
  const t = el('rtable');
  const bestM = {}; for (const m of Object.keys(LOWER)) bestM[m] = ranked.reduce((p, c) => ((LOWER[m] ? c.agg[m].mean < p.agg[m].mean : c.agg[m].mean > p.agg[m].mean) ? c : p)).key;
  let h = '<thead><tr><th>System</th><th>Seeds</th><th>CER %</th><th>WER %</th><th>BLEU</th><th>chrF</th><th>Train min</th><th>Peak GPU GB</th></tr></thead><tbody>';
  const rows = ranked.map(s => ({s}));
  const b = D.baseline;
  let inserted = false;
  const baseRow = `<tr><td><span class="dot" style="background:var(--base)"></span>Copy the noisy input (baseline)</td><td class="num">-</td><td class="num">${fmt(b.CER)}</td><td class="num">${fmt(b.WER)}</td><td class="num">${fmt(b.BLEU)}</td><td class="num">${fmt(b.chrF)}</td><td class="num">-</td><td class="num">-</td></tr>`;
  for (const {s} of rows) {
    if (!inserted && s.agg.CER.mean > b.CER) { h += baseRow; inserted = true; }
    h += `<tr><td><span class="dot" style="background:${COLORS[s.key]}"></span>${s.label}</td><td class="num">${s.n_seeds}</td>` +
      ['CER','WER','BLEU','chrF'].map(m => `<td class="num ${bestM[m] === s.key ? 'win' : ''}">${bestM[m] === s.key ? '<strong>' + cell(s, m) + '</strong>' : cell(s, m)}</td>`).join('') +
      `<td class="num">${fmt(s.train_min, 1)}</td><td class="num">${fmt(s.peak_gpu_gb)}</td></tr>`;
  }
  if (!inserted) h += baseRow;
  t.innerHTML = h + '</tbody>';
}

let metric = 'CER';
function renderTabs() {
  const box = el('mtabs'); box.innerHTML = '';
  for (const m of Object.keys(LOWER)) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = m + (LOWER[m] ? ' (lower is better)' : ' (higher is better)');
    b.setAttribute('aria-pressed', String(m === metric));
    b.onclick = () => { metric = m; renderTabs(); renderBars(); };
    box.appendChild(b);
  }
}
function renderLegend() {
  el('legend').innerHTML = D.systems.map(s => `<span><span class="dot" style="background:${COLORS[s.key]}"></span>${s.label}</span>`).join('');
}
function renderBars() {
  const items = [...D.systems].sort((a, b) => LOWER[metric] ? a.agg[metric].mean - b.agg[metric].mean : b.agg[metric].mean - a.agg[metric].mean);
  const bv = D.baseline[metric];
  const maxV = Math.max(bv, ...items.flatMap(s => s.runs.map(r => r[metric]))) * 1.12;
  const W = 760, left = 230, right = 56, rowH = 44, top = 14, H = top + rowH * items.length + 26;
  const x = v => left + (v / maxV) * (W - left - right);
  const svg = S('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${metric} by system`});
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = maxV * i / ticks;
    svg.appendChild(S('line', {x1: x(v), x2: x(v), y1: top, y2: H - 22, stroke: 'var(--line)', 'stroke-width': 1}));
    svg.appendChild(S('text', {x: x(v), y: H - 6, 'text-anchor': 'middle'}, fmt(v, v >= 10 ? 0 : 1)));
  }
  items.forEach((s, i) => {
    const y = top + i * rowH;
    svg.appendChild(S('text', {x: left - 12, y: y + rowH / 2 + 4, 'text-anchor': 'end', class: 'lab'}, s.label.replace(/ \(.*\)/, '') + (s.label.includes('(') ? ' *' : '')));
    svg.appendChild(S('rect', {x: left, y: y + 9, width: Math.max(1, x(s.agg[metric].mean) - left), height: rowH - 20, rx: 3, fill: COLORS[s.key], opacity: 0.88}));
    s.runs.forEach(r => svg.appendChild(S('circle', {cx: x(r[metric]), cy: y + rowH / 2, r: 3.6, fill: 'var(--surface)', stroke: 'var(--fg)', 'stroke-width': 1.4})));
    svg.appendChild(S('text', {x: x(Math.max(s.agg[metric].mean, ...s.runs.map(r => r[metric]))) + 10, y: y + rowH / 2 + 4, class: 'val'}, fmt(s.agg[metric].mean)));
  });
  svg.appendChild(S('line', {x1: x(bv), x2: x(bv), y1: top - 4, y2: H - 22, stroke: 'var(--base)', 'stroke-width': 1.6, 'stroke-dasharray': '5 4'}));
  svg.appendChild(S('text', {x: x(bv) + 5, y: top + 6, fill: 'var(--base)'}, 'baseline ' + fmt(bv, 1)));
  const box = el('barchart'); box.innerHTML = ''; box.appendChild(svg);
  if (items.some(s => s.label.includes('('))) box.insertAdjacentHTML('beforeend', '<p class="muted mono" style="margin-top:8px;font-size:.78rem">* see the deviation notes for the training setting of this run.</p>');
}

function renderForest() {
  const P = D.pairs; if (!P.length) return;
  const W = 760, left = 270, right = 30, rowH = 46, top = 16, H = top + rowH * P.length + 30;
  const lim = Math.ceil(Math.max(...P.map(p => Math.max(Math.abs(p.lo), Math.abs(p.hi)))) + 0.5);
  const x = v => left + ((v + lim) / (2 * lim)) * (W - left - right);
  const svg = S('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Paired bootstrap CER differences'});
  for (let v = -lim; v <= lim; v++) {
    svg.appendChild(S('line', {x1: x(v), x2: x(v), y1: top, y2: H - 24, stroke: v === 0 ? 'var(--fg)' : 'var(--line)', 'stroke-width': v === 0 ? 1.5 : 1}));
    svg.appendChild(S('text', {x: x(v), y: H - 8, 'text-anchor': 'middle'}, (v > 0 ? '+' : '') + v));
  }
  P.forEach((p, i) => {
    const y = top + i * rowH + rowH / 2;
    const clear = p.hi < 0 || p.lo > 0;
    svg.appendChild(S('text', {x: left - 12, y: y + 4, 'text-anchor': 'end', class: 'lab'}, lbl(p.a).replace(/ \(.*\)/, '') + ' minus ' + lbl(p.b).replace(/ \(.*\)/, '')));
    svg.appendChild(S('line', {x1: x(p.lo), x2: x(p.hi), y1: y, y2: y, stroke: clear ? 'var(--fg)' : 'var(--muted)', 'stroke-width': 3, 'stroke-linecap': 'round'}));
    svg.appendChild(S('circle', {cx: x(p.mean), cy: y, r: 5.5, fill: clear ? 'var(--fg)' : 'var(--surface)', stroke: clear ? 'var(--fg)' : 'var(--muted)', 'stroke-width': 2}));
    svg.appendChild(S('text', {x: x(p.hi) + 10, y: y + 4, class: 'val'}, `${p.mean > 0 ? '+' : ''}${fmt(p.mean)}${clear ? '' : ' (overlaps 0)'}`));
  });
  svg.appendChild(S('text', {x: x(0) - 6, y: top - 3, 'text-anchor': 'end'}, 'first is better'));
  svg.appendChild(S('text', {x: x(0) + 6, y: top - 3}, 'second is better'));
  el('forest').appendChild(svg);
}

function renderLoss() {
  const W = 760, H = 300, l = 44, r = 20, t = 14, b = 32;
  const maxL = Math.ceil(Math.max(...D.systems.flatMap(s => s.loss_history)));
  const maxN = Math.max(...D.systems.map(s => s.loss_history.length));
  const x = i => l + (i / (maxN - 1)) * (W - l - r), y = v => t + (1 - v / maxL) * (H - t - b);
  const svg = S('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Training loss curves'});
  for (let v = 0; v <= maxL; v += Math.ceil(maxL / 4)) {
    svg.appendChild(S('line', {x1: l, x2: W - r, y1: y(v), y2: y(v), stroke: 'var(--line)'}));
    svg.appendChild(S('text', {x: l - 8, y: y(v) + 4, 'text-anchor': 'end'}, String(v)));
  }
  svg.appendChild(S('text', {x: W - r, y: H - 8, 'text-anchor': 'end'}, 'logged steps (every 10 optimiser steps)'));
  D.systems.forEach(s => {
    const pts = s.loss_history.map((v, i) => `${x(i)},${y(v)}`).join(' ');
    svg.appendChild(S('polyline', {points: pts, fill: 'none', stroke: COLORS[s.key], 'stroke-width': 2.2, 'stroke-dasharray': s.key === 'mt5_base' ? '6 4' : '0', 'stroke-linejoin': 'round'}));
    const last = s.loss_history.length - 1;
    svg.appendChild(S('circle', {cx: x(last), cy: y(s.loss_history[last]), r: 3.4, fill: COLORS[s.key]}));
    svg.appendChild(S('text', {x: x(last) - 8, y: y(s.loss_history[last]) - 8, 'text-anchor': 'end', class: 'val'}, fmt(s.loss_history[last])));
  });
  el('loss').innerHTML = '<div class="legend">' + D.systems.map(s => `<span><span class="dot" style="background:${COLORS[s.key]}"></span>${s.label}</span>`).join('') + '</div>';
  el('loss').appendChild(svg);
}

function renderExamples() {
  const keys = D.systems.map(s => s.key);
  let h = '<thead><tr><th>Noisy input</th><th>Clean target</th>' + keys.map(k => `<th>${lbl(k).replace(/ \(.*\)/, '')}${lbl(k).includes('(') ? ' *' : ''}</th>`).join('') + '</tr></thead><tbody>';
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  D.examples.forEach(e => {
    h += `<tr><td>${esc(e.noisy)}</td><td>${esc(e.clean)}</td>` + keys.map(k => `<td class="${e[k] === e.clean ? 'ok' : 'no'}">${esc(e[k])}</td>`).join('') + '</tr>';
  });
  el('examples').innerHTML = h + '</tbody>';
}

function renderFails() {
  const P = k => (sys[k] ? sys[k].patterns : {});
  const cards = [
    ['byt5_base', 'ByT5 drops text', 'Faithful on the words it keeps, but in its worst cases it cuts the sentence down to the last words: "wine glass kayyil pidichirikkunna old man" came back as "old man".'],
    ['mbart_large', 'mBART swaps in real words', 'When a word is hard to read it substitutes a fluent one: "birth day celebrate" became "white color", and "car reverse" became "race".'],
    ['mt5_base_lr1e-3', 'mT5 invents English', 'It replaces garbled English words with unrelated ones ("helmet" became "elephant") and leaves more repeated letters in place.'],
  ];
  el('fails').innerHTML = cards.map(([k, t, d]) => {
    const p = P(k);
    const st = (p.over_correction_rate_pct !== undefined ? `over-correction ${fmt(p.over_correction_rate_pct, 1)}%` : '') + (p.unresolved_elongation_pct !== undefined ? ` · unresolved repeats ${fmt(p.unresolved_elongation_pct, 1)}%` : '');
    return `<div class="fail" style="border-top-color:${COLORS[k]}"><h3>${t}</h3><p>${d}</p><div class="stat">${st}</div></div>`;
  }).join('');
}

function fillCopy() {
  const C = D.copy;
  el('headline').textContent = C.headline;
  el('lede').innerHTML = C.lede;
  el('results-sub').innerHTML = C.resultsSub;
  el('pair-sub').innerHTML = C.pairSub;
  el('reco').innerHTML = C.reco;
  el('limits').innerHTML = C.limits.map(x => `<li>${x}</li>`).join('');
  el('devs').innerHTML = C.devs.map(x => `<li>${x}</li>`).join('');
}

fillCopy(); renderTable(); renderTabs(); renderLegend(); renderBars(); renderForest(); renderLoss(); renderExamples(); renderFails();
el('themebtn').onclick = () => {
  const r = document.documentElement, cur = r.getAttribute('data-theme');
  const dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  r.setAttribute('data-theme', dark ? 'light' : 'dark');
};
</script>
"""


def main():
    data = json.loads((RES_DIR / "report_data.json").read_text(encoding="utf-8"))
    copy_path = ROOT / "report" / "copy.json"
    data["copy"] = json.loads(copy_path.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    body = BODY.replace("__DATA__", payload)
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "fragment.html").write_text(HEAD + body, encoding="utf-8")
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + HEAD + '</head>\n<body>\n' + body + '</body>\n</html>\n')
    (out / "index.html").write_text(full, encoding="utf-8")
    print("wrote", out / "index.html", f"({len(full) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
