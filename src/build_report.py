"""Build the static results page report/index.html from results/ in the repository (run seeds.py and collect_results.py first).
Also writes report/fragment.html, the same page without the document wrapper, for hosts that add their own.

The page text lives in report/copy.json; every number on the page comes from the files in results/.
"""
import argparse
import json
import sys
from pathlib import Path

from config import ROOT

HEAD = r"""<title>Manglish DAE Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a sticky menu, then a headline strip and the recommendation before any evidence; evidence sections follow
   in the order a reviewer asks for them, on a wide grid so charts and tables fit without sideways scrolling.
   Colour: cool slate neutrals; one hue per model (blue, teal, amber) reused everywhere, baselines in grey.
   Type: Instrument Sans for reading, JetBrains Mono for labels and figures. */
:root {
  --bg: #f4f6f9; --surface: #ffffff; --fg: #121821; --muted: #566274; --line: #dbe0e8; --soft: #eceff5;
  --mbart: #2b57d4; --byt5: #0a8671; --mt5: #bb7210; --base: #7b8696; --bad: #c0392b; --good: #0a8671;
  --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1218; --surface: #161b23; --fg: #e8ebf1; --muted: #9aa5b6; --line: #2a303b; --soft: #1e2430;
  --mbart: #7b9cff; --byt5: #3cc9ad; --mt5: #e3a23c; --base: #7f8a9c; --bad: #ef7a6d; --good: #3cc9ad; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1218; --surface: #161b23; --fg: #e8ebf1; --muted: #9aa5b6; --line: #2a303b; --soft: #1e2430;
  --mbart: #7b9cff; --byt5: #3cc9ad; --mt5: #e3a23c; --base: #7f8a9c; --bad: #ef7a6d; --good: #3cc9ad; color-scheme: dark; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 16px; line-height: 1.55; margin: 0; }
a { color: inherit; text-decoration-color: var(--base); text-underline-offset: 3px; }
a:hover { text-decoration-color: var(--fg); }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); border-bottom: 1px solid var(--line); }
.bar-in { max-width: 1360px; margin-inline: auto; padding-inline: clamp(16px, 3vw, 40px); display: flex; align-items: center; gap: 16px; min-height: 52px; }
.brand { font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; text-decoration: none; }
nav { display: flex; gap: 0; overflow-x: auto; flex: 1; min-width: 0; scrollbar-width: none; }
nav::-webkit-scrollbar { display: none; }
nav a { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); text-decoration: none; padding: 6px 7px; border-radius: 5px; white-space: nowrap; }
nav a:hover, nav a:focus-visible { color: var(--fg); background: var(--soft); }
.themebtn { font: inherit; font-family: var(--mono); font-size: 0.74rem; color: var(--muted); background: none; border: 1px solid var(--line); border-radius: 6px; padding: 5px 11px; cursor: pointer; white-space: nowrap; }
button:focus-visible, a:focus-visible { outline: 2px solid var(--mbart); outline-offset: 2px; }
.wrap { max-width: 1360px; margin-inline: auto; padding-inline: clamp(16px, 3vw, 40px); padding-block: 0 88px; }
.hero { padding-block: 56px 8px; }
h1 { font-size: clamp(2rem, 4.2vw, 3.3rem); line-height: 1.08; font-weight: 600; letter-spacing: -0.028em; margin: 0 0 18px; max-width: 24ch; text-wrap: balance; }
h2 { font-size: 1.45rem; font-weight: 600; letter-spacing: -0.015em; margin: 0 0 8px; text-wrap: balance; }
h3 { font-size: 0.95rem; font-weight: 600; margin: 0 0 6px; }
p { margin: 0 0 12px; max-width: 72ch; }
.eyebrow { font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
.lede { font-size: 1.2rem; line-height: 1.5; max-width: 74ch; }
.muted { color: var(--muted); }
section { margin-top: 72px; padding-top: 28px; border-top: 1px solid var(--line); scroll-margin-top: 56px; }
.hero + section { margin-top: 44px; }
.sub { color: var(--muted); max-width: 90ch; margin-bottom: 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px 28px; margin-top: 36px; }
.tile { border-top: 3px solid var(--tile, var(--fg)); padding-top: 10px; min-width: 0; }
.tile .k { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
.tile .n { font-size: clamp(2rem, 3.4vw, 2.8rem); font-weight: 600; letter-spacing: -0.03em; line-height: 1.1; font-variant-numeric: tabular-nums; margin: 4px 0 2px; }
.tile .c { color: var(--muted); font-size: 0.9rem; max-width: 34ch; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; min-width: 0; overflow-x: auto; }
.panel h3 { margin-bottom: 12px; }
.cols2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 600px), 1fr)); gap: 20px; align-items: start; }
.cols3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 20px 32px; }
.cols2 > *, .cols3 > * { min-width: 0; }
.prose { columns: 2 440px; column-gap: 48px; }
.prose p { break-inside: avoid; }
.stack > * + * { margin-top: 20px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { font-size: 0.7rem; font-family: var(--mono); font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); line-height: 1.3; }
tbody tr:last-child td { border-bottom: 0; }
td.num { font-family: var(--mono); font-size: 0.88rem; white-space: nowrap; }
td.tl, th.tl { text-align: left; }
td.wrap-ok { white-space: normal; }
td .sd, th .sd { color: var(--muted); font-size: 0.78em; font-weight: 400; }
.total td { border-top: 2px solid var(--fg); font-weight: 600; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: baseline; }
td.cell { position: relative; min-width: 130px; }
.mini { position: absolute; left: 0; top: 4px; bottom: 4px; opacity: 0.2; border-radius: 0 3px 3px 0; }
.v { position: relative; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
.tabs button { font: inherit; font-size: 0.82rem; font-family: var(--mono); color: var(--fg); background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: 5px 12px; cursor: pointer; }
.tabs button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
svg { display: block; width: 100%; height: auto; }
svg text { font-family: var(--mono); font-size: 11.5px; fill: var(--muted); }
svg .lab { font-family: var(--sans); font-size: 13px; fill: var(--fg); }
svg .val { fill: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 0.85rem; margin-bottom: 12px; }
.fail { border-top: 3px solid var(--line); padding-top: 12px; }
.stat { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); margin-top: 8px; }
.ex th, .ex td { text-align: left; white-space: normal; min-width: 100px; font-family: var(--mono); font-size: 0.8rem; }
.ex td.ok { color: var(--good); }
.ex td.no { color: var(--bad); }
ul.plain { padding-left: 1.15em; margin: 0 0 12px; max-width: 78ch; }
ul.plain li { margin-bottom: 7px; }
.note { color: var(--muted); font-size: 0.88rem; margin-top: 12px; max-width: 100ch; }
footer { margin-top: 72px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.85rem; max-width: 110ch; }
@media (max-width: 700px) {
  .bar-in { gap: 10px; }
  .panel svg { min-width: 560px; }
  h1 { max-width: none; }
}
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
</style>
"""

BODY = r"""<header class="bar"><div class="bar-in">
  <a class="brand" href="#top">Manglish DAE</a>
  <nav id="nav" aria-label="Sections"></nav>
  <button class="themebtn" id="themebtn" type="button" aria-label="Switch colour theme">theme</button>
</div></header>
<main class="wrap" id="top">
  <div class="hero">
    <div class="eyebrow">Denoising adaptation · mBART-50 / mT5 / ByT5 · LoRA</div>
    <h1 id="headline">Which backbone cleans up noisy Manglish best?</h1>
    <p class="lede" id="lede"></p>
    <div class="tiles" id="tiles"></div>
  </div>

  <section id="reco"><h2>What to use</h2><div class="prose" id="recotext"></div></section>

  <section id="datasets" hidden>
    <h2>Datasets</h2>
    <p class="sub" id="data-sub"></p>
    <div class="stack">
      <div class="panel"><table id="dstab"></table></div>
      <div class="panel"><h3>What the sanitiser removed from each extra source</h3><table id="sanitab"></table><p class="note" id="sani-note"></p></div>
    </div>
    <p class="note" id="notused"></p>
  </section>

  <section id="results">
    <h2>Ranked results</h2>
    <p class="sub" id="results-sub"></p>
    <div class="stack">
      <div class="panel"><table id="rtable"></table></div>
      <div class="panel">
        <div class="tabs" id="mtabs" role="group" aria-label="Metric"></div>
        <div class="legend" id="legend"></div>
        <div id="barchart"></div>
        <p class="note">Bars are the mean over training seeds and dots are individual seeds. The dashed line is the baseline that copies the noisy input unchanged.</p>
      </div>
    </div>
  </section>

  <section id="moredata" hidden>
    <h2>Effect of more data</h2>
    <p class="sub" id="more-sub"></p>
    <div class="panel"><table id="scaletab"></table></div>
  </section>

  <section id="bysource" hidden>
    <h2>Combined test set, by source</h2>
    <p class="sub" id="comb-sub"></p>
    <div class="panel"><table id="combtab"></table></div>
  </section>

  <section id="significance">
    <h2>Are the differences real?</h2>
    <p class="sub" id="pair-sub"></p>
    <div class="cols2">
      <div class="panel"><h3>Original validation set, 325 sentences</h3><div id="forest"></div></div>
      <div class="panel"><h3>Combined test set, 2,059 sentences from three datasets</h3><div id="forest2"></div></div>
    </div>
  </section>

  <section id="robustness" hidden>
    <h2>Noise the models were not trained on</h2>
    <p class="sub" id="rob-sub"></p>
    <div class="stack">
      <div class="panel"><div class="legend" id="roblegend"></div><div id="robchart"></div></div>
      <div class="panel"><table id="robtab"></table></div>
    </div>
  </section>

  <section id="decoding" hidden>
    <h2>Decoding: beam search and a length guard</h2>
    <p class="sub" id="dec-sub"></p>
    <div class="panel"><table id="dectab"></table></div>
  </section>

  <section id="training">
    <h2>Training loss</h2>
    <p class="sub">Per-token loss at each logged step, first seed of each system. The mT5 run at the specified learning rate is still falling when training stops on the small data, which is why it fails there.</p>
    <div class="cols2">
      <div class="panel"><h3>2,934 training sentences, 552 steps</h3><div id="loss"></div></div>
      <div class="panel" id="losspanel2"><h3>18,548 training sentences, 3,480 steps</h3><div id="loss2"></div></div>
    </div>
  </section>

  <section id="failures">
    <h2>How each model fails</h2>
    <div class="cols3" id="fails"></div>
  </section>

  <section id="examples-sec">
    <h2>Same sentences, every model</h2>
    <p class="sub">The first eight validation sentences. Green means the output equals the clean target, red means it does not.</p>
    <div class="panel"><table class="ex" id="examples"></table></div>
  </section>

  <section id="compute" hidden>
    <h2>Compute and time</h2>
    <p class="sub" id="compute-sub"></p>
    <div class="stack">
      <div class="panel"><table id="runtab"></table></div>
      <div class="panel"><h3>Evaluation and studies</h3><table id="steptab"></table></div>
    </div>
  </section>

  <section id="limits">
    <h2>Limits and deviations</h2>
    <div class="cols2">
      <div><h3>Limits</h3><ul class="plain" id="limitlist"></ul></div>
      <div><h3>Deviations from the proposal</h3><ul class="plain" id="devs"></ul></div>
    </div>
  </section>

  <footer>Two held-out sets are used: the original 325 Manglish validation sentences, and a 2,059-sentence combined test set with held-out sentences from three datasets (the 325 are part of it). Noise is synthetic and fixed per sentence. Lower CER and WER are better; higher BLEU and chrF are better. This study makes no translation-quality claim.</footer>
</main>
<script>
const D = __DATA__;
const X = D.extra || {};
const LOWER = {CER: true, WER: true, BLEU: false, chrF: false};
const fmt = (x, d = 2) => x.toFixed(d);
const el = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
function S(tag, attrs, text) { const n = document.createElementNS(ns, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text !== undefined) n.textContent = text; return n; }
const sys = Object.fromEntries(D.systems.map(s => [s.key, s]));
const ranked = [...D.systems].sort((a, b) => a.agg.CER.mean - b.agg.CER.mean);

const SHORT = {mbart_large: 'mBART-50', byt5_base: 'ByT5-base', 'mt5_base_lr1e-3': 'mT5, LR 1e-3', mt5_base: 'mT5, LR 2e-4',
  mbart_large_plusall: 'mBART-50 + data', byt5_base_plusall: 'ByT5 + data', mt5_base_plusall: 'mT5, LR 2e-4 + data', 'mt5_base_lr1e-3_plusall': 'mT5, LR 1e-3 + data'};
const CN = {mbart: 'mBART-50', byt5: 'ByT5-base', 'mt5_lr1e-3': 'mT5, LR 1e-3', mt5: 'mT5, LR 2e-4', mbart_plusall: 'mBART-50 + data',
  byt5_plusall: 'ByT5 + data', mt5_plusall: 'mT5, LR 2e-4 + data', 'mt5_lr1e-3_plusall': 'mT5, LR 1e-3 + data',
  copy_input: 'copy the input', rule_collapse_runs: 'rule-based cleaner'};
const NEURAL = ['mbart', 'byt5', 'mt5_lr1e-3', 'mt5', 'mbart_plusall', 'byt5_plusall', 'mt5_lr1e-3_plusall', 'mt5_plusall'];
const SRC_NAME = {ALL: 'All sources', sellai_manglish: 'SeLLaI Manglish', 'sellai_mixed-code': 'SeLLaI Mixed-Code', dakshina_ml: 'Dakshina'};
const short = k => SHORT[k] || k;
const cname = k => CN[k] || SHORT[k] || k;
function colorOf(k) {
  if (k === 'identity') return 'var(--base)';
  if (k.indexOf('mbart') === 0) return 'var(--mbart)';
  if (k.indexOf('byt5') === 0) return 'var(--byt5)';
  if (k.indexOf('mt5') === 0) return 'var(--mt5)';
  return 'var(--base)';
}
const dot = k => `<span class="dot" style="background:${colorOf(k)}"></span>`;
const pm = a => fmt(a.mean) + (a.sd !== null && a.sd !== undefined ? ` <span class="sd">± ${fmt(a.sd)}</span>` : '');
const cell = (s, m) => pm(s.agg[m]);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const cw = id => Math.max(520, Math.floor(el(id).getBoundingClientRect().width) || 760);

function renderTiles() {
  const t = [];
  const rows = (X.combined || []).filter(r => r.source === 'ALL' && NEURAL.indexOf(r.system) >= 0);
  if (rows.length) {
    const b = rows.reduce((p, c) => (c.CER.mean < p.CER.mean ? c : p));
    t.push(['Lowest error on the combined test', fmt(b.CER.mean, 1) + '%', `${cname(b.system)}, character error on ${b.n.toLocaleString()} held-out sentences`, colorOf(b.system)]);
  }
  if (X.data) {
    t.push(['Training sentences', X.data.train_total.toLocaleString(), 'from three datasets, cleaned and de-duplicated', 'var(--fg)']);
    t.push(['Held-out test sentences', X.data.test_total.toLocaleString(), 'a tenth of each source, never used for training', 'var(--fg)']);
  }
  if (X.runs) {
    const h = (X.runs.training_minutes + X.runs.step_minutes) / 60;
    t.push(['GPU time', fmt(h, 1) + ' h', `${X.runs.runs.length} training runs plus evaluation, on ${X.runs.hardware.split(',')[0].replace('NVIDIA GeForce ', '')}`, 'var(--fg)']);
  }
  el('tiles').innerHTML = t.map(([k, n, c, col]) => `<div class="tile" style="--tile:${col}"><div class="k">${k}</div><div class="n">${n}</div><div class="c">${c}</div></div>`).join('');
}

function renderNav() {
  const items = [['reco', 'What to use'], ['datasets', 'Datasets'], ['results', 'Results'], ['moredata', 'More data'], ['bysource', 'By source'],
    ['significance', 'Significance'], ['robustness', 'Robustness'], ['decoding', 'Decoding'], ['training', 'Training'], ['failures', 'Failures'],
    ['examples-sec', 'Examples'], ['compute', 'Compute'], ['limits', 'Limits']];
  el('nav').innerHTML = items.filter(([id]) => !el(id).hidden).map(([id, l]) => `<a href="#${id}">${l}</a>`).join('');
}

function renderDatasets() {
  const P = X.profile; if (!P) return;
  el('datasets').hidden = false;
  let h = '<thead><tr><th>Dataset</th><th class="tl">Provider and licence</th><th class="tl">What the text is</th><th>Rows in file</th><th>After cleaning</th><th>Train</th><th>Test</th><th>Average sentence</th></tr></thead><tbody>';
  let tot = {raw: 0, clean: 0, train: 0, test: 0};
  P.datasets.forEach(d => {
    tot.raw += d.raw_rows; tot.clean += d.clean_rows; tot.train += d.train; tot.test += d.test;
    h += `<tr><td class="wrap-ok"><strong>${d.name}</strong></td><td class="tl wrap-ok">${d.provider}<br><a href="${d.url}" target="_blank" rel="noopener">${d.licence}</a></td>` +
      `<td class="tl wrap-ok">${d.register}</td><td class="num">${d.raw_rows.toLocaleString()}</td><td class="num">${d.clean_rows.toLocaleString()}</td>` +
      `<td class="num">${d.train.toLocaleString()}</td><td class="num">${d.test.toLocaleString()}</td><td class="num">${d.avg_words} words<br>${d.avg_chars} chars</td></tr>`;
  });
  h += `<tr class="total"><td>Total</td><td></td><td></td><td class="num">${tot.raw.toLocaleString()}</td><td class="num">${tot.clean.toLocaleString()}</td><td class="num">${tot.train.toLocaleString()}</td><td class="num">${tot.test.toLocaleString()}</td><td></td></tr>`;
  el('dstab').innerHTML = h + '</tbody>';
  const st = X.data;
  if (st) {
    let z = '<thead><tr><th>Source</th><th>Lines in</th><th>Clean sentences kept</th><th class="tl">Largest reasons for dropping a line</th></tr></thead><tbody>';
    Object.entries(st.per_source).forEach(([name, p]) => {
      const why = Object.entries(p.rejected_by_reason).slice(0, 6).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(', ');
      z += `<tr><td>${SRC_NAME[name] || name}</td><td class="num">${p.input_lines.toLocaleString()}</td><td class="num">${p.kept_unique.toLocaleString()}</td><td class="tl wrap-ok">${why}</td></tr>`;
    });
    const a = st.base_manglish_audit_unchanged;
    z += `<tr><td>${SRC_NAME.sellai_manglish} (original, not re-filtered)</td><td class="num">${a.sentences.toLocaleString()}</td><td class="num">${(a.sentences - a.would_drop).toLocaleString()}</td><td class="tl wrap-ok">the rules would drop only ${a.would_drop}</td></tr>`;
    el('sanitab').innerHTML = z + '</tbody>';
  }
  el('notused').innerHTML = 'Not used: ' + P.not_used.map(n => `${n.name}${n.rows ? ' (' + n.rows.toLocaleString() + ' rows)' : ''}, ${n.why}`).join('; ') + '.';
}

function renderTable() {
  const bestM = {};
  for (const m of Object.keys(LOWER)) bestM[m] = ranked.reduce((p, c) => ((LOWER[m] ? c.agg[m].mean < p.agg[m].mean : c.agg[m].mean > p.agg[m].mean) ? c : p)).key;
  let h = '<thead><tr><th>System</th><th>Seeds</th><th>CER %</th><th>WER %</th><th>BLEU</th><th>chrF</th><th>Training sentences</th><th>Train min</th><th>Peak GPU GB</th></tr></thead><tbody>';
  const b = D.baseline;
  const baseRow = `<tr><td>${dot('identity')}Copy the noisy input (baseline)</td><td class="num">-</td><td class="num">${fmt(b.CER)}</td><td class="num">${fmt(b.WER)}</td><td class="num">${fmt(b.BLEU)}</td><td class="num">${fmt(b.chrF)}</td><td class="num">-</td><td class="num">-</td><td class="num">-</td></tr>`;
  let inserted = false;
  ranked.forEach(s => {
    if (!inserted && s.agg.CER.mean > b.CER) { h += baseRow; inserted = true; }
    h += `<tr><td>${dot(s.key)}${s.label}</td><td class="num">${s.n_seeds}</td>` +
      ['CER', 'WER', 'BLEU', 'chrF'].map(m => `<td class="num">${bestM[m] === s.key ? '<strong>' + cell(s, m) + '</strong>' : cell(s, m)}</td>`).join('') +
      `<td class="num">${(s.train_rows || 2934).toLocaleString()}</td><td class="num">${fmt(s.train_min, 1)}</td><td class="num">${fmt(s.peak_gpu_gb)}</td></tr>`;
  });
  if (!inserted) h += baseRow;
  el('rtable').innerHTML = h + '</tbody>';
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
  el('legend').innerHTML = D.systems.map(s => `<span>${dot(s.key)}${short(s.key)}</span>`).join('');
}
function renderBars() {
  const items = [...D.systems].sort((a, b) => LOWER[metric] ? a.agg[metric].mean - b.agg[metric].mean : b.agg[metric].mean - a.agg[metric].mean);
  const bv = D.baseline[metric];
  const maxV = Math.max(bv, ...items.flatMap(s => s.runs.map(r => r[metric]))) * 1.12;
  const W = cw('barchart'), left = 190, right = 60, rowH = 38, top = 16, H = top + rowH * items.length + 26;
  const x = v => left + (v / maxV) * (W - left - right);
  const svg = S('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${metric} by system`});
  for (let i = 0; i <= 5; i++) {
    const v = maxV * i / 5;
    svg.appendChild(S('line', {x1: x(v), x2: x(v), y1: top, y2: H - 22, stroke: 'var(--line)', 'stroke-width': 1}));
    svg.appendChild(S('text', {x: x(v), y: H - 6, 'text-anchor': 'middle'}, fmt(v, v >= 10 ? 0 : 1)));
  }
  items.forEach((s, i) => {
    const y = top + i * rowH;
    svg.appendChild(S('text', {x: left - 12, y: y + rowH / 2 + 4, 'text-anchor': 'end', class: 'lab'}, short(s.key)));
    svg.appendChild(S('rect', {x: left, y: y + 8, width: Math.max(1, x(s.agg[metric].mean) - left), height: rowH - 16, rx: 3, fill: colorOf(s.key), opacity: s.key.indexOf('plusall') > 0 ? 1 : 0.62}));
    s.runs.forEach(r => svg.appendChild(S('circle', {cx: x(r[metric]), cy: y + rowH / 2, r: 3.4, fill: 'var(--surface)', stroke: 'var(--fg)', 'stroke-width': 1.4})));
    svg.appendChild(S('text', {x: x(Math.max(s.agg[metric].mean, ...s.runs.map(r => r[metric]))) + 10, y: y + rowH / 2 + 4, class: 'val'}, fmt(s.agg[metric].mean)));
  });
  svg.appendChild(S('line', {x1: x(bv), x2: x(bv), y1: top - 4, y2: H - 22, stroke: 'var(--base)', 'stroke-width': 1.6, 'stroke-dasharray': '5 4'}));
  svg.appendChild(S('text', {x: x(bv) + 5, y: top + 6}, 'baseline ' + fmt(bv, 1)));
  const box = el('barchart'); box.innerHTML = ''; box.appendChild(svg);
}

function renderMore() {
  if (!X.data) return;
  el('moredata').hidden = false;
  const pairs = [['mbart_large', 'mbart_large_plusall'], ['byt5_base', 'byt5_base_plusall'], ['mt5_base_lr1e-3', 'mt5_base_lr1e-3_plusall'], ['mt5_base', 'mt5_base_plusall']];
  const s42 = k => { const s = sys[k]; return s ? s.runs.find(r => r.seed === 42) : null; };
  let t = '<thead><tr><th>Model (seed 42, original 325-sentence validation set)</th><th>CER %</th><th>WER %</th><th>BLEU</th><th>chrF</th></tr></thead><tbody>';
  pairs.forEach(([a, b]) => {
    const ra = s42(a), rb = s42(b); if (!ra || !rb) return;
    t += `<tr><td>${dot(a)}${short(a)}</td>` + ['CER', 'WER', 'BLEU', 'chrF'].map(m => `<td class="num">${fmt(ra[m], 1)} → <strong>${fmt(rb[m], 1)}</strong></td>`).join('') + '</tr>';
  });
  el('scaletab').innerHTML = t + '</tbody>';
}

function renderCombined() {
  const rows = X.combined; if (!rows) return;
  el('bysource').hidden = false;
  const sources = ['ALL', 'sellai_manglish', 'sellai_mixed-code', 'dakshina_ml'];
  const get = (s, src) => rows.find(r => r.system === s && r.source === src);
  const systems = ['copy_input', 'rule_collapse_runs'].concat(NEURAL.filter(k => rows.some(r => r.system === k)));
  const maxV = Math.max(...rows.map(r => r.CER.mean));
  let h = '<thead><tr><th>System (CER %, lower is better)</th><th>Seeds</th>' + sources.map(s => `<th>${SRC_NAME[s]}<br><span class="sd">${(get('copy_input', s) || {n: ''}).n.toLocaleString()} sentences</span></th>`).join('') + '</tr></thead><tbody>';
  systems.forEach(k => {
    const r0 = get(k, 'ALL'); if (!r0) return;
    h += `<tr><td>${dot(k)}${cname(k)}</td><td class="num">${r0.seeds}</td>` + sources.map(src => {
      const r = get(k, src); if (!r) return '<td></td>';
      return `<td class="num cell"><span class="mini" style="width:${(r.CER.mean / maxV * 100).toFixed(1)}%;background:${colorOf(k)}"></span><span class="v">${pm(r.CER)}</span></td>`;
    }).join('') + '</tr>';
  });
  el('combtab').innerHTML = h + '</tbody>';
}

const pname = k => CN[k] || short(k);
function renderForest(target, P) {
  if (!P || !P.length) { el(target).parentElement.hidden = true; return; }
  const W = cw(target), left = Math.min(300, Math.round(W * 0.46)), right = 90, rowH = 44, top = 18, H = top + rowH * P.length + 30;
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
    svg.appendChild(S('text', {x: left - 12, y: y + 4, 'text-anchor': 'end', class: 'lab'}, pname(p.a) + ' minus ' + pname(p.b)));
    svg.appendChild(S('line', {x1: x(p.lo), x2: x(p.hi), y1: y, y2: y, stroke: clear ? 'var(--fg)' : 'var(--muted)', 'stroke-width': 3, 'stroke-linecap': 'round'}));
    svg.appendChild(S('circle', {cx: x(p.mean), cy: y, r: 5.5, fill: clear ? 'var(--fg)' : 'var(--surface)', stroke: clear ? 'var(--fg)' : 'var(--muted)', 'stroke-width': 2}));
    svg.appendChild(S('text', {x: x(p.hi) + 10, y: y + 4, class: 'val'}, `${p.mean > 0 ? '+' : ''}${fmt(p.mean)}${clear ? '' : ' (overlaps 0)'}`));
  });
  svg.appendChild(S('text', {x: x(0) - 6, y: top - 5, 'text-anchor': 'end'}, 'first is better'));
  svg.appendChild(S('text', {x: x(0) + 6, y: top - 5}, 'second is better'));
  el(target).innerHTML = ''; el(target).appendChild(svg);
}

function lossChart(target, series, xmax) {
  const W = cw(target), H = 300, l = 40, r = 24, t = 14, b = 34;
  const maxL = Math.ceil(Math.max(...series.flatMap(s => s.loss_history)));
  const x = i => l + (i * 10 / xmax) * (W - l - r), y = v => t + (1 - v / maxL) * (H - t - b);
  const svg = S('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Training loss curves'});
  for (let v = 0; v <= maxL; v += Math.ceil(maxL / 4)) {
    svg.appendChild(S('line', {x1: l, x2: W - r, y1: y(v), y2: y(v), stroke: 'var(--line)'}));
    svg.appendChild(S('text', {x: l - 8, y: y(v) + 4, 'text-anchor': 'end'}, String(v)));
  }
  [0, 0.25, 0.5, 0.75, 1].forEach(f => svg.appendChild(S('text', {x: l + f * (W - l - r), y: H - 12, 'text-anchor': f === 1 ? 'end' : (f === 0 ? 'start' : 'middle')}, String(Math.round(f * xmax)))));
  svg.appendChild(S('text', {x: W - r, y: H - 0, 'text-anchor': 'end'}, 'optimiser steps'));
  series.forEach(s => {
    svg.appendChild(S('polyline', {points: s.loss_history.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: colorOf(s.key), 'stroke-width': 2.2,
      'stroke-dasharray': s.key === 'mt5_base' || s.key === 'mt5_base_plusall' ? '6 4' : '0', 'stroke-linejoin': 'round'}));
    const last = s.loss_history.length - 1;
    svg.appendChild(S('circle', {cx: x(last), cy: y(s.loss_history[last]), r: 3.4, fill: colorOf(s.key)}));
    svg.appendChild(S('text', {x: x(last) - 8, y: y(s.loss_history[last]) - 8, 'text-anchor': 'end', class: 'val'}, fmt(s.loss_history[last])));
  });
  el(target).innerHTML = '<div class="legend">' + series.map(s => `<span>${dot(s.key)}${short(s.key)}${s.key.indexOf('mt5_base') === 0 && s.key.indexOf('lr1e-3') < 0 ? ' (dashed)' : ''}</span>`).join('') + '</div>';
  el(target).appendChild(svg);
}
function renderLoss() {
  const small = D.systems.filter(s => s.key.indexOf('plusall') < 0), big = D.systems.filter(s => s.key.indexOf('plusall') > 0);
  lossChart('loss', small, 550);
  if (big.length) lossChart('loss2', big, 3480); else el('losspanel2').hidden = true;
}

function renderRobust() {
  const rows = X.robustness; if (!rows) return;
  el('robustness').hidden = false;
  const conds = ['base', 'heavy', 'extreme', 'vowel_drop', 'phonetic', 'phonetic_plus_base'];
  const CL = {base: 'as trained', heavy: '2x noise', extreme: '3x noise', vowel_drop: 'vowels dropped', phonetic: 'phonetic spelling', phonetic_plus_base: 'phonetic + noise'};
  const all = ['copy_input', 'rule_collapse_runs'].concat(NEURAL.filter(k => rows.some(r => r.system === k)));
  const CHART = ['copy_input', 'rule_collapse_runs', 'mbart', 'byt5', 'mbart_plusall', 'byt5_plusall', 'mt5_lr1e-3_plusall'];
  const names = all.filter(k => CHART.indexOf(k) >= 0);
  const val = (s, c) => { const r = rows.find(q => q.system === s && q.condition === c); return r ? r.CER.mean : null; };
  const maxV = Math.ceil(Math.max(...rows.filter(r => names.indexOf(r.system) >= 0).map(r => r.CER.mean)) / 10) * 10;
  const W = cw('robchart'), H = 380, l = 46, r = 70, t = 16, b = 64;
  const x = i => l + (i / (conds.length - 1)) * (W - l - r), y = v => t + (1 - v / maxV) * (H - t - b);
  const svg = S('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'CER under each noise condition'});
  for (let v = 0; v <= maxV; v += maxV / 5) {
    svg.appendChild(S('line', {x1: l, x2: W - r, y1: y(v), y2: y(v), stroke: 'var(--line)'}));
    svg.appendChild(S('text', {x: l - 8, y: y(v) + 4, 'text-anchor': 'end'}, String(Math.round(v))));
  }
  conds.forEach((c, i) => svg.appendChild(S('text', {x: x(i), y: H - 38, 'text-anchor': i === conds.length - 1 ? 'end' : (i === 0 ? 'start' : 'middle')}, CL[c])));
  svg.appendChild(S('text', {x: l, y: H - 8}, 'CER %, lower is better'));
  names.forEach(k => {
    const pts = conds.map((c, i) => [x(i), val(k, c)]).filter(p => p[1] !== null);
    const base = k === 'copy_input' || k === 'rule_collapse_runs';
    svg.appendChild(S('polyline', {points: pts.map(p => `${p[0]},${y(p[1])}`).join(' '), fill: 'none', stroke: colorOf(k), 'stroke-width': base ? 1.6 : 2.2,
      'stroke-dasharray': base ? (k === 'copy_input' ? '2 4' : '7 4') : (k.indexOf('plusall') > 0 ? '0' : '5 3'), 'stroke-linejoin': 'round', opacity: base ? 0.8 : 1}));
    pts.forEach(p => svg.appendChild(S('circle', {cx: p[0], cy: y(p[1]), r: 3, fill: colorOf(k)})));
  });
  el('robchart').innerHTML = ''; el('robchart').appendChild(svg);
  el('roblegend').innerHTML = names.map(k => `<span>${dot(k)}${cname(k)}${k.indexOf('plusall') > 0 || k === 'copy_input' || k === 'rule_collapse_runs' ? '' : ' (dashed)'}</span>`).join('');
  let th = '<thead><tr><th>CER % (all systems)</th>' + conds.map(c => `<th>${CL[c]}</th>`).join('') + '<th>Seeds</th></tr></thead><tbody>';
  all.forEach(k => {
    th += `<tr><td>${dot(k)}${cname(k)}</td>` + conds.map(c => { const v = val(k, c); return `<td class="num">${v === null ? '' : fmt(v)}</td>`; }).join('') +
      `<td class="num">${(rows.find(q => q.system === k) || {seeds: ''}).seeds}</td></tr>`;
  });
  el('robtab').innerHTML = th + '</tbody>';
}

function renderDecode() {
  const rows = X.decoding; if (!rows) return;
  el('decoding').hidden = false;
  const cfgs = ['beam1', 'beam1+guard', 'beam4', 'beam4+guard'];
  const CFN = {beam1: 'greedy', 'beam1+guard': 'greedy + guard', beam4: 'beam 4', 'beam4+guard': 'beam 4 + guard'};
  const systems = [...new Set(rows.filter(r => r.system !== 'rule_collapse_runs').map(r => r.system))];
  let h = '<thead><tr><th>System (CER %)</th>' + cfgs.map(c => `<th>${CFN[c]}</th>`).join('') + '<th>Outputs the guard replaced, out of 600</th></tr></thead><tbody>';
  systems.forEach(s => {
    const vals = cfgs.map(c => rows.find(r => r.system === s && r.config === c));
    if (vals.some(v => !v)) return;
    const best = Math.min(...vals.map(v => v.CER.mean));
    h += `<tr><td>${dot(s)}${cname(s)} <span class="sd">(${vals[0].seeds} seeds)</span></td>` + vals.map(v => `<td class="num">${v.CER.mean === best ? '<strong>' + pm(v.CER) + '</strong>' : pm(v.CER)}</td>`).join('') +
      `<td class="num">${fmt(vals[1].guard_replaced, 1)}</td></tr>`;
  });
  const rule = rows.find(r => r.system === 'rule_collapse_runs');
  if (rule) h += `<tr><td>${dot('copy_input')}rule-based cleaner (no model)</td><td class="num" colspan="4">${fmt(rule.CER.mean)}</td><td></td></tr>`;
  el('dectab').innerHTML = h + '</tbody>';
}

function renderExamples() {
  const keys = ['byt5_base', 'mbart_large', 'byt5_base_plusall', 'mbart_large_plusall', 'mt5_base_lr1e-3_plusall'].filter(k => sys[k]);
  let h = '<thead><tr><th>Noisy input</th><th>Clean target</th>' + keys.map(k => `<th>${short(k)}</th>`).join('') + '</tr></thead><tbody>';
  D.examples.forEach(e => {
    h += `<tr><td>${esc(e.noisy)}</td><td>${esc(e.clean)}</td>` + keys.map(k => `<td class="${e[k] === e.clean ? 'ok' : 'no'}">${esc(e[k])}</td>`).join('') + '</tr>';
  });
  el('examples').innerHTML = h + '</tbody>';
}

function renderFails() {
  const P = k => (sys[k] ? sys[k].patterns : {});
  const cards = [
    ['byt5_base', 'ByT5 drops text when data is small', 'Trained on 2,934 sentences, about 1 in 9 outputs on the combined test lost more than 40% of the input: "wine glass kayyil pidichirikkunna old man" came back as "old man". Trained on 18,548 sentences, none did in the run we checked.'],
    ['mbart_large', 'mBART swaps in real words', 'When a word is hard to read it substitutes a fluent one: "birth day celebrate" became "white color", and "car reverse" became "race".'],
    ['mt5_base_lr1e-3', 'mT5 invents English', 'It replaces garbled English words with unrelated ones ("helmet" became "elephant") and leaves more repeated letters in place.'],
  ];
  el('fails').innerHTML = cards.map(([k, t, d]) => {
    const p = P(k);
    const st = (p.over_correction_rate_pct !== undefined ? `over-correction ${fmt(p.over_correction_rate_pct, 1)}%` : '') + (p.unresolved_elongation_pct !== undefined ? ` · unresolved repeats ${fmt(p.unresolved_elongation_pct, 1)}%` : '');
    return `<div class="fail" style="border-top-color:${colorOf(k)}"><h3>${t}</h3><p>${d}</p><div class="stat">${st}</div></div>`;
  }).join('');
}

function renderCompute() {
  const R = X.runs; if (!R) return;
  el('compute').hidden = false;
  const groups = {};
  R.runs.forEach(r => (groups[r.system] = groups[r.system] || []).push(r));
  const order = Object.keys(groups).sort((a, b) => (a.indexOf('plusall') > 0) - (b.indexOf('plusall') > 0) || a.localeCompare(b));
  let h = '<thead><tr><th>System</th><th>Training sentences</th><th>Runs</th><th>Steps per run</th><th>Micro-batch × accumulation</th><th>Minutes per run</th><th>Total minutes</th><th>Peak GPU GB</th></tr></thead><tbody>';
  order.forEach(k => {
    const g = groups[k], mins = g.map(r => r.minutes), f = g[0];
    const range = g.length > 1 ? `${fmt(mins.reduce((a, b) => a + b, 0) / g.length, 1)} <span class="sd">(${fmt(Math.min(...mins), 1)} to ${fmt(Math.max(...mins), 1)})</span>` : fmt(mins[0], 1);
    h += `<tr><td>${dot(k)}${short(k)}</td><td class="num">${f.train_rows.toLocaleString()}</td><td class="num">${g.length}</td><td class="num">${f.steps.toLocaleString()}</td><td class="num">${f.micro_batch} × ${f.grad_accum}</td><td class="num">${range}</td><td class="num">${fmt(mins.reduce((a, b) => a + b, 0), 1)}</td><td class="num">${fmt(Math.max(...g.map(r => r.peak_gpu_gb)))}</td></tr>`;
  });
  h += `<tr class="total"><td>All training</td><td></td><td class="num">${R.runs.length}</td><td></td><td></td><td></td><td class="num">${fmt(R.training_minutes, 1)}</td><td></td></tr>`;
  el('runtab').innerHTML = h + '</tbody>';
  const NAMES = {eval_combined: 'Score every adapter (17) on the 2,059-sentence combined test set', decode_study: 'Decoding study: 4 settings, 17 adapters, 600 sentences',
    robustness: 'Robustness test: 6 noise conditions, 17 adapters, 600 sentences', seeds: 'Aggregate seeds and run the bootstrap comparisons'};
  let s = '<thead><tr><th>Step</th><th>Minutes</th></tr></thead><tbody>';
  R.steps.forEach(x => { s += `<tr><td>${NAMES[x.name] || x.name}</td><td class="num">${fmt(x.minutes, 1)}</td></tr>`; });
  s += `<tr class="total"><td>All evaluation and studies</td><td class="num">${fmt(R.step_minutes, 1)}</td></tr>`;
  el('steptab').innerHTML = s + '</tbody>';
  const gpu = R.hardware.split(',')[0].replace('NVIDIA GeForce ', ''), gb = Math.round(parseInt(R.hardware.split(',')[1], 10) / 1024);
  el('compute-sub').textContent = `Everything ran on one ${gpu} with ${gb} GB of memory: ${fmt(R.training_minutes / 60, 1)} hours of training in ${R.runs.length} runs and ${fmt(R.step_minutes / 60, 1)} hours of evaluation and studies, ${fmt((R.training_minutes + R.step_minutes) / 60, 1)} hours in all. Runs on the larger data take 3,480 steps instead of 552, and use the largest micro-batch that fits in memory; the effective batch is 16 in every run.`;
}

function fillCopy() {
  const C = D.copy;
  el('headline').textContent = C.headline;
  el('lede').innerHTML = C.lede;
  el('results-sub').innerHTML = C.resultsSub;
  el('pair-sub').innerHTML = C.pairSub;
  el('recotext').innerHTML = C.reco;
  el('limitlist').innerHTML = C.limits.map(x => `<li>${x}</li>`).join('');
  el('devs').innerHTML = C.devs.map(x => `<li>${x}</li>`).join('');
  [['dataSub', 'data-sub'], ['saniNote', 'sani-note'], ['moreSub', 'more-sub'], ['combSub', 'comb-sub'], ['robSub', 'rob-sub'], ['decSub', 'dec-sub']].forEach(([k, id]) => { if (C[k]) el(id).innerHTML = C[k]; });
}

function renderCharts() {
  renderBars(); renderForest('forest', D.pairs); renderForest('forest2', D.pairs_combined); renderLoss(); renderRobust();
}
fillCopy(); renderTable(); renderTabs(); renderLegend(); renderExamples(); renderFails();
renderDatasets(); renderMore(); renderCombined(); renderDecode(); renderCompute(); renderTiles();
renderCharts(); renderNav();
let resizeTimer = null;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderCharts, 150); });
el('themebtn').onclick = () => {
  const r = document.documentElement, cur = r.getAttribute('data-theme');
  const dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  r.setAttribute('data-theme', dark ? 'light' : 'dark');
};
</script>
"""

EXTRA_FILES = (("combined", "combined_all/combined_summary.json"), ("robustness", "robustness_all/robustness.json"),
               ("decoding", "decoding_all/decoding_summary.json"), ("data", "data/train_plus_all.stats.json"),
               ("profile", "data/dataset_profile.json"), ("runs", "run_times.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "report"), help="where index.html and fragment.html are written")
    ap.add_argument("--copy", default=str(ROOT / "report" / "copy.json"), help="the page text")
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder holding report_data.json and the study outputs")
    a = ap.parse_args(argv)
    res = Path(a.results)
    data = json.loads((res / "report_data.json").read_text(encoding="utf-8"))
    data["copy"] = json.loads(Path(a.copy).read_text(encoding="utf-8"))
    data["extra"] = {}
    for key, rel in EXTRA_FILES:
        p = res / rel
        data["extra"][key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    body = BODY.replace("__DATA__", payload)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fragment.html").write_text(HEAD + body, encoding="utf-8")
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + HEAD + '</head>\n<body>\n' + body + '</body>\n</html>\n')
    (out / "index.html").write_text(full, encoding="utf-8")
    print("wrote", out / "index.html", f"({len(full) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
