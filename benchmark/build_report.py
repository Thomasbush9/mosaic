"""Build the standalone matched-protocol report.

Every figure is inlined as a base64 data URI rather than linked, for two
reasons: the Artifact CSP blocks external hosts outright, and the report has to
stay readable when the cluster holding the PNGs is unreachable.

    python benchmark/build_report.py --benchmark <dir> --out <file.html>
"""

import argparse
import base64
from pathlib import Path

FIGS = [
    ("topk", "matrix_figs_matched/topk_agreement.png"),
    ("accuracy", "accuracy_figs_matched/accuracy_nipah-glycoprotein-g.png"),
    ("matrices", "matrix_figs_matched/matrices_complex.png"),
    ("monomer", "matrix_figs_matched/matrices_monomer.png"),
    ("redundancy", "matrix_figs_matched/metric_redundancy_complex.png"),
]


def data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


HTML = """<title>Six Knobs and a Held-Out Judge</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&display=swap">
<style>
:root{
  --ground:#fcfcfb; --raised:#ffffff; --sunken:#f4f3f0;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#83817b;
  --rule:#e3e2de; --rule-2:#d3d1cb;
  --accent:#2a78d6; --accent-soft:#e8f0fb;
  --warn:#c2551f; --warn-soft:#fbeee7;
  --critical:#d03b3b;
  --plate:#fcfcfb;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#131316; --raised:#1b1b1f; --sunken:#0e0e11;
    --ink:#f2f1ee; --ink-2:#b3b1aa; --ink-3:#87857e;
    --rule:#2c2c31; --rule-2:#3b3b41;
    --accent:#6aa6ea; --accent-soft:#16263a;
    --warn:#e8905f; --warn-soft:#33200f;
    --critical:#e86b6b;
    --plate:#e9e8e4;
  }
}
:root[data-theme="dark"]{
  --ground:#131316; --raised:#1b1b1f; --sunken:#0e0e11;
  --ink:#f2f1ee; --ink-2:#b3b1aa; --ink-3:#87857e;
  --rule:#2c2c31; --rule-2:#3b3b41;
  --accent:#6aa6ea; --accent-soft:#16263a;
  --warn:#e8905f; --warn-soft:#33200f;
  --critical:#e86b6b;
  --plate:#e9e8e4;
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Serif",Georgia,serif;
  font-size:17px; line-height:1.65;
  margin:0; padding:0 24px 96px;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto}
.measure{max-width:68ch}
h1,h2,h3,.label,.tag,table,.meta,.stat{font-family:"IBM Plex Sans",system-ui,sans-serif}
h1{font-size:clamp(2rem,4.2vw,3.1rem);line-height:1.06;letter-spacing:-.022em;
   font-weight:600;margin:0 0 .5rem;text-wrap:balance}
h2{font-size:1.5rem;font-weight:600;letter-spacing:-.012em;margin:0 0 .2rem;text-wrap:balance}
h3{font-size:1.06rem;font-weight:600;margin:2.2rem 0 .5rem;text-wrap:balance}
p{margin:0 0 1.05rem}
a{color:var(--accent)}
strong{font-weight:600}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.87em}
code{background:var(--sunken);padding:.1em .34em;border-radius:3px;
     border:1px solid var(--rule)}

header{padding:72px 0 0;border-bottom:1px solid var(--rule);margin-bottom:44px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.73rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--warn);margin:0 0 1.1rem}
.thesis{font-size:1.24rem;line-height:1.5;color:var(--ink-2);max-width:60ch;margin:0 0 2rem}
.meta{display:flex;flex-wrap:wrap;gap:0 2.4rem;font-size:.78rem;color:var(--ink-3);
  border-top:1px solid var(--rule);padding:14px 0 22px;font-variant-numeric:tabular-nums}
.meta b{color:var(--ink-2);font-weight:500}

section{margin:0 0 60px}
.lede{font-size:1.06rem;color:var(--ink-2);margin-bottom:1.6rem;max-width:64ch}

/* Figures are light-ground PNGs; give them an explicit plate so they read as
   printed plates in dark mode instead of glowing rectangles. */
figure{margin:30px 0 34px}
.plate{background:var(--plate);border:1px solid var(--rule-2);border-radius:4px;
  padding:10px;overflow-x:auto}
.plate img{display:block;width:100%;min-width:680px;height:auto}
figcaption{font-family:"IBM Plex Sans",sans-serif;font-size:.83rem;color:var(--ink-3);
  margin-top:10px;line-height:1.5;max-width:78ch}
figcaption b{color:var(--ink-2);font-weight:500}

.tablewrap{overflow-x:auto;margin:22px 0 26px}
table{border-collapse:collapse;font-size:.86rem;width:100%;min-width:440px;
  font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink-2);font-size:.72rem;
  letter-spacing:.07em;text-transform:uppercase;
  border-bottom:1px solid var(--rule-2);padding:0 14px 8px 0;white-space:nowrap}
td{padding:8px 14px 8px 0;border-bottom:1px solid var(--rule);vertical-align:top}
td.num,th.num{text-align:right;padding-right:20px}
tr:last-child td{border-bottom:none}
.up{color:var(--accent);font-weight:600}
.down{color:var(--warn);font-weight:600}
.win{background:var(--accent-soft)}

.tag{display:inline-block;font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  font-weight:600;padding:.18em .5em;border-radius:3px;white-space:nowrap}
.tag.fixed{background:var(--accent-soft);color:var(--accent)}
.tag.open{background:var(--warn-soft);color:var(--warn)}

.callout{border-left:2px solid var(--warn);background:var(--sunken);
  padding:18px 22px;margin:26px 0;border-radius:0 4px 4px 0}
.callout p:last-child{margin-bottom:0}
.callout .label{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--warn);font-weight:600;margin-bottom:.5rem}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin:26px 0}
.stat{background:var(--ground);padding:16px 18px}
.stat .k{font-size:1.55rem;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.stat .v{font-size:.75rem;color:var(--ink-3);margin-top:5px;line-height:1.35}

footer{border-top:1px solid var(--rule);padding-top:24px;margin-top:56px;
  font-size:.8rem;color:var(--ink-3);font-family:"IBM Plex Sans",sans-serif}
footer code{font-size:.82em}
@media (max-width:640px){body{font-size:16px;padding:0 16px 64px}header{padding-top:44px}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Binder-scoring benchmark &middot; matched re-run</p>
  <h1>Six Knobs and a Held-Out Judge</h1>
  <p class="thesis">Six structure models were never being compared on equal terms.
  Six separate harness settings differed between them &mdash; one of them silently
  gave Protenix a two-step diffusion sample against Boltz&rsquo;s twenty-five. With all
  six matched, the ranking changes, agreement rises, and the finding that actually
  bears on the campaign turns out to be about where consensus stops.</p>
  <div class="meta">
    <span><b>Agreement</b> 416 designs &times; 3 reps &times; 6 models</span>
    <span><b>Accuracy</b> 434 pairs, 56 binders, Nipah-G</span>
    <span><b>NaN</b> 0</span>
    <span><b>Failed designs</b> 0</span>
    <span><b>2026-08-25</b></span>
  </div>
</header>

<section>
  <h2>What was wrong</h2>
  <p class="lede">Each of these made two models see different compute while the
  submit command looked identical. All six are now pinned and echoed in a
  <code>PROTOCOL</code> line every run prints.</p>
  <div class="tablewrap"><table>
    <tr><th>Knob</th><th>What differed</th><th>Effect</th><th>Status</th></tr>
    <tr><td>Target MSA</td><td>OF3, Protenix, ESMFold2 and OpenDDE ignored
      <code>msa_path</code> and queried the ColabFold server</td>
      <td>AF2/Boltz on a 3,032-seq local alignment, ESMFold2 on a 3,643-seq
      server one &mdash; both files still in the cache</td>
      <td><span class="tag fixed">fixed</span></td></tr>
    <tr><td>Diffusion steps</td><td>Unset meant each backend&rsquo;s own default</td>
      <td>Boltz 25, OF3 20, ESMFold2 14, <strong>Protenix 2</strong></td>
      <td><span class="tag fixed">fixed</span></td></tr>
    <tr><td>Trunk passes</td><td>OF3 and ESMFold2 add one internally</td>
      <td><code>--recycling 3</code> gave three models 3 passes, two models 4</td>
      <td><span class="tag fixed">fixed</span></td></tr>
    <tr><td>Featurization</td><td><code>binder</code> stubs sidechains to poly-G</td>
      <td>Uniform but not neutral &mdash; penalises all-atom models differentially</td>
      <td><span class="tag fixed">fixed</span></td></tr>
    <tr><td>AF2 parameter set</td><td>Ensemble draw vs pinned <code>model_idx</code></td>
      <td>AF2 reproduced its own ranking at 0.786 vs 0.972</td>
      <td><span class="tag fixed">fixed</span></td></tr>
    <tr><td>Sample count</td><td>1 vs 6 diffusion samples</td>
      <td>Moved Boltz-2 ~ OF3 by ~0.08</td>
      <td><span class="tag fixed">fixed</span></td></tr>
  </table></div>

  <div class="callout">
    <div class="label">The control that says the fix is inert</div>
    <p>Rescoring the same designs under old and new settings moved <strong>AF2 by
    exactly 0.0000</strong> on every metric, and Boltz-1/2 by ~1e-4 &mdash; nothing about
    their protocol changed. Protenix moved +0.080 ipTM. Only the models whose
    protocol actually changed moved, which is what rules out an incidental break
    somewhere else in the harness.</p>
  </div>
</section>

<section>
  <h2>Agreement rose once the protocol matched</h2>
  <p class="lede">Much of what looked like models disagreeing was the harness.
  Diagonals tightened from a ragged 0.79&ndash;1.00 to 0.95&ndash;0.98, so the
  off-diagonals now sit against a solid ceiling.</p>
  <div class="stats">
    <div class="stat"><div class="k">0.485 &rarr; <span class="up">0.590</span></div>
      <div class="v">complex ipSAE<sub>min</sub>, mean off-diagonal</div></div>
    <div class="stat"><div class="k">0.613 &rarr; <span class="up">0.696</span></div>
      <div class="v">complex ipTM</div></div>
    <div class="stat"><div class="k">0.436 &rarr; <span class="up">0.612</span></div>
      <div class="v">AF2 ~ OF3 &mdash; the ranking run says 0.612</div></div>
    <div class="stat"><div class="k">0.95&ndash;0.98</div>
      <div class="v">self-agreement, all six models</div></div>
  </div>
  <div class="measure">
  <p><strong>af2~of3 reproduces the ranking run exactly.</strong> The old 0.436 came
  from a truncated 100-design OF3 file, not from the models &mdash; the last open gap
  in the protocol table, closed.</p>
  <p>Two earlier claims are now dead. &ldquo;Boltz-2 &harr; OF3 is 0.35, one of the
  lowest pairs&rdquo; &mdash; it is 0.671, mid-pack, in a band that runs only
  0.602&ndash;0.723. Architecture lineage still fails to predict agreement, but the
  evidence previously offered for that was a protocol artifact.</p>
  </div>
  <figure>
    <div class="plate"><img src="__MATRICES__" alt="Model by model agreement matrices, complex condition, one panel per metric"></div>
    <figcaption><b>Agreement matrices, complex condition.</b> Spearman between each
    model pair, one matrix per metric, 416 designs. The diagonal is measured from
    replicates rather than assumed to be 1.0 &mdash; without it an off-diagonal of
    0.6 has no scale.</figcaption>
  </figure>
</section>

<section>
  <h2>Consensus does not survive to the top</h2>
  <p class="lede">This is the result that bears on the campaign, and it is not
  visible in any global number.</p>
  <div class="measure">
  <p>A Spearman over a set spanning bad&nbsp;&rarr;&nbsp;good is dominated by both
  models agreeing that the bad designs are bad. That is the easy half of the
  problem and not the half anyone makes decisions on. Restrict to the top 25 of
  each model&rsquo;s own ranking &mdash; the region a campaign actually draws from
  &mdash; and agreement collapses.</p>
  <p>Restriction attenuates a correlation mechanically, so a drop alone proves
  nothing. The control is each model against <em>itself</em> on the same restricted
  subset: it pays exactly the same penalty. Where cross sits far below self, the
  disagreement is real.</p>
  </div>
  <figure>
    <div class="plate"><img src="__TOPK__" alt="Left: every pair, global vs top-25 agreement. Right: self-agreement ceiling vs mean cross-model agreement."></div>
    <figcaption><b>Left:</b> every pair, all 416 designs (blue) against the same
    pair restricted to the top 25 (orange). <b>Right:</b> the control &mdash; each
    model against its own replicate, same restriction. Models reproduce themselves
    at 0.87&ndash;0.98 while agreeing with each other at 0.03&ndash;0.46.</figcaption>
  </figure>
  <div class="tablewrap"><table>
    <tr><th>Model</th><th class="num">Agrees with itself</th><th class="num">Agrees with others</th><th>Reading</th></tr>
    <tr><td>Protenix</td><td class="num">0.98</td><td class="num">0.14</td><td>stable, dissenting</td></tr>
    <tr><td>Boltz-1</td><td class="num">0.97</td><td class="num">0.08</td><td>stable, dissenting</td></tr>
    <tr><td>Boltz-2</td><td class="num">0.97</td><td class="num">0.03</td><td>design model &mdash; scores compressed ~4&times;</td></tr>
    <tr><td>ESMFold2</td><td class="num">0.90</td><td class="num">0.18</td><td>stable, dissenting</td></tr>
    <tr><td>AF2</td><td class="num">0.87</td><td class="num">0.46</td><td>most consensus-like at the top</td></tr>
    <tr><td>OpenFold3</td><td class="num">0.51</td><td class="num">0.06</td><td>unstable &mdash; low cross says little</td></tr>
  </table></div>
  <div class="callout">
    <div class="label">Goodhart, made visible</div>
    <p>Boltz-2&rsquo;s scores are compressed roughly fourfold against every non-Boltz
    model (sd 0.101 vs 0.33&ndash;0.42). These designs were optimized against its
    composite, so it rates nearly all of them excellent and has little
    discriminative range left where it matters. The same circularity already
    logged for MPNN sequence recovery, now visible in the design model&rsquo;s own
    confidence.</p>
  </div>
</section>

<section>
  <h2>Accuracy against real binders</h2>
  <p class="lede">ROC-AUC separating 56 experimentally validated binders from 378
  non-binders on Nipah-G, with the codebase sign convention applied so
  PAE reads higher&nbsp;=&nbsp;better. The bar is the length control at 0.64, not 0.5.</p>
  <div class="tablewrap"><table>
    <tr><th>Model</th><th class="num">Best</th><th>Metric</th><th class="num">Median</th><th class="num">&gt; control</th><th class="num">Was</th></tr>
    <tr class="win"><td><strong>ESMFold2</strong></td><td class="num">0.833</td><td>tb_pae</td><td class="num">0.810</td><td class="num">11/11</td><td class="num">0.822</td></tr>
    <tr class="win"><td><strong>Boltz-2</strong></td><td class="num">0.841</td><td>ipLDDT</td><td class="num">0.765</td><td class="num">10/11</td><td class="num">0.815</td></tr>
    <tr><td>Protenix</td><td class="num"><span class="up">0.680</span></td><td>tb_pae</td><td class="num">0.652</td><td class="num">5/11</td><td class="num">0.552</td></tr>
    <tr><td>Boltz-1</td><td class="num">0.671</td><td>tb_pae</td><td class="num">0.638</td><td class="num">3/11</td><td class="num">0.647</td></tr>
    <tr><td>AF2</td><td class="num">0.647</td><td>pTM energy</td><td class="num">0.607</td><td class="num">1/11</td><td class="num">0.647</td></tr>
    <tr><td>OpenFold3</td><td class="num"><span class="down">0.597</span></td><td>ipSAE<sub>min</sub></td><td class="num">0.583</td><td class="num">0/11</td><td class="num">0.652</td></tr>
  </table></div>
  <div class="measure">
  <p><strong>Protenix moved from last to third.</strong> Best 0.552&nbsp;&rarr;&nbsp;0.680,
  median 0.492&nbsp;&rarr;&nbsp;0.652, and from 0 to 5 metrics above control. It had been
  scored on a two-step diffusion sample; at 25 steps its confidence metrics go
  from indistinguishable from chance to genuinely predictive. Its last-place
  finish was the harness, not the model.</p>
  <p><strong>OpenFold3 moved the other way</strong>, 0.652&nbsp;&rarr;&nbsp;0.597 and now
  below control on every metric. It lost a deeper server MSA and one trunk pass.
  Some of its earlier showing was extra compute and a different alignment.</p>
  </div>
  <figure>
    <div class="plate"><img src="__ACCURACY__" alt="Model by metric ROC-AUC heatmap for Nipah-G with sequence-only negative controls"></div>
    <figcaption><b>AUC per model &times; metric.</b> The three rows below the rule are
    sequence-only negative controls &mdash; molecular weight, net charge, length.
    Length reads 0.36, i.e. inverted, so as a predictor it is worth 0.64: that,
    not 0.5, is the threshold a model has to clear.</figcaption>
  </figure>
</section>

<section>
  <h2>So which model referees the campaign</h2>
  <div class="measure">
  <p><strong>ESMFold2</strong> is the defensible pick. Highest median at 0.810, the
  only model above the control bar on all eleven metrics, and it did not design
  these binders. Boltz-2 scores comparably &mdash; 0.841 at its best &mdash; but it
  is the design model and cannot referee its own work.</p>
  <p>Three things temper that, and they should travel with the recommendation.</p>
  <p><strong>MSA depth is confounded with the winner.</strong> Nipah&rsquo;s alignment is
  64 sequences; DIO3&rsquo;s is 3,032. The model that leans hardest on an MSA (AF2)
  finishes near the bottom here, and this ranking is roughly what MSA depth alone
  would predict. It should not be carried to DIO3 unexamined &mdash; a deep-MSA
  target is what settles it.</p>
  <p><strong>One target, 56 positives, no cross-target replication.</strong> EGFR, IL7R
  and PD-L1 have labels but no recoverable target construct.</p>
  <p><strong>Naming a judge is the decision, not a tie-break.</strong> ESMFold2 agrees
  with the other candidate judges at 0.18 in the region you draw from. Picking it
  largely determines which designs advance rather than ratifying a consensus that
  exists.</p>
  </div>
</section>

<section>
  <h2>Supporting panels</h2>
  <figure>
    <div class="plate"><img src="__MONOMER__" alt="Agreement matrices for the monomer condition"></div>
    <figcaption><b>Monomer condition</b> &mdash; the binder folded alone, no MSA.
    Agreement runs higher here than for the complex (pLDDT 0.700, Rg 0.907):
    <b>the models agree far more about whether a binder folds than about whether it
    docks.</b></figcaption>
  </figure>
  <figure>
    <div class="plate"><img src="__REDUNDANCY__" alt="Metric by metric correlation within each model, complex condition"></div>
    <figcaption><b>Metric redundancy within each model.</b> Most confidence metrics
    are near-duplicates of each other, which is why the eleven columns of the
    accuracy table carry far less independent evidence than their count
    suggests.</figcaption>
  </figure>
</section>

<footer>
  <p>Agreement <code>mfull_*</code>, job 41676259 &middot; accuracy <code>macc_*</code>,
  jobs 41678846 and 41682157 &middot; protocol identical across both: local a3m,
  <code>target_only</code>, 6 samples, 3 trunk passes, 25 diffusion steps, seed 42.
  Tables in <code>benchmark/matched_summary/</code>, full write-up in
  <code>BENCHMARK.md</code>. Figures embedded, so this page reads with the cluster
  offline.</p>
</footer>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    html = HTML
    for key, rel in FIGS:
        p = args.benchmark / rel
        if not p.exists():
            raise SystemExit(f"missing figure: {p}")
        html = html.replace(f"__{key.upper()}__", data_uri(p))

    args.out.write_text(html)
    mb = len(html.encode()) / 1e6
    print(f"wrote {args.out}  ({mb:.2f} MB, {len(FIGS)} figures inlined)")
    if mb > 15:
        raise SystemExit("over the 16 MB artifact limit")


if __name__ == "__main__":
    main()
