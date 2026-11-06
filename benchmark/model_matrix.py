"""Score one design set with one model, in one condition, and emit a tidy table.

The deliverable is a stack of model x model agreement matrices -- one per
metric, for each of two conditions:

    complex   binder + target (target gets its real MSA)
    monomer   binder alone (no MSA exists for a de novo binder, by definition)

Every matrix is a groupby on the single table this writes:

    design_idx | seq | model | condition | replicate | metric | value

WHY REPLICATES. A model x model matrix is unreadable without knowing what a
model scores against ITSELF, so the off-diagonals only mean something relative
to a measured diagonal, not an assumed 1.0. Replicate 0 runs the full design
set; replicates 1..n-1 run the first --diag-n designs, which is plenty to
estimate a within-model correlation and keeps the cost down.

The replicate axis is per-model, because the models are stochastic in different
places. Every diffusion backend samples a trajectory from the key, so its
replicate is a fold_in on the seed. AF2 is not stochastic at all: it sets
global_config.deterministic = True and eval_dropout = False (models/af2.py:114-
118), so the same input and model_idx returns bit-identical output. It therefore
has TWO candidate axes, and they measure different things:

    model_idx   five SEPARATELY TRAINED networks -- ensemble spread
    msa_seed    MSA clusters resampled within ONE network

Only msa_seed is the like-for-like analogue of a diffusion reseed. Measured on
100 DIO3 designs (benchmark/af2_replicate_axes.csv): ipsae_min 0.72 ensemble vs
0.87 sampling, iptm 0.76 vs 0.90. Charging AF2 the ensemble number while every
other model pays only sampling noise understates AF2's reliability, so use
--af2-replicate msa_seed for a fair diagonal. The ensemble number is still the
right one operationally, because model_output draws model_idx at random when it
is not pinned (af2.py:501).

msa_seed does nothing when the MSA is shallower than max_msa_clusters (512), and
nothing at all in the monomer condition, where there is no MSA to resample.

FEATURES ARE BUILT PER LENGTH. A feature pack is tied to a binder length, and a
new length means a new JIT compile -- which for the slower backends costs more
than the folds it enables. Designs are therefore grouped by length, and each
group pays one featurization and one compile. A uniform-length set (the DIO3
hallucination designs) collapses to a single group and behaves exactly as
before; the Adaptyv sets have 15-167 distinct lengths per target, which is why
--max-length-groups exists.

binder_features() is the correct call rather than target_only_features([binder,
target]): it stubs the binder's reference sidechain atoms, which is what you
want when one pack is reused across many different binder sequences (see
CLAUDE.md on the split).

LABELS. --label-col carries an experimental readout through to the output table
untouched, so the same runner serves both jobs: model-vs-model agreement on
unlabelled designs, and model-vs-truth accuracy on the Adaptyv sets.
"""

import argparse
import math
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.structure_prediction import TargetChain

CA = 1
IFACE_CUTOFF = 8.0      # A, CA-CA, for interface pLDDT
FLUSH_EVERY = 25


def load_model(name):
    """Constructors are matched to the checkpoints actually on disk.

    Protenix is the *mini* v0.5.0 build (weights/protenix/) -- the other
    Protenix variants would trigger a download.
    """
    if name == "af2":
        from mosaic.models.af2 import AlphaFold2
        return AlphaFold2()
    if name == "boltz1":
        from mosaic.models.boltz1 import Boltz1
        return Boltz1()
    if name == "boltz2":
        from mosaic.models.boltz2 import Boltz2
        return Boltz2()
    if name == "of3":
        from mosaic.models.of3 import OF3
        return OF3()
    if name == "protenix":
        from mosaic.models.protenix import ProtenixMini
        return ProtenixMini()
    if name == "esmfold2":
        # Full, not Fast: Fast has no MSA encoder and *raises* if any chain sets
        # use_msa (models/esmfold2.py:271). The target must carry its MSA in the
        # complex condition, so Fast cannot participate on equal terms.
        from mosaic.models.esmfold2 import ESMFold2Full
        return ESMFold2Full()
    raise ValueError(f"unknown model {name}")


MODELS = ["af2", "boltz1", "boltz2", "of3", "protenix", "esmfold2"]
CONDITIONS = ["complex", "monomer"]

# --- Protocol normalization -------------------------------------------------
#
# "Same recycling, same sampling" has to mean the same COMPUTE, not the same
# argument. Two knobs do not survive being passed through unchanged.
#
# RECYCLING. Every backend takes recycling_steps, and they do not agree on what
# it counts. AF2 (af2.py: scan(length=recycling_steps)), Boltz-2 (losses/
# boltz2.py: scan(length=recycling_steps)), Boltz-1 and Protenix (losses/
# protenix.py: while iter < recycling_steps-1, then one final body call) all
# treat it as the number of trunk passes. OF3 adds one (of3.py: run_trunk(
# batch, recycling_steps + 1) -> fori_loop(0, num_cycles)), and ESMFold2's
# sister library does the same (esmjfold2/model.py: total_steps = num_loops+1).
# So --recycling 3 gave three models 3 passes and two models 4. The offset
# below cancels that, and TRUNK_PASSES is what the run should be described by.
RECYCLING_OFFSET = {"of3": -1, "esmfold2": -1}

# DIFFUSION SAMPLING. Leaving --sampling-steps unset does NOT mean "the same";
# it means each backend falls back to its own default, and those differ by more
# than an order of magnitude:
#
#     Boltz-1 / Boltz-2   25   (losses/boltz*.py, `if ... is not None else 25`)
#     OF3                 20   (of3.py: default_sampling_steps)
#     ESMFold2            14   (esmfold2.py: _DEFAULT_NUM_SAMPLING_STEPS)
#     Protenix (mini)      2   (protenix.py:194 -- ProtenixMini(..., 2))
#
# Protenix scoring a design on a 2-step diffusion sample while Boltz gets 25 is
# not a model difference, and it is invisible in the output table. Runs before
# 2026-08-24 all carry it. --sampling-steps now defaults to a single value for
# every diffusion backend; AF2 has no diffusion and is never handed one.
DEFAULT_SAMPLING_STEPS = 25


def fold(name, model, pssm, features, *, replicate, sample, key, recycling,
         sampling, af2_model_idx=None):
    kw = {}
    if name == "af2":
        # deterministic=True -> the key does nothing; the parameter set is the
        # only replicate axis, and there are exactly five of them. Pinning it
        # kills that axis, which is why --af2-model-idx demands msa_seed.
        kw["model_idx"] = (replicate % 5 if af2_model_idx is None
                           else af2_model_idx)
        k = key
    else:
        k = jax.random.fold_in(jax.random.fold_in(key, replicate), sample)
        # AF2 has no diffusion sampler; every other backend does.
        kw["sampling_steps"] = sampling
    return model.model_output(
        PSSM=pssm, features=features,
        recycling_steps=recycling + RECYCLING_OFFSET.get(name, 0),
        key=k, **kw
    )


def _interface_plddt(out, L):
    """Mean pLDDT over binder residues in contact with the target.

    Not a mosaic LossTerm -- 'ipLDDT' as reported by Adaptyv/Proteinbase is a
    geometric selection, so it is computed here from the predicted coords.
    Returns NaN when nothing is in contact (a failed dock has no interface, and
    averaging over an empty set would silently report 0).
    """
    ca = np.asarray(out.atom37_coords[:, CA])
    b, t = ca[:L], ca[L:]
    d = np.linalg.norm(b[:, None, :] - t[None, :, :], axis=-1)
    contact = (d < IFACE_CUTOFF).any(axis=1)
    if not contact.any():
        return float("nan")
    return float(np.asarray(out.plddt[:L])[contact].mean())


def build_features(model, *, L, seq, target, msa_path, condition,
                   featurization):
    """One feature pack.

    binder_features stubs the binder's reference sidechain atoms to poly-G/UNK,
    so one pack serves every sequence of a given length -- cheap, but it denies
    the all-atom models the sidechain conditioning they were trained with.
    target_only_features bakes the real binder sequence in, so it must be
    rebuilt per design; shapes are unchanged, so it costs a featurization but
    no recompile. hallucinate.py:131 and rank_designs.py:226 both use the
    latter, which is why the two protocols disagree (see BENCHMARK.md).
    """
    if featurization == "target_only":
        chains = [TargetChain(sequence=seq, use_msa=False)]
        if condition == "complex":
            chains.append(TargetChain(sequence=target, use_msa=True,
                                      msa_path=str(msa_path)))
        return model.target_only_features(chains=chains)[0]

    if condition == "complex":
        chains = [TargetChain(sequence=target, use_msa=True,
                              msa_path=str(msa_path))]
    else:
        # binder_features(L, []) reduces to a single all-G chain -- the binder
        # alone, no MSA. A de novo binder has no homologs to align.
        chains = []
    return model.binder_features(binder_length=int(L), chains=chains)[0]


def mean_metrics(dicts):
    """Mean over diffusion samples, ignoring NaN.

    Equivalent to build_multisample_loss(reduction=jnp.mean) for any loss that
    is a linear combination of these metrics -- mean-of-metrics and
    mean-of-losses coincide. iplddt is NaN when nothing docked, and a design
    that failed to dock in some samples should still report the samples where
    it did, so the NaNs are dropped rather than poisoning the mean.
    """
    out = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if d.get(k) is not None
                and not math.isnan(d[k])]
        out[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def metrics(pssm, out, *, condition, L, key):
    """All metrics for one fold. Aux dicts carry the natural sign."""
    def aux(term, field):
        return float(term(pssm, out, key=key)[1][field])

    if condition == "monomer":
        # No second chain exists, so every interface quantity is undefined.
        return {
            "plddt": float(np.asarray(out.plddt).mean()),
            "ptm": aux(sp.BinderPTMLoss(), "binder_ptm"),
            "pae": aux(sp.WithinBinderPAE(), "bb_pae"),
            "rg": aux(sp.ActualRadiusOfGyration(target_radius=0.0), "actual_rg"),
        }

    # The exact composite hallucinate.py:137 and rank_designs.py:232 rank on:
    # 1.00*IPTMLoss + 0.5*TargetBinderIPSAE + 0.5*BinderTargetIPSAE. Those
    # LossTerms each return the negated score, so the loss they report is
    # -(iptm + 0.5*tb_ipsae + 0.5*bt_ipsae) -- recorded here as the score, so
    # every metric in this table keeps the natural "higher is better" sign.
    # Averaging it over samples equals build_multisample_loss(reduction=mean),
    # because the composite is linear in its three parts.
    iptm = aux(sp.IPTMLoss(), "iptm")
    tb_ipsae = aux(sp.TargetBinderIPSAE(), "tb_ipsae")
    bt_ipsae = aux(sp.BinderTargetIPSAE(), "bt_ipsae")

    return {
        "rank_composite": iptm + 0.5 * tb_ipsae + 0.5 * bt_ipsae,
        "ipsae_min": aux(sp.IPSAE_min(), "ipsae_min"),
        "bt_ipsae": bt_ipsae,
        "tb_ipsae": tb_ipsae,
        "iptm": iptm,
        "bt_iptm": aux(sp.BinderTargetIPTM(), "bt_iptm"),
        "binder_ptm": aux(sp.BinderPTMLoss(), "binder_ptm"),
        "binder_plddt": aux(sp.PLDDTLoss(), "plddt"),
        "complex_plddt": float(np.asarray(out.plddt).mean()),
        "iplddt": _interface_plddt(out, L),
        "bt_pae": aux(sp.BinderTargetPAE(), "bt_pae"),
        "tb_pae": aux(sp.TargetBinderPAE(), "tb_pae"),
        "ptm_energy": aux(sp.pTMEnergy(), "pTMEnergy"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--designs", type=Path, required=True)
    p.add_argument("--seq-col", default="seq")
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-msa", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--diag-n", type=int, default=100,
                   help="designs scored by replicates 1..n-1 (the diagonal)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=int, default=0,
                   help="0-based index of this shard. Splits the design set so "
                        "one model can run as an array instead of one long "
                        "job; merge the per-shard CSVs afterwards. design_idx "
                        "is assigned BEFORE sharding, so ids stay global and "
                        "the merged table is indistinguishable from one run.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total shards. Each pays its own model load and JIT "
                        "compile, so this trades throughput for wall time; "
                        "worth it when one job would near its walltime. Scale "
                        "--diag-n down by the same factor to keep the diagonal "
                        "the same size overall.")
    p.add_argument("--recycling", type=int, default=3,
                   help="TRUNK PASSES, normalized across backends -- OF3 and "
                        "ESMFold2 add one internally and are handed one less "
                        "to compensate (see RECYCLING_OFFSET). Not the raw "
                        "recycling_steps argument each wrapper takes.")
    p.add_argument("--sampling-steps", type=int,
                   default=DEFAULT_SAMPLING_STEPS,
                   help="diffusion steps for every non-AF2 backend. Pinned "
                        "rather than left to each model's own default, which "
                        "range from 2 (Protenix mini) to 25 (Boltz) and made "
                        "every earlier run incomparable. See the note above "
                        "DEFAULT_SAMPLING_STEPS.")
    p.add_argument("--featurization", choices=["binder", "target_only"],
                   default="binder",
                   help="binder: binder_features(L, [target]) -- one pack per "
                        "length, binder sidechain reference atoms stubbed to "
                        "poly-G. target_only: target_only_features([binder, "
                        "target]) -- real binder sidechains, rebuilt per "
                        "design. The ranking runs used target_only; matching "
                        "them raises boltz2~of3 agreement substantially.")
    p.add_argument("--clear-cache-every", type=int, default=0,
                   help="jax.clear_caches() every N designs (0 = never). Only "
                        "meaningful with --featurization target_only: real "
                        "sidechains make the atom count vary per sequence, so "
                        "each design can trigger its own JIT compile and the "
                        "kernels accumulate until the GPU runs out. Protenix "
                        "OOMed at design 53 of 416 without this.")
    p.add_argument("--af2-model-idx", type=int, default=None,
                   help="pin AF2's parameter set instead of sweeping it. Needed "
                        "to reproduce a label made elsewhere: af2_ranking.py "
                        "and rank_designs.py both hand AlphaFoldLoss a fixed "
                        "key=jax.random.key(42), which resolves to model_idx 4. "
                        "Requires --af2-replicate msa_seed, since pinning "
                        "removes the ensemble replicate axis.")
    p.add_argument("--num-samples", type=int, default=1,
                   help="fold each design N times and average the metrics, "
                        "matching build_multisample_loss(num_samples=N). "
                        "Degenerate for AF2, which is deterministic.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-col", default=None,
                   help="experimental readout, copied through to the output")
    p.add_argument("--af2-replicate", choices=["model_idx", "msa_seed"],
                   default="model_idx",
                   help="AF2's replicate axis. model_idx spans five SEPARATELY "
                        "TRAINED networks (ensemble spread); msa_seed resamples "
                        "MSA clusters within ONE network, which is the "
                        "like-for-like analogue of a diffusion reseed. Measured "
                        "on DIO3: 0.72 vs 0.87 rho on ipsae_min. Only bites "
                        "when the MSA is deeper than max_msa_clusters (512).")
    p.add_argument("--max-length-groups", type=int, default=None,
                   help="keep only the N most common binder lengths. Each "
                        "distinct length costs one JIT compile, which for the "
                        "slower backends outweighs the folds it enables.")
    args = p.parse_args()

    df = pd.read_csv(args.designs)
    df["mx_seq"] = df[args.seq_col].astype(str)
    df["mx_label"] = df[args.label_col] if args.label_col else np.nan
    df["mx_len"] = df.mx_seq.str.len()
    df = df.reset_index(drop=True)
    df["mx_idx"] = df.index          # stable id, independent of grouping order

    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit(f"--shard must be in [0, {args.num_shards})")
        # Strided, so no shard inherits all the slow designs. For a set with
        # many binder lengths this spreads lengths across shards and each pays
        # their compiles; a contiguous split of a length-sorted frame would be
        # cheaper there.
        df = df.iloc[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(df)} designs",
              flush=True)

    if args.max_length_groups:
        keep = df.mx_len.value_counts().nlargest(args.max_length_groups).index
        dropped = len(df) - int(df.mx_len.isin(keep).sum())
        df = df[df.mx_len.isin(keep)]
        # Never let a coverage cap pass silently -- a filtered set reads as a
        # complete one downstream.
        print(f"length cap: kept {df.mx_len.nunique()} lengths, "
              f"{len(df)} designs, dropped {dropped}", flush=True)
    if args.limit:
        df = df.head(args.limit)

    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if not l.startswith(">"))
    groups = sorted(df.groupby("mx_len"), key=lambda kv: -len(kv[1]))

    print(f"model={args.model} condition={args.condition} designs={len(df)} "
          f"length_groups={len(groups)} target_len={len(target)}", flush=True)

    model = load_model(args.model)
    msa_seed_axis = args.model == "af2" and args.af2_replicate == "msa_seed"
    if msa_seed_axis and args.condition == "monomer":
        # No MSA exists for a lone binder, so there are no clusters to resample
        # and AF2 is bit-deterministic: this axis would report a diagonal of
        # exactly 1.0, which is true but vacuous. Say so rather than emit it.
        print("WARNING: msa_seed axis is degenerate for the monomer condition "
              "(no MSA to resample); AF2 is deterministic here.", flush=True)
    if msa_seed_axis:
        print("AF2 replicate axis = msa_seed (model_idx pinned to 0)", flush=True)

    n_samples = args.num_samples
    if args.model == "af2" and n_samples > 1:
        # deterministic=True and eval_dropout=False (af2.py:116): every sample
        # would be bit-identical. Averaging them is pure cost.
        print(f"NOTE: AF2 is deterministic; --num-samples {n_samples} would "
              f"fold the same structure {n_samples}x. Forcing 1.", flush=True)
        n_samples = 1
    # Print the EFFECTIVE protocol, not the requested one. Two runs are only
    # comparable if these lines match, and the raw argument is not enough to
    # tell -- --recycling 3 means a different number of trunk passes to
    # different backends, so the resolved value is what goes in the log.
    raw_recycling = args.recycling + RECYCLING_OFFSET.get(args.model, 0)
    print(
        f"PROTOCOL model={args.model} condition={args.condition} "
        f"featurization={args.featurization} num_samples={n_samples} "
        f"trunk_passes={args.recycling} (recycling_steps={raw_recycling}) "
        f"sampling_steps="
        f"{'n/a' if args.model == 'af2' else args.sampling_steps} "
        f"target_msa={args.target_msa} seed={args.seed}",
        flush=True,
    )
    if raw_recycling < 1:
        raise SystemExit(
            f"--recycling {args.recycling} leaves {args.model} with "
            f"recycling_steps={raw_recycling}; it must run at least one pass."
        )
    if args.af2_model_idx is not None and args.model == "af2":
        if args.af2_replicate != "msa_seed":
            raise SystemExit(
                "--af2-model-idx pins the parameter set, which IS the "
                "model_idx replicate axis; the diagonal would be exactly 1.0. "
                "Pass --af2-replicate msa_seed.")
        print(f"AF2 parameter set pinned to model_idx={args.af2_model_idx}",
              flush=True)

    key = jax.random.key(args.seed)
    rows = []
    n_folds = 0
    n_designs_seen = 0
    t0 = time.time()

    for L, grp in groups:
        # Every length compiles its own kernel, and JAX preallocates a fixed
        # pool (mosaic.def sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.95), so the
        # pool fills monotonically as groups accumulate rather than by size.
        # Protenix on a 413 aa target died at group 13 of 20 without this --
        # having folded a longer binder successfully at group 7.
        jax.clear_caches()
        tf = time.time()
        shared = None
        if args.featurization == "binder":
            shared = build_features(
                model, L=L, seq=None, target=target,
                msa_path=args.target_msa, condition=args.condition,
                featurization="binder")
            print(f"  len {L}: {len(grp)} designs, featurized in "
                  f"{time.time() - tf:.1f}s", flush=True)
        else:
            print(f"  len {L}: {len(grp)} designs, features rebuilt per design",
                  flush=True)

        for rep in range(args.replicates):
            fmodel, feats = model, shared
            if msa_seed_axis and rep > 0:
                # msa_seed is consumed at featurization (af2.py:439), so the
                # pack is rebuilt per replicate. Shapes are unchanged, so this
                # costs a featurization but no recompile.
                fmodel = eqx.tree_at(lambda m: m.msa_seed, model, rep)
                if shared is not None:
                    feats = build_features(
                        fmodel, L=L, seq=None, target=target,
                        msa_path=args.target_msa, condition=args.condition,
                        featurization="binder")
            # The diagonal is a property of the model, not of the design set, so
            # replicates run on a slice rather than the whole group.
            sub = grp if rep == 0 else grp.head(
                max(1, round(args.diag_n * len(grp) / len(df))))
            for r in sub.itertuples():
                n_designs_seen += 1
                if (args.clear_cache_every
                        and n_designs_seen % args.clear_cache_every == 0):
                    jax.clear_caches()
                pssm = jax.nn.one_hot(
                    jnp.array([TOKENS.index(a) for a in r.mx_seq]), 20)
                try:
                    f = feats if feats is not None else build_features(
                        fmodel, L=L, seq=r.mx_seq, target=target,
                        msa_path=args.target_msa, condition=args.condition,
                        featurization="target_only")
                    per_sample = []
                    for smp in range(n_samples):
                        out = fold(args.model, model, pssm, f,
                                   replicate=0 if msa_seed_axis else rep,
                                   sample=smp, key=key,
                                   recycling=args.recycling,
                                   sampling=args.sampling_steps,
                                   af2_model_idx=args.af2_model_idx)
                        per_sample.append(metrics(pssm, out,
                                                  condition=args.condition,
                                                  L=int(L), key=key))
                    m = mean_metrics(per_sample)
                except Exception as e:
                    # One bad design must not lose the rest. Recorded as absent
                    # so the gap is visible rather than silently missing.
                    print(f"  design {r.mx_idx} rep {rep} FAILED: "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                    m = {}
                n_folds += n_samples
                for name, v in m.items():
                    rows.append({"design_idx": r.mx_idx, "seq": r.mx_seq,
                                 "label": r.mx_label, "binder_len": int(L),
                                 "model": args.model,
                                 "condition": args.condition, "replicate": rep,
                                 "metric": name, "value": v})
                if n_folds % FLUSH_EVERY == 0:
                    pd.DataFrame(rows).to_csv(args.out, index=False)
                    el = time.time() - t0
                    print(f"    {n_folds} folds  {el / n_folds:.2f}s/fold  "
                          f"elapsed {el / 60:.1f}m", flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(rows)} rows, {n_folds} folds, "
          f"{(time.time() - t0) / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
