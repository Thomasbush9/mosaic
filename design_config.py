"""Campaign configuration: the catalog, and how to turn a config into a loss.

This is deliberately importable **without mosaic installed** — the webapp runs
host-side in a plain Python environment and needs the catalogs to build its UI,
while ``run_design.py`` runs inside the container and needs the builders. Every
mosaic import therefore lives inside a function.

A config is a plain dict (JSON on disk). It is both the input to a run and the
record of what that run did, which flags alone were not — the previous flag
interface could not express per-model or per-loss parameters at all, and passing
them through SLURM's ``--export`` is impossible because it splits on commas.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Catalog: structure models
#
# `params` are what build_loss accepts for that backend. They differ per model
# and the differences are real, not oversights:
#   - af2 asserts sampling_steps is None (models/af2.py:403)
#   - of3 defaults recycling_steps to 3, everything else to 1
#   - only of3/protenix expose num_samples through their multisample builders
# --------------------------------------------------------------------------

STRUCTURE_MODELS: dict[str, dict[str, Any]] = {
    "boltz2": {
        "label": "Boltz-2",
        "blurb": "Current-generation co-folding. Best all-round default.",
        "cost5": 91.0,
        "accepts_msa": True,
        "params": {
            "recycling_steps": {"type": "int", "default": 1, "min": 1, "max": 10},
            "sampling_steps": {"type": "int", "default": 25, "min": 5, "max": 200},
            "num_samples": {"type": "int", "default": 1, "min": 1, "max": 8},
            "deterministic": {"type": "bool", "default": True},
        },
    },
    "boltz1": {
        "label": "Boltz-1",
        "blurb": "Previous generation. Slowest; keep for comparison.",
        "cost5": 123.8,
        "accepts_msa": True,
        "params": {
            "recycling_steps": {"type": "int", "default": 1, "min": 1, "max": 10},
            "sampling_steps": {"type": "int", "default": 25, "min": 5, "max": 200},
            "num_samples": {"type": "int", "default": 1, "min": 1, "max": 8},
            "deterministic": {"type": "bool", "default": True},
        },
    },
    "af2": {
        "label": "AlphaFold2 multimer",
        "blurb": "Independent lineage — good for consensus. Never receives an "
                 "MSA: the wrapper rejects them (models/af2.py:391).",
        "cost5": 109.2,
        "accepts_msa": False,
        "params": {
            "recycling_steps": {"type": "int", "default": 1, "min": 1, "max": 10},
            "use_dropout": {"type": "bool", "default": False},
        },
    },
    "of3": {
        "label": "OpenFold3",
        "blurb": "AlphaFold3-lineage open implementation.",
        "cost5": 92.4,
        "accepts_msa": True,
        "params": {
            "recycling_steps": {"type": "int", "default": 3, "min": 1, "max": 10},
            "sampling_steps": {"type": "int", "default": 20, "min": 5, "max": 200},
            "num_samples": {"type": "int", "default": 1, "min": 1, "max": 8},
        },
    },
    "protenix": {
        "label": "Protenix",
        "blurb": "Fastest by ~3x. Use while tuning, confirm with Boltz-2.",
        "cost5": 39.4,
        "accepts_msa": True,
        "params": {
            "variant": {"type": "choice", "default": "mini",
                        "options": ["mini", "tiny", "base", "2025", "v2"]},
            "recycling_steps": {"type": "int", "default": 1, "min": 1, "max": 10},
            "sampling_steps": {"type": "int", "default": 20, "min": 5, "max": 200},
            "num_samples": {"type": "int", "default": 1, "min": 1, "max": 8},
        },
    },
}

# --------------------------------------------------------------------------
# Catalog: loss terms
#
# `confidence: True` marks terms that read the structure / confidence modules.
# Under JIT, JAX prunes those modules when nothing reads them — which is why a
# distogram-only objective is fast. Enabling any confidence term forces them to
# run and is materially slower. The UI must say so at the point of choice.
# --------------------------------------------------------------------------

LOSS_TERMS: dict[str, dict[str, Any]] = {
    "BinderTargetContact": {
        "group": "Contact", "default_weight": 4.0, "confidence": False,
        "blurb": "Pull the binder into contact with the target.",
        "params": {
            "contact_distance": {"type": "float", "default": 20.0, "min": 4.0, "max": 40.0},
            "paratope_size": {"type": "optint", "default": None, "min": 1, "max": 200},
            "paratope_idx": {"type": "idxlist", "default": None,
                             "help": "binder residue indices allowed to contact"},
            "epitope_idx": {"type": "idxlist", "default": None,
                            "help": "target residue indices to aim at"},
        },
    },
    "WithinBinderContact": {
        "group": "Contact", "default_weight": 1.0, "confidence": False,
        "blurb": "Keep the binder folded rather than sprawling.",
        "params": {
            "max_contact_distance": {"type": "float", "default": 14.0, "min": 4.0, "max": 40.0},
            "min_sequence_separation": {"type": "int", "default": 8, "min": 0, "max": 50},
            "num_contacts_per_residue": {"type": "int", "default": 25, "min": 1, "max": 100},
        },
    },
    "HelixLoss": {
        "group": "Shape", "default_weight": 1.0, "confidence": False,
        "blurb": "Bias toward helical secondary structure.",
        "params": {
            "max_distance": {"type": "float", "default": 6.0, "min": 3.0, "max": 15.0},
            "target_value": {"type": "float", "default": -2.0, "min": -10.0, "max": 10.0},
        },
    },
    "DistogramRadiusOfGyration": {
        "group": "Shape", "default_weight": 1.0, "confidence": False,
        "blurb": "Compactness, read off the distogram (cheap).",
        "params": {"target_radius": {"type": "optfloat", "default": None,
                                     "min": 5.0, "max": 60.0}},
    },
    "MAERadiusOfGyration": {
        "group": "Shape", "default_weight": 1.0, "confidence": True,
        "blurb": "Compactness from actual coordinates (needs the structure module).",
        "params": {"target_radius": {"type": "optfloat", "default": None,
                                     "min": 5.0, "max": 60.0}},
    },
    "DistogramIPTMProxy": {
        "group": "Interface", "default_weight": 1.0, "confidence": False,
        "blurb": "ipTM-like signal from the distogram — cheap approximation.",
        "params": {"contact_distance": {"type": "float", "default": 8.0,
                                        "min": 4.0, "max": 30.0}},
    },
    "PLDDTLoss": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                  "blurb": "Maximise binder pLDDT.", "params": {}},
    "WithinBinderPAE": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                        "blurb": "Minimise intra-binder PAE.", "params": {}},
    "BinderTargetPAE": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                        "blurb": "Minimise binder->target PAE.", "params": {}},
    "TargetBinderPAE": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                        "blurb": "Minimise target->binder PAE.", "params": {}},
    "IPTMLoss": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                 "blurb": "Maximise interface pTM.", "params": {}},
    "BinderTargetIPTM": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                         "blurb": "Directional interface pTM.", "params": {}},
    "BinderPTMLoss": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                      "blurb": "Maximise binder pTM.", "params": {}},
    "BinderTargetIPSAE": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                          "blurb": "ipSAE, binder->target.", "params": {}},
    "TargetBinderIPSAE": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                          "blurb": "ipSAE, target->binder.", "params": {}},
    "IPSAE_min": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                  "blurb": "Minimum of the two ipSAE directions.", "params": {}},
    "pTMEnergy": {"group": "Confidence", "default_weight": 1.0, "confidence": True,
                  "blurb": "pTM-derived energy.", "params": {}},
}

# ProteinMPNN is inverse folding: given a backbone, how likely is this sequence?
# It is a *structure-conditioned sequence* critic, so it needs a structure — it
# reads the backbone the structure model just predicted, which is why it lives
# with the structure losses rather than the sequence ones.
MPNN_TERMS: dict[str, dict[str, Any]] = {
    "InverseFoldingSequenceRecovery": {
        "default_weight": 1.0,
        "blurb": "Would ProteinMPNN redesign this backbone to this sequence? "
                 "The standard de novo filter — sequences that fail it rarely "
                 "express.",
        "params": {
            "temp": {"type": "float", "default": 0.1, "min": 0.01, "max": 2.0},
            "num_samples": {"type": "int", "default": 16, "min": 1, "max": 64},
            "jacobi_iterations": {"type": "int", "default": 10, "min": 1, "max": 50},
        },
    },
    "ProteinMPNNLoss": {
        "default_weight": 1.0,
        "blurb": "Direct MPNN likelihood of the sequence given the backbone.",
        "params": {
            "num_samples": {"type": "int", "default": 8, "min": 1, "max": 64},
            "stop_grad": {"type": "bool", "default": True},
        },
    },
    "AllResiduePLLLoss": {
        "default_weight": 1.0,
        "blurb": "Per-residue pseudo-likelihood over the whole binder.",
        "params": {
            "chunk_size": {"type": "int", "default": 10, "min": 1, "max": 64,
                           "help": "larger is faster and uses more memory"},
        },
    },
}

MPNN_WEIGHTS = {
    "vanilla": "v_48_020.pt (general)",
    "soluble": "soluble_v_48_020.pt (biased to soluble designs)",
    "abmpnn": "abmpnn.pt (antibody-tuned)",
}

SEQUENCE_MODELS: dict[str, dict[str, Any]] = {
    "esmc": {
        "label": "ESM-C pseudo-likelihood",
        "blurb": "Is this a plausible protein sequence? Clipped by default: raw "
                 "PLLs over-optimize to homopolymers.",
        "antibody_only": False,
        "params": {
            "checkpoint": {"type": "choice", "default": "biohub/ESMC-300M",
                           "options": ["biohub/ESMC-300M", "biohub/ESMC-600M",
                                       "biohub/ESMC-6B"]},
            "clip_lower": {"type": "float", "default": 2.0, "min": 0.0, "max": 50.0},
            "clip_upper": {"type": "float", "default": 100.0, "min": 1.0, "max": 1000.0},
        },
    },
    "ablang": {
        "label": "AbLang (heavy/light)", "antibody_only": True,
        "blurb": "Antibody-specific. Only meaningful on VH/VL sequences.",
        "params": {"chain": {"type": "choice", "default": "heavy",
                             "options": ["heavy", "light"]}},
    },
    "ablang2": {
        "label": "AbLang2 (paired)", "antibody_only": True,
        "blurb": "Antibody-specific, paired model.",
        "params": {"heavy_len": {"type": "optint", "default": None, "min": 0, "max": 400}},
    },
}

# Generative models. These implement no binder_features/build_loss — there is no
# question to differentiate — so they cannot appear in a campaign objective. They
# run BEFORE it, proposing starting points.
GENERATIVE_MODELS: dict[str, dict[str, Any]] = {
    "boltzgen": {
        "label": "BoltzGen",
        "blurb": "Diffuses binder backbones against a target STRUCTURE, then "
                 "reads residues off sidechain geometry. ~2.5 s per design "
                 "after a one-off ~50 s conditioning pass.",
        "needs_structure": True,
        "params": {
            "num_designs": {"type": "int", "default": 8, "min": 1, "max": 200},
            "binder_length": {"type": "int", "default": 80, "min": 20, "max": 200},
            "n_helices": {"type": "int", "default": 3, "min": 1, "max": 6,
                          "help": "three-helix bundle is the standard de novo "
                                  "mini-binder topology"},
            "loop_length": {"type": "int", "default": 4, "min": 2, "max": 12},
            "sampling_steps": {"type": "int", "default": 300, "min": 20, "max": 1000},
            "step_scale": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0},
            "noise_scale": {"type": "float", "default": 0.88, "min": 0.1, "max": 2.0},
            "recycling_steps": {"type": "int", "default": 3, "min": 1, "max": 10},
            "target_chain": {"type": "choice", "default": "A",
                             "options": list("ABCDEFGH")},
        },
    },
    "proteina": {
        "label": "Proteina",
        "blurb": "Flow-matching backbone generator. Staged and loads, but not "
                 "yet wired into a workflow here.",
        "needs_structure": False,
        "enabled": False,
        "params": {},
    },
}

OPTIMIZER_PARAMS: dict[str, dict[str, Any]] = {
    "n_steps": {"type": "int", "min": 1, "max": 500},
    "stepsize": {"type": "float", "min": 0.001, "max": 1.0},
    "momentum": {"type": "float", "min": 0.0, "max": 0.99},
    "scale": {"type": "float", "min": 0.1, "max": 10.0},
    "logspace": {"type": "bool"},
    "max_gradient_norm": {"type": "optfloat", "min": 0.01, "max": 1000.0},
}


def default_config() -> dict[str, Any]:
    """A working baseline: contact-driven, cheap, no confidence terms."""
    return {
        "target": {"fasta": None, "msa": None, "use_msa": True},
        "binder": {"length": 80, "no_cys": True, "init_fasta": None,
                   "init_noise": 0.15},
        "models": [{"name": "boltz2", "weight": 1.0,
                    "params": {"recycling_steps": 1, "sampling_steps": 25}}],
        "losses": [
            {"name": "BinderTargetContact", "weight": 4.0,
             "params": {"contact_distance": 20.0}},
            {"name": "WithinBinderContact", "weight": 1.0, "params": {}},
        ],
        "sequence_models": [{"name": "esmc", "weight": 0.5,
                             "params": {"checkpoint": "biohub/ESMC-300M",
                                        "clip_lower": 2.0, "clip_upper": 100.0}}],
        "optimizer": {
            "soft": {"n_steps": 100, "stepsize": 0.1, "momentum": 0.9,
                     "scale": 1.0, "logspace": False, "max_gradient_norm": None},
            "sharp": {"n_steps": 25, "stepsize": 0.025, "momentum": 0.5,
                      "scale": 2.0, "logspace": False, "max_gradient_norm": None},
        },
        "mpnn": {"weights": "soluble", "backbone_noise": 0.0, "terms": []},
        "run": {"batch": 2, "seed": 0},
        "cluster": {"account": "kempner_bsabatini_lab",
                    "partition": "kempner_h100", "n_tasks": 16,
                    "throttle": 0, "time_limit": "08:00:00",
                    "mem": "128G", "cpus": 8},
    }


def validate(cfg: dict[str, Any]) -> list[str]:
    """Return human-readable problems. Empty list means it will run."""
    errs: list[str] = []
    if not cfg.get("models"):
        errs.append("No structure model selected.")
    for m in cfg.get("models", []):
        if m["name"] not in STRUCTURE_MODELS:
            errs.append(f"Unknown model {m['name']!r}.")
        if m["name"] == "af2" and m.get("params", {}).get("sampling_steps"):
            errs.append("AF2 does not accept sampling_steps (it asserts None).")
    if not cfg.get("losses") and not cfg.get("mpnn", {}).get("terms"):
        errs.append("No loss terms selected — there is nothing to optimize.")
    for t in cfg.get("mpnn", {}).get("terms", []):
        if t["name"] not in MPNN_TERMS:
            errs.append(f"Unknown ProteinMPNN term {t['name']!r}.")
    for l in cfg.get("losses", []):
        if l["name"] not in LOSS_TERMS:
            errs.append(f"Unknown loss {l['name']!r}.")
    for stage in ("soft", "sharp"):
        st = cfg.get("optimizer", {}).get(stage, {})
        if st.get("n_steps", 0) < 1:
            errs.append(f"{stage} stage n_steps must be >= 1.")
    for m in cfg.get("models", []):
        if m.get("params", {}).get("recycling_steps", 1) < 1:
            errs.append(
                f"{m['name']}: recycling_steps must be >= 1. At 0 the trunk runs "
                "inside jax.lax.scan(length=0), so the body never executes and "
                "the loss stops depending on the sequence — a silently zero "
                "gradient that looks like a working run."
            )
    if cfg.get("binder", {}).get("length", 0) < 10:
        errs.append("Binder length must be at least 10.")
    return errs


def uses_confidence(cfg: dict[str, Any]) -> bool:
    """True if anything reads the structure/confidence modules.

    ProteinMPNN counts: it scores a sequence against predicted backbone
    coordinates, so it too prevents JAX from pruning the structure module.
    """
    if any(LOSS_TERMS.get(l["name"], {}).get("confidence")
           for l in cfg.get("losses", [])):
        return True
    return bool(cfg.get("mpnn", {}).get("terms"))


def estimate_seconds(cfg: dict[str, Any]) -> float:
    """Rough wall-clock per array task, from measured per-model costs.

    Measured at 5 soft steps / batch 2 / 237-residue target + 80-residue binder
    on an H100. Scales linearly in steps; confidence terms roughly triple it
    because they defeat JIT pruning of the structure and confidence modules.
    """
    per_step = sum(
        STRUCTURE_MODELS.get(m["name"], {}).get("cost5", 90.0) for m in cfg["models"]
    ) / 5.0
    steps = cfg["optimizer"]["soft"]["n_steps"] + cfg["optimizer"]["sharp"]["n_steps"]
    load = 60.0 * max(1, len(cfg["models"]))
    mult = 3.0 if uses_confidence(cfg) else 1.0
    return per_step * steps * mult + load


# --------------------------------------------------------------------------
# Builders — these DO import mosaic, so they only run inside the container.
# --------------------------------------------------------------------------

def build_structure_model(name: str, params: dict[str, Any]):
    if name == "boltz2":
        from mosaic.models.boltz2 import Boltz2
        return Boltz2()
    if name == "boltz1":
        from mosaic.models.boltz1 import Boltz1
        return Boltz1()
    if name == "af2":
        from mosaic.models.af2 import AlphaFold2
        return AlphaFold2(multimer=True)
    if name == "of3":
        from mosaic.models.of3 import OF3
        return OF3()
    if name == "protenix":
        import mosaic.models.protenix as P
        return {"mini": P.ProtenixMini, "tiny": P.ProtenixTiny, "base": P.ProtenixBase,
                "2025": P.Protenix2025, "v2": P.ProtenixV2}[
            params.get("variant", "mini")]()
    raise SystemExit(f"unknown structure model {name!r}")


def build_inner_loss(losses: list[dict[str, Any]], mpnn_cfg: dict | None = None):
    """Weighted sum of loss terms, evaluated against one model's output.

    ProteinMPNN terms belong here rather than with the sequence models: their
    __call__ signature is (sequence, output, key), i.e. they score the sequence
    *given the backbone the structure model just predicted*. That also means
    they read the structure module, so like the confidence terms they defeat
    JIT pruning and cost accordingly.
    """
    import mosaic.losses.structure_prediction as sp

    total = None
    for spec in losses:
        cls = getattr(sp, spec["name"])
        params = {k: v for k, v in (spec.get("params") or {}).items() if v is not None}
        term = spec["weight"] * cls(**params)
        total = term if total is None else total + term

    if mpnn_cfg and mpnn_cfg.get("terms"):
        import mosaic.losses.protein_mpnn as pm
        from mosaic.proteinmpnn.mpnn import load_abmpnn, load_mpnn, load_mpnn_sol

        loader = {"vanilla": load_mpnn, "soluble": load_mpnn_sol,
                  "abmpnn": load_abmpnn}[mpnn_cfg.get("weights", "soluble")]
        mpnn = loader(backbone_noise=mpnn_cfg.get("backbone_noise", 0.0))
        for spec in mpnn_cfg["terms"]:
            cls = getattr(pm, spec["name"])
            params = {k: v for k, v in (spec.get("params") or {}).items()
                      if v is not None}
            term = spec["weight"] * cls(mpnn, **params)
            total = term if total is None else total + term
    return total


def build_sequence_terms(specs: list[dict[str, Any]]):
    """Sequence-plausibility terms, applied to the soft sequence directly."""
    from mosaic.losses.transformations import ClippedLoss

    out = []
    for spec in specs:
        p = spec.get("params") or {}
        if spec["name"] == "esmc":
            from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
            # Lowercase repo id: losses/esmc.py's alias uses a capital B and the
            # shared HF cache is case-sensitive, so the alias re-downloads 1.3 GB.
            esm = load_esmc(p.get("checkpoint", "biohub/ESMC-300M"))
            term = ClippedLoss(ESMCPseudoLikelihood(esm),
                               p.get("clip_lower", 2.0), p.get("clip_upper", 100.0))
        elif spec["name"] == "ablang":
            from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
            m, tok = load_ablang(p.get("chain", "heavy"))
            term = AbLangPseudoLikelihood(m, tok)
        elif spec["name"] == "ablang2":
            from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
            m, tok = load_ablang2()
            term = Ablang2PseudoLikelihood(m, tok, p.get("heavy_len") or 0, None)
        else:
            raise SystemExit(f"unknown sequence model {spec['name']!r}")
        out.append(spec["weight"] * term)
    return out
