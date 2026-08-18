"""Hallucinate DIO3 binders: one array task designs sequences in a loop.

Run inside the mosaic container, one GPU per task:

    singularity/mosaic-exec.sh python hallucinate/hallucinate.py \
        --array_id 0 --max_runtime 7.5 --save_dir <dir>

Each task writes its own designs_<array_id>.txt, so N array tasks never contend
for the same file. Sequences are appended as they finish, which means a task
killed by the walltime still leaves everything it completed.

The models and the design loss are built ONCE per process, not once per
sequence. Boltz2() loads a 2.3 GB torch checkpoint and converts it to Equinox,
which costs minutes; doing that inside the design loop would dominate the run.
Only the parts that genuinely depend on the designed sequence — the re-fold and
the ranking prediction — are rebuilt per design.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.models.boltz2 import Boltz2
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain

# define constants:
TARGET_SEQUENCE = "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQRLVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERLYVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
BINDER_LENGHT = 80
MSA_PATH = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/msa_out/sequences/dio3_cut/msa/DIO3.a3m"


def evaluate_loss(loss, pssm, key):
    return loss(pssm, key=key)


def build():
    """Load the models and build the design objective. Once per process."""
    folder = Boltz2()
    mpnn = load_mpnn_sol(0.05)

    # -inf for Cys in the binder for MPNN. NoCys below already removes cysteine
    # from the optimizer's alphabet; this stops MPNN proposing it either.
    bias = jnp.zeros((BINDER_LENGHT, 20)).at[:, TOKENS.index("C")].set(-1e6)

    # define the loss function to optimize
    sp_loss = (
        sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001), bias=bias)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )

    # define features -> structure model
    features, _ = folder.binder_features(
        binder_length=BINDER_LENGHT,
        chains=[TargetChain(sequence=TARGET_SEQUENCE, use_msa=True, msa_path=MSA_PATH)],
    )

    # loss object: apply loss fn to features
    loss = NoCys(
        folder.build_multisample_loss(
            loss=sp_loss,
            features=features,
            recycling_steps=1,
            num_samples=4,
        )
    )
    return folder, loss


def design(folder, loss, seed):
    # NoCys makes the optimizer's alphabet 19 tokens, not 20 — cysteine is
    # spliced back in with zero probability by NoCys.sequence below.
    _pssm = np.random.uniform(low=0.25, high=0.75) * jax.random.gumbel(
        key=jax.random.key(seed),
        shape=(BINDER_LENGHT, 19),
    )

    _, pssm = simplex_APGM(
        loss_function=loss,
        x=jax.nn.softmax(_pssm),
        stepsize=0.2 * np.sqrt(BINDER_LENGHT),
        n_steps=100,
        momentum=0.3,
        scale=1.00,
        logspace=False,
        max_gradient_norm=1.0,
    )

    # sharp the PSSM into discrete sequence -> more optim
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        stepsize=0.5 * np.sqrt(BINDER_LENGHT),
        n_steps=50,
        momentum=0.0,
        scale=1.25,
        logspace=True,
        max_gradient_norm=1.0,
    )
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        n_steps=15,
        stepsize=0.5 * np.sqrt(BINDER_LENGHT),
        momentum=0.0,
        scale=1.4,
        logspace=True,
        max_gradient_norm=1.0,
    )

    pssm = NoCys.sequence(pssm)
    seq = pssm.argmax(-1)

    seq_str = "".join(TOKENS[i] for i in seq)
    boltz_features, _ = folder.target_only_features(
        chains=[
            TargetChain(sequence=seq_str, use_msa=False),
            TargetChain(sequence=TARGET_SEQUENCE, use_msa=True, msa_path=MSA_PATH),
        ]
    )
    ranking_loss = folder.build_multisample_loss(
        loss=1.00 * sp.IPTMLoss()
        + 0.5 * sp.TargetBinderIPSAE()
        + 0.5 * sp.BinderTargetIPSAE(),
        features=boltz_features,
        recycling_steps=3,
        num_samples=6,
    )

    loss_value, _ = evaluate_loss(
        ranking_loss, jax.nn.one_hot(seq, 20), key=jax.random.key(0)
    )
    return (seq_str, loss_value.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array_id", type=int, default=0,
                        help="SLURM array task id; names the output file and "
                             "seeds the RNG so tasks explore different inits")
    parser.add_argument("--max_runtime", type=float, default=7.5,
                        help="hours to keep designing; set below the job "
                             "walltime so the last design finishes")
    parser.add_argument("--n_designs", type=int, default=0,
                        help="stop after this many designs (0 = time-bounded)")
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    array_id = args.array_id

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = f"{args.save_dir}/designs_{array_id}.txt"

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}")
    t_build = time.time()
    folder, loss = build()
    print(f"models + loss built in {time.time() - t_build:.1f}s", flush=True)

    # start the iteration:
    start_time = time.time()
    results = []
    max_runtime_sec = args.max_runtime * 60 * 60
    print(f"Starting Design {start_time} -> {out_path}", flush=True)
    while time.time() - start_time < max_runtime_sec:
        if args.n_designs and len(results) >= args.n_designs:
            break
        # Distinct per (task, design) so no two trajectories share an init.
        seed = array_id * 100_000 + len(results)
        t0 = time.time()
        seq, loss_value = design(folder, loss, seed)
        with open(out_path, "a") as f:
            f.write(f">{loss_value:.4f}\n{seq}\n")
        results.append((seq, loss_value))
        print(f"[{len(results)}] loss={loss_value:.4f} "
              f"({time.time() - t0:.0f}s)  {seq}", flush=True)

    print(f"done: {len(results)} designs in "
          f"{(time.time() - start_time) / 60:.1f} min -> {out_path}")
