import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo 
    import jax.numpy as jnp
    import numpy as numpy
    import jax 
    from rich import inspect
    from pathlib import Path 
    import matplotlib.pyplot as plt 
    import pandas as pd
    import re

    return Path, jax, jnp, mo, pd, plt


@app.cell
def _(Path):
    TARGET_SEQUENCE = "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQRLVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERLYVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
    MSA_PATH = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup/dio3_cut/msa_out/sequences/dio3_cut/msa/DIO3.a3m"
    DESIGN_PATH=Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup/dio3_cut/designs")
    return DESIGN_PATH, MSA_PATH, TARGET_SEQUENCE


@app.cell
def _():
    # now we can compute a new loss function using af2 + esm-c scores: 
    import mosaic.losses.structure_prediction as sp 
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.models.af2 import AlphaFold2
    from mosaic.losses.esmc import ESMCPseudoLikelihood, ESMCPseudoPerplexity
    from mosaic.models.esmfold2 import ESMC
    from mosaic.losses.esmc import load_esmc
    from mosaic.structure_prediction import TargetChain
    import rich

    return AlphaFold2, TOKENS, TargetChain, load_mpnn_sol, sp


@app.cell
def _(DESIGN_PATH, jax, pd):
    designs = pd.read_csv(DESIGN_PATH/"correctly_ranked_candidates.csv")
    SEED=42
    key= jax.random.key(SEED)
    return designs, key


@app.cell
def _(AlphaFold2):
    af2 = AlphaFold2()
    binder_length = 80
    return (af2,)


@app.cell
def _(jax, jnp, pd):
    def encode_designs_simple(df: pd.DataFrame, tokens_list: list) -> jnp.ndarray:
        """
        Replaces 'U' with 'C' in sequences, maps them to integers based on 
        the provided tokens_list, and returns a one-hot encoded JAX array.
        """
        # 1. Map known tokens to integer indices (0 to N-1)
        token_to_idx = {token: idx for idx, token in enumerate(tokens_list)}
        num_classes = len(tokens_list)

        int_sequences = []
        for seq in df["seq"]:
            # Substitute 'U' with 'C' on the fly
            cleaned_seq = seq.replace("U", "C")

            # Convert string characters to integer indices
            int_seq = [token_to_idx[char] for char in cleaned_seq]
            int_sequences.append(int_seq)

        # 2. Convert to JAX array and apply one-hot encoding
        # Output shape: (num_sequences, sequence_length, num_classes)
        jax_matrix = jnp.array(int_sequences)
        return jax.nn.one_hot(jax_matrix, num_classes=num_classes)

    return (encode_designs_simple,)


@app.cell
def _(TOKENS, designs, encode_designs_simple):
    one_hot_seq = encode_designs_simple(designs, TOKENS)
    return (one_hot_seq,)


@app.cell
def _(MSA_PATH, TARGET_SEQUENCE, TargetChain, af2, key, sp):
    # define ranking func

    def evaluate_loss(loss, pssm, key):
        return loss(pssm, key=key)
    def rank_loss_af2(seq_str:str,onehot):
        features, writer = af2.target_only_features(
            chains=[
                TargetChain(sequence=seq_str, use_msa=False),
                TargetChain(TARGET_SEQUENCE, use_msa=True, msa_path=MSA_PATH)
            ],
        )
        ranking_loss = af2.build_loss(
            loss = 1.00 * sp.IPTMLoss()
            + 0.5 * sp.TargetBinderIPSAE()
            + 0.5 * sp.BinderTargetIPSAE(),
            features = features, 
            recycling_steps=3,
        ) 
        loss_value, _ = evaluate_loss(ranking_loss, onehot, key)
        return (seq_str, loss_value.item())

    return (rank_loss_af2,)


@app.cell
def _(DESIGN_PATH, designs, mo, one_hot_seq, rank_loss_af2):
    # rank everything: 
    rows = list(designs.iterrows())
    af2_losses = []

    with mo.status.progress_bar(total=len(rows), title="Scoring with AF2:") as bar:
        for p, (idx, row) in enumerate(rows):
            try:
                af2_losses.append(float(rank_loss_af2(row["seq"], one_hot_seq[p])[1]))
            except Exception as e:
                print(f"row {idx} failed: {type(e).__name__}: {e}")
                af2_losses.append(float("nan"))
            
    designs["af2_losses"] = af2_losses
    designs.to_csv(DESIGN_PATH/"correctly_ranked_candidates_af2.csv")
    return


@app.cell
def _(designs, pd, plt):
    # plot the losses:
    # now plot the best of3 scores
    top_g = 10
    best_of  = designs.nlargest(top_g, "af2_losses")    # highest esmc_loss
    worst_of = designs.nsmallest(top_g, "af2_losses")   # lowest esmc_loss

    fix, axs = plt.subplots(figsize=(6, 5))
    axs.scatter(designs["loss"], designs["af2_losses"],
               s=18, c="0.8", edgecolor="none", label=f"all designs (n={len(designs)})")
    axs.scatter(best_of["loss"], best_of["af2_losses"],
               s=30, c="tab:red", edgecolor="k", linewidth=0.4, label=f"worst {top_g} af2_losses")
    axs.scatter(worst_of["loss"], worst_of["af2_losses"],
               s=30, c="tab:green", edgecolor="k", linewidth=0.4, label=f"best {top_g} af2_losses")

    for _, v in pd.concat([best_of, worst_of]).iterrows():
        axs.annotate(str(v.name), (v["loss"], v["af2_losses"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    axs.set_xlabel("design loss")
    axs.set_ylabel("OF3-Loss")
    axs.legend(frameon=False, fontsize=8)
    fix.tight_layout()
    plt.show()
    return


@app.cell
def _(designs, plt):
    from matplotlib.colors import PowerNorm

    _fig, _ax = plt.subplots(figsize=(7, 4.2), layout='constrained')

    _sc = _ax.scatter(
        designs['af2_losses'],
        designs['of3_loss'],
        c=designs['loss'],
        cmap='turbo',
        norm=PowerNorm(gamma=0.4, vmin=designs['loss'].min(), vmax=designs['loss'].max()),
        s=45,
        alpha=0.85,
        linewidths=0,
    )

    _ax.set_xlabel('af2_losses')
    _ax.set_ylabel('of3_loss')
    _ax.grid(True, alpha=0.3)
    _ax.set_axisbelow(True)

    _cb = _fig.colorbar(_sc, ax=_ax, label='loss')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Self-consistency loss:
    """)
    return


@app.cell
def _(TOKENS, TargetChain, af2, jax, load_mpnn_sol):
    from mosaic.losses.protein_mpnn import inverse_fold
    import numpy as np
    from mosaic.util import calculate_rmsd

    mpnn = load_mpnn_sol((0.05))

    #sequence->different_folder-> mpnn-> folder-> sim structures
    #for simplicty we'll keep using af2 as it's already loaded (desing generated with Boltz2)
    def self_consistency_loss(seq_str, k, n=10, model_idx=0):
        k_fold, k_mpnn = jax.random.split(k, 2)

        features, _ = af2.target_only_features(
            chains=[TargetChain(sequence=seq_str, use_msa=False)],
        )

        ref_out = af2.model_output(
            PSSM=None, features=features, recycling_steps=3,
            model_idx=model_idx, key=k_fold,
        )

        seq_idx = jax.vmap(lambda ki: inverse_fold(
            mpnn, binder_length=len(seq_str), output=ref_out,
            temp=0.1, key=ki, jacobi_iterations=10,
        ))(jax.random.split(k_mpnn, n))          # Int[Array, "n L"]

        seq_strs = ["".join(TOKENS[i] for i in row) for row in np.asarray(seq_idx)]

        # now fold each sequence:
        CA = 1  # atom37 index; ATOM37_NAMES = ("N", "CA", "C", ...)

        # ... continuing from seq_idx ...
        ref_ca = ref_out.atom37_coords[:, CA]

        rmsds, plddts = [], []
        for ki, s in zip(jax.random.split(k_fold, n), seq_idx):
            out = af2.model_output(
                PSSM=jax.nn.one_hot(s, 20), features=features,
                recycling_steps=3, model_idx=model_idx, key=k_fold,
            )
            rmsds.append(float(calculate_rmsd(out.atom37_coords[:, CA], ref_ca)))
            plddts.append(float(out.plddt.mean()))

        rmsds, plddts = np.array(rmsds), np.array(plddts)
        best = rmsds.argmin()
        return {
            "sc_rmsd": rmsds[best],                    # the headline number
            "sc_rmsd_mean": rmsds.mean(),
            "sc_plddt": plddts[best],
            "frac_under_2A": float((rmsds < 2.0).mean()),
        }


    return (self_consistency_loss,)


@app.cell
def _(DESIGN_PATH, designs, key, mo, pd, self_consistency_loss):
    # now we add the self-consistency loss: 
    # rank everything: 
    rows_s = list(designs.iterrows())
    sc_records = []

    NAN_RECORD = {
        "sc_rmsd": float("nan"), "sc_rmsd_mean": float("nan"),
        "sc_plddt": float("nan"), "frac_under_2A": float("nan"), "best_seq": None,
    }

    with mo.status.progress_bar(total=len(rows_s), title="Self-consistency (AF2):") as bar_s:
        for id, rox in rows_s:
            try:
                sc_records.append(self_consistency_loss(rox["seq"], key))
            except Exception as e:
                print(f"row {id} failed: {type(e).__name__}: {e}")
                sc_records.append(dict(NAN_RECORD))
            bar_s.update()

    sc_df = pd.DataFrame(sc_records, index=designs.index)
    for col in sc_df.columns:
        designs[col] = sc_df[col]

    designs.to_csv(DESIGN_PATH / "correctly_ranked_candidates_af2_sc.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
