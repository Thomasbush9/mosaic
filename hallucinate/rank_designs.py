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

    return Path, jax, jnp, mo, pd, plt, re


@app.cell
def _(Path):
    TARGET_SEQUENCE = "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQRLVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERLYVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
    MSA_PATH = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/msa_out/sequences/dio3_cut/msa/DIO3.a3m"
    DESIGN_PATH=Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/designs")
    return DESIGN_PATH, MSA_PATH, TARGET_SEQUENCE


@app.cell
def _(DESIGN_PATH, pd, re):
    # now we create a df for each design vs. custom loss function: 
    d0 = DESIGN_PATH / "designs_0.txt"
    designs = {}
    i = 0
    for d in DESIGN_PATH.iterdir():
        with open(d, "r") as f:
            data =[re.sub(r'[>\n]', '', text) for text in f.readlines()]
            loss = data[::2]
            sequences = data[1::2]
            # filters already not unique sequences
            for l, seq in zip(loss, sequences):
                designs[i] = {"seq":seq, "loss":float(l)}
                i += 1
    designs = pd.DataFrame.from_dict(designs, orient="index")
    
    return (designs,)


@app.cell
def _(designs):
    designs
    return


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
    import rich

    return (
        AlphaFold2,
        ESMCPseudoLikelihood,
        ESMCPseudoPerplexity,
        TOKENS,
        load_esmc,
        sp,
    )


@app.cell
def _(jax):
    SEED=42
    key = jax.random.key(SEED)
    return (key,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute ESMC PseudoLikelihodd Loss

    For each sequence we can rank them using the ESMC loss (which has not been used during optimizaiton)
    """)
    return


@app.cell
def _(load_esmc):
    # start with sequence loss
    esmc = load_esmc("biohub/ESMC-600M")
    return (esmc,)


@app.cell
def _(ESMCPseudoLikelihood, ESMCPseudoPerplexity, esmc):
    # now we can use that for our loss: 
    loss_sequences = ESMCPseudoLikelihood(esmc)
    perplexity = ESMCPseudoPerplexity(esmc)
    return loss_sequences, perplexity


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
def _(designs, key, loss_sequences, one_hot_seq, perplexity):
    # compute the loss for all the sequences
    from rich.progress import track

    losses = []
    for n in track(range(one_hot_seq.shape[0]), description="ESM-C Loss"):
        loss_esmc = loss_sequences(one_hot_seq[n], key=key) + perplexity(one_hot_seq[n], key=key)
        losses.append(float(loss_esmc[0]))

    # Assign the entire list to the column instantly
    designs["esmc_loss"] = losses
    return


@app.cell
def _(designs, pd, plt):
    k = 10
    best  = designs.nlargest(k, "esmc_loss")    # highest esmc_loss
    worst = designs.nsmallest(k, "esmc_loss")   # lowest esmc_loss

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(designs["loss"], designs["esmc_loss"],
               s=18, c="0.8", edgecolor="none", label=f"all designs (n={len(designs)})")
    ax.scatter(best["loss"], best["esmc_loss"],
               s=50, c="tab:green", edgecolor="k", linewidth=0.4, label=f"top {k} esmc_loss")
    ax.scatter(worst["loss"], worst["esmc_loss"],
               s=50, c="tab:red", edgecolor="k", linewidth=0.4, label=f"bottom {k} esmc_loss")

    for _, r in pd.concat([best, worst]).iterrows():
        ax.annotate(str(r.name), (r["loss"], r["esmc_loss"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("design loss")
    ax.set_ylabel("ESM-C loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.show()
    return (k,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OF3 based ranking:

    Now we can compute the ranking using a different folding model (OF3)
    """)
    return


@app.cell
def _():
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.of3 import OF3

    return OF3, TargetChain


@app.cell
def _(OF3, one_hot_seq):
    # we can compute a double loss: af2 binder features + combined features
    of3 = OF3()
    binder_length = one_hot_seq.shape[1]
    return (of3,)


@app.cell
def _(MSA_PATH, TARGET_SEQUENCE, TargetChain, key, of3, sp):
    # define the loss function-> we are going to keep the simple loss funciton 
    #that we used for the hallucination optimization: 
    def evaluate_loss(loss, pssm, key):
        return loss(pssm, key=key)

    def rank_loss(seq_str:str,onehot):
        features, writer = of3.target_only_features(
            chains=[
                TargetChain(sequence=seq_str, use_msa=False),
                TargetChain(TARGET_SEQUENCE, use_msa=True, msa_path=MSA_PATH)
            ],
        )
        ranking_loss = of3.build_multisample_loss(
            loss = 1.00 * sp.IPTMLoss()
            + 0.5 * sp.TargetBinderIPSAE()
            + 0.5 * sp.BinderTargetIPSAE(),
            features = features, 
            recycling_steps=3,
            num_samples=6
        ) 
        loss_value, _ = evaluate_loss(ranking_loss, onehot, key)
        return (seq_str, loss_value.item())

    return evaluate_loss, rank_loss


@app.cell
def _(designs, mo, one_hot_seq, rank_loss):

    rows = list(designs.iterrows())
    of3_losses = []

    with mo.status.progress_bar(total=len(rows), title="scoring designs") as bar:
        for p, (idx, row) in enumerate(rows):
            try:
                of3_losses.append(float(rank_loss(row["seq"], one_hot_seq[p])[1]))
            except Exception as e:
                print(f"row {idx} failed: {type(e).__name__}: {e}")
                of3_losses.append(float("nan"))
            bar.update()

    designs["of3_loss"] = of3_losses
    return


@app.cell
def _(designs, k, pd, plt):
    # now plot the best of3 scores
    top_g = 10
    best_of  = designs.nlargest(k, "of3_loss")    # highest esmc_loss
    worst_of = designs.nsmallest(k, "of3_loss")   # lowest esmc_loss

    fix, axs = plt.subplots(figsize=(6, 5))
    axs.scatter(designs["loss"], designs["of3_loss"],
               s=18, c="0.8", edgecolor="none", label=f"all designs (n={len(designs)})")
    axs.scatter(best_of["loss"], best_of["of3_loss"],
               s=30, c="tab:red", edgecolor="k", linewidth=0.4, label=f"worst {top_g} of3_loss")
    axs.scatter(worst_of["loss"], worst_of["of3_loss"],
               s=30, c="tab:green", edgecolor="k", linewidth=0.4, label=f"best {top_g} of3_loss")

    for _, v in pd.concat([best_of, worst_of]).iterrows():
        axs.annotate(str(v.name), (v["loss"], v["of3_loss"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    axs.set_xlabel("design loss")
    axs.set_ylabel("OF3-Loss")
    axs.legend(frameon=False, fontsize=8)
    fix.tight_layout()
    plt.show()
    return


@app.cell
def _(DESIGN_PATH, designs):
    designs.to_csv(DESIGN_PATH/"correctly_ranked_candidates.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AF2 Based Ranking
    """)
    return


@app.cell
def _(AlphaFold2, one_hot_seq):
    af2 = AlphaFold2()
    binder_length = one_hot_seq.shape[1]
    return (af2,)


@app.cell
def _(MSA_PATH, TARGET_SEQUENCE, TargetChain, af2, evaluate_loss, key, sp):
    # define the loss function-> we are going to keep the simple loss funciton 
    #that we used for the hallucination optimization: 
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
def _(designs, one_hot_seq, rank_loss_af2):
    # write simple loss for af2: 
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )

    af2_losses = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("scoring designs", total=len(designs))

        for p, (idx, row) in enumerate(designs.iterrows()):
            try:
                af2_losses.append(float(rank_loss_af2(row["seq"], one_hot_seq[p])[1]))
            except Exception as e:
                progress.console.print(f"[red]row {idx} failed:[/red] {type(e).__name__}: {e}")
                af2_losses.append(float("nan"))
            progress.advance(task)

    designs["af2_losses"] = af2_losses
    return


@app.cell
def _(designs):
    designs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the best vs. worst binders
    """)
    return


@app.cell
def _(DESIGN_PATH, designs):
    designs.to_csv(DESIGN_PATH/"ranked_designed.csv")
    return


@app.cell
def _(designs, plt):
    from matplotlib.colors import PowerNorm

    _fig, _ax = plt.subplots(figsize=(7, 4.2), layout='constrained')

    _sc = _ax.scatter(
        designs['esmc_loss'],
        designs['of3_loss'],
        c=designs['loss'],
        cmap='turbo',
        norm=PowerNorm(gamma=0.4, vmin=designs['loss'].min(), vmax=designs['loss'].max()),
        s=45,
        alpha=0.85,
        linewidths=0,
    )

    _ax.set_xlabel('esmc_loss')
    _ax.set_ylabel('of3_loss')
    _ax.grid(True, alpha=0.3)
    _ax.set_axisbelow(True)

    _cb = _fig.colorbar(_sc, ax=_ax, label='loss')

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
