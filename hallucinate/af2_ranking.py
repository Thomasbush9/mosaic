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

    return Path, pd, re


@app.cell
def _(Path):
    TARGET_SEQUENCE = "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQRLVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERLYVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
    MSA_PATH = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/msa_out/sequences/dio3_cut/msa/DIO3.a3m"
    DESIGN_PATH=Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/designs")
    return (DESIGN_PATH,)


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

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
