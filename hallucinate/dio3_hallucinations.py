import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo 
    import time
    import uuid 
    import jax 
    import jax.numpy as jnp 
    import equinox as eqx 
    import numpy as np 

    from mosaic.models.boltz2 import Boltz2
    import mosaic.losses.structure_prediction as sp 
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain
    from mosaic.optimizers import simplex_APGM
    import rich

    return (
        Boltz2,
        InverseFoldingSequenceRecovery,
        NoCys,
        TOKENS,
        TargetChain,
        jax,
        jnp,
        load_mpnn_sol,
        mo,
        np,
        simplex_APGM,
        sp,
        time,
        uuid,
    )


@app.cell
def _():
    TARGET_SEQUENCE = "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQRLVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERLYVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
    BINDER_LENGHT = 80
    MSA_PATH = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/msa_out/sequences/dio3_cut/msa/DIO3.a3m"
    return BINDER_LENGHT, MSA_PATH, TARGET_SEQUENCE


@app.cell
def _(uuid):
    worker_id = str(uuid.uuid4())[:8]
    return (worker_id,)


@app.cell
def _(Boltz2, load_mpnn_sol):
    folder = Boltz2()
    mpnn = load_mpnn_sol((0.05))
    return folder, mpnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Construction:

    Here we build the loss term that is optimized, here the loss is composed by:

    -
    """)
    return


@app.cell
def _():
    from rich import inspect 

    return


@app.cell
def _(BINDER_LENGHT, TOKENS, jnp):
    # -inf for Cys in the binder for MPNN
    bias = (
        jnp.zeros((BINDER_LENGHT, 20)).at[:BINDER_LENGHT, TOKENS.index("C")].set(-1e6)
    )
    return (bias,)


@app.cell
def _(InverseFoldingSequenceRecovery, bias, jnp, mpnn, sp):
    # cosntruct the loss function 
    sp_loss = (
    sp.BinderTargetContact() 
    + sp.WithinBinderContact()
    + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001), bias=bias)
    + 0.05 * sp.TargetBinderPAE()
    + 0.05 * sp.BinderTargetPAE()
    + 0.025 * sp.IPTMLoss() 
    + 0.4 * sp.WithinBinderPAE()
    + 0.025 * sp.pTMEnergy()
    + 0.1 * sp.PLDDTLoss())
    return (sp_loss,)


@app.cell
def _(BINDER_LENGHT, MSA_PATH, TARGET_SEQUENCE, TargetChain, folder):
    # use the folder to create the binder features -> hallucination into folded strcutrue 
    features, _  = boltz_features, boltz_writer = folder.binder_features(
        binder_length=BINDER_LENGHT,
        chains=[TargetChain(sequence=TARGET_SEQUENCE, use_msa=True, msa_path=MSA_PATH)],
    )
    return (features,)


@app.cell
def _(NoCys, features, folder, sp_loss):
    loss = NoCys(
        folder.build_multisample_loss(
            loss=sp_loss,
            features=features,
            recycling_steps=1,
            num_samples=4
        )
    )
    return (loss,)


@app.cell
def _(
    BINDER_LENGHT,
    MSA_PATH,
    NoCys,
    TARGET_SEQUENCE,
    TOKENS,
    TargetChain,
    folder,
    jax,
    jnp,
    loss,
    np,
    simplex_APGM,
    sp,
):
    def evaluate_loss(loss, pssm, key):
        return loss(pssm, key=key)

    def design():
        _pssm = np.random.uniform(low=0.25, high=0.75)* jax.random.gumbel(
            key=jax.random.key(np.random.randint(10000000)),
            shape=(BINDER_LENGHT,19)
        )

        _, pssm = simplex_APGM(
            loss_function=loss,
            x=jax.nn.softmax(_pssm),
            stepsize=0.2*np.sqrt(BINDER_LENGHT),
            n_steps=100,
            momentum=0.3,
            scale=1.00,
            logspace=False,
            max_gradient_norm=1.0,
        )

        #sharp the PSSM into discrete sequence -> more optim
        pssm, _ = simplex_APGM(
            loss_function=loss,
            x= jnp.log(pssm + 1e-5),
            stepsize=0.5*np.sqrt(BINDER_LENGHT),
            n_steps=50,
            momentum=0.0,
            scale=1.25,
            logspace=True,
            max_gradient_norm=1.0,
        )
        pssm, _ = simplex_APGM(
            loss_function=loss, 
            x = jnp.log(pssm+1e-5),
            n_steps=15,
            stepsize=0.5*np.sqrt(BINDER_LENGHT),
            momentum=0.0,
            scale=1.4,
            logspace=True, 
            max_gradient_norm=1.0
        )

        pssm = NoCys.sequence(pssm)
        seq = pssm.argmax(-1)

        seq_str = "".join(TOKENS[i] for i in seq)
        boltz_features, boltz_writer = folder.target_only_features(
            chains=[
                TargetChain(sequence=seq_str, use_msa=False),
                TargetChain(sequence=TARGET_SEQUENCE, use_msa=True, msa_path=MSA_PATH)
            ]
        )
        ranking_loss = folder.build_multisample_loss(
            loss= 1.00 * sp.IPTMLoss()
        + 0.5 * sp.TargetBinderIPSAE()
        + 0.5 * sp.BinderTargetIPSAE(),
        features=boltz_features,
        recycling_steps=3,
        num_samples=6)

        loss_value, _ = evaluate_loss(
            ranking_loss, jax.nn.one_hot(seq, 20), key=jax.random.key(0)
        )

        return (seq_str, loss_value.item())


    return (design,)


@app.cell
def _(design, time, worker_id):
    start_time = time.time()
    results = []
    max_runtime_sec = 2.0 * 60 * 60 
    while time.time() - start_time < max_runtime_sec:
        seq, loss_value = design()
        with open(f"/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/dio3_cut/designs/designs_{worker_id}.txt", "a") as f:
            f.write(f">{loss_value:4f}\n{seq}\n")
            results.append((seq, loss_value))
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
