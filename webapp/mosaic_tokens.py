"""Alphabet constants, duplicated so the webapp needs no mosaic import.

Kept in sync by hand with:
  - src/mosaic/common.py       TOKENS
  - run_design.py              sanitize_target()'s analog map

Small enough to duplicate; importing mosaic host-side would drag in JAX and the
entire model stack just to validate a pasted sequence.
"""

TOKENS = "ARNDCQEGHILKMFPSTWYV"

# Residues with no column in TOKENS, mapped to their closest standard analog.
# U->C matters most in practice: selenocysteine is the selenium analog of
# cysteine and Sec->Cys mutants keep the fold while losing most catalytic
# activity — right for structure, wrong for any catalytic claim.
ANALOGS = {"U": "C", "O": "K", "B": "D", "Z": "E", "J": "L", "X": "A"}
