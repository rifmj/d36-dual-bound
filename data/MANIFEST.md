# Manifest — "A dual linear programming bound for sphere packing in dimension 36"

Self-contained: the exact certificate data, the receipts, and the complete code needed to
reproduce every numerical claim in the paper are all present. No external files are required.

In this repository the files are split into two directories: the data and receipts described
below live in `data/` (this directory), and the Python code described further down lives in the
sibling `code/` directory. (In the arXiv ancillary bundle they are combined under one `anc/`
tree; the file descriptions are identical.)

## Data and receipts (`data/`)
- `certificate_exact_data.txt` — the exact rational certificate: the 20 free coordinates of the
  dual vertex (in the reduced basis constructed by `ct_dual_d36_cutgen.build_reduced_rows`), the
  exact Eisenstein coefficients `e_delta(g~)` of the Fricke transform, and the exact binding
  combination `r_3` / `COEF18*r_3`.
- `guarded_cE.txt` — the exact zeta(17)-guarded per-class Eisenstein lower-bound constants
  c_E(gamma), both sides (paper eq. (guard)); output of `code/d36_guarded_cE.py`.
- `receipt_d36_cutgen.txt` — construction receipt: constraint-generation iterations, exact optima
  (iter0 36.3470x -> iter1 32.9104x), exact sign scans to n=800.
- `receipt_d36_cs.txt` — verification receipt: newform/oldform decomposition of S_18(Gamma_0(24))
  (dim 64, level table), exact reconstruction gates (deviation identically 0 both sides), level-1
  eigenvalue gate, per-class Eisenstein constants (all 8 classes positive, binding gcd=3), the
  lift-aware Deligne constants C_w with the RIGOROUS outward-rounded interval certification section
  (exact-rational d=1 blocks + Sturm-interval algebraic blocks; soundness stress tests).

## Code (in the sibling `code/` directory, pure Python: sympy/mpmath/fractions — no CAS)
The `code/` directory ships the exact 12-module dependency closure plus the four cached
`.pkl` intermediates (all regenerable from the code), a pinned `requirements.txt`, and its own
`README.md`. Each module resolves its data paths relative to `code/`, so the layout is portable.
numpy/scipy are used only by the floating-point LP heuristic in the construction step, never in
the exact verification.
- `ct_dual_d36.py`, `ct_dual_general.py` — eta-quotient/Eisenstein bases, exact q-expansions,
  Fricke transform (i^k convention), value normalization.
- `exact_lp.py`, `ct_dual_d36_cutgen.py` — exact rational two-phase simplex + constraint generation.
- `eisen_projection.py` — exact Eisenstein projector on M_18(Gamma_0(24)) (Hecke-T5 spectral).
- `d36_cusp_reconstruct.py` — exact cusp components S = g - P_E(g), both sides; caches the vertex.
- `d36_newform_decomp.py` — exact echelon cusp basis (dim 64) + exact Hecke matrices T5,T7,U2,U3.
- `d36_cs_certificate.py` — newform/oldform decomposition, exact reconstruction gates, C_w, and the
  rigorous interval certification (`rigorous_C_S_w`).
- Independent second implementation of the decomposition (numerical simultaneous diagonalization
  path): `d36_tail_domination.py`, `d36_CS_crosschecks.py` (reproduces C_w to all reported digits;
  extends the exact sign scans to n=1100).

## Reproduction order (from the `code/` directory)
    pip install -r requirements.txt
1. `python3 d36_cusp_reconstruct.py 800` (rebuilds/caches the exact vertex; ~8 min)
2. `python3 d36_newform_decomp.py --selftest`
3. `python3 d36_cs_certificate.py` (decomposition + gates + interval certification)
4. `python3 d36_guarded_cE.py` (exact guarded c_E constants, paper eq. (guard))

The cached `.pkl` files are shipped, so steps 2–4 run immediately; step 1 regenerates the vertex
from scratch. See `code/README.md` for a per-module description.
