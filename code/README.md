# Code — d=36 dual linear programming bound

Self-contained Python implementation for the paper *"A dual linear programming
bound for sphere packing in dimension 36."* This directory contains the exact
dependency closure plus the cached intermediate data (4 `.pkl`, all regenerable),
so every numerical claim in the paper can be reproduced with no external files.

**Start here (v2/v3):** `python3 verify_certificate.py` — the single verification
driver. It rebuilds the certificate from the published text data alone
(`../certificate_exact_data.txt`; no `.pkl` caches are read) and prints
`VERIFIED` only if all exact gates pass. The interval certification
of the two `C_w` upper bounds is the separate step
`python3 d36_cs_certificate.py`; the optional independent Arb re-verification
of its 33×33 interval solve is `d36_arb_check.py` (python-flint). File tiers
(proof core / independent cross-checks / construction history): `../MANIFEST.md`.

Runtime metadata (commodity laptop, Python 3.11): `verify_certificate.py` ≈ 4 min /
<1 GB RAM; `d36_cs_certificate.py` ≈ 2–3 min (with shipped caches);
`d36_cusp_reconstruct.py 800` ≈ 8 min; `d36_span_rank_certify.py` ≈ 25 s;
`d36_C_pari_indep.py` ≈ 1–2 min; `d36_arb_check.py` ≈ 5 s.

## Environment
Pure Python — no computer-algebra system (no Sage/Pari) required.
See `requirements.txt`. Verified on Python 3.11.15 with
sympy 1.14.0, mpmath 1.4.1, numpy 2.4.6, scipy 1.17.1.

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

Exact verification uses only `sympy` / `mpmath` / `fractions`. `numpy`/`scipy`
are used **only** by the floating-point LP heuristic in the *construction* step;
they play no role in the exact rigor pass.

## Modules
Construction (the dual vertex):
- `ct_dual_general.py`   — eta-quotient/Eisenstein bases, exact q-expansions, Fricke transform (i^k), value normalization.
- `ct_dual_d36.py`       — d=36 specialization: DIVS, COEF18, fast exact q-expansions; float-LP heuristic (scipy).
- `exact_lp.py`          — exact rational two-phase simplex.
- `ct_dual_d36_cutgen.py`— constraint-generation driver (exact); builds `cutgen_rows_cache.pkl`, writes `receipt_d36_cutgen.txt`.
- `ct_dual_d36_mpmath800.py` — high-precision LP fallback / scan support.
- `eisen_projection.py`  — exact Eisenstein projector on M_18(Gamma_0(24)) (uses `eisen_projection_PROJ.pkl`).

Verification (eventual positivity):
- `verify_certificate.py`   — v2 single driver: rebuilds the certificate from `../certificate_exact_data.txt` alone and runs all exact gates (G1–G7); prints `VERIFIED`.
- `d36_cusp_reconstruct.py` — exact cusp components S = g - P_E(g), both sides; builds/caches `d36_iter1_vertex.pkl`.
- `d36_newform_decomp.py`   — exact cusp basis (dim 64) + exact Hecke matrices; builds `d36_cusp_basis_hecke.pkl`. `--selftest` gate.
- `d36_cs_certificate.py`   — newform/oldform decomposition, exact reconstruction gates, lift-aware C_w, and the RIGOROUS outward-rounded interval certification. Writes `receipt_d36_cs.txt`.
- `d36_guarded_cE.py`       — exact zeta(17)-guarded per-class Eisenstein constants c_E(gamma) (paper eq. (guard)); writes `guarded_cE.txt`.
- `d36_tail_domination.py`, `d36_CS_crosschecks.py` — independent second implementation (numerical simultaneous diagonalization); reproduces C_w to all reported digits and extends the exact sign scans to n=1100.

## Cached data (regenerable)
All four `.pkl` are produced by the code above (not opaque inputs):
- `cutgen_rows_cache.pkl`      <- ct_dual_d36_cutgen.build_reduced_rows
- `d36_cusp_basis_hecke.pkl`   <- d36_newform_decomp.py (step 1)
- `d36_iter1_vertex.pkl`       <- d36_cusp_reconstruct.py
- `eisen_projection_PROJ.pkl`  <- eisen_projection.py

Paths are resolved relative to this directory (each module sets
`CODE = os.path.dirname(os.path.abspath(__file__))`), so the bundle is portable.

## Reproduction order
1. `python3 d36_cusp_reconstruct.py 800`   # rebuilds/caches the exact vertex (~8 min)
2. `python3 d36_newform_decomp.py --selftest`
3. `python3 d36_cs_certificate.py`          # decomposition + gates + interval certification
4. `python3 d36_guarded_cE.py`              # exact guarded c_E constants (paper eq. (guard))

Shipping the caches lets steps 2–4 run immediately; step 1 regenerates the vertex from scratch.
