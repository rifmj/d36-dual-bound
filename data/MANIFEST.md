# Ancillary files — "A dual linear programming bound for sphere packing in dimension 36"

This bundle is self-contained: the exact certificate data, the receipts, and the complete
code needed to reproduce every numerical claim in the paper are all included here. No external
files are required. **v2 (2026-07-14):** files are now tiered into PROOF CORE / INDEPENDENT
CROSS-CHECKS / CONSTRUCTION HISTORY, a single verification driver is provided, and
`SHA256SUMS` pins the data files.

## Quick verification (the referee path)

    cd code/
    pip install -r requirements.txt
    python3 verify_all.py                # one command — checksums + exact + interval + Arb, nonzero exit on any failure
    # or run the stages individually:
    python3 verify_certificate.py        # ~3-4 min; prints VERIFIED iff every gate passes
    python3 d36_cs_certificate.py        # the interval certification of the C_w upper bounds

`verify_certificate.py` reconstructs everything from the PUBLISHED TEXT DATA alone
(`certificate_exact_data.txt`): it re-expands the 29 listed basis forms exactly, rebuilds the
reduced basis as the exact nullspace of the printed vanishing conditions, reassembles the
certificate from the printed 20 coordinates, and checks — all by exact rational comparison —
feasibility to n=800, the published exact b_0 and B, the Eisenstein projection e_delta(g~),
and the guarded tail crossovers n0 = 63/25. It does NOT read any `.pkl` cache.
`d36_cs_certificate.py` supplies the one remaining ingredient: the outward-rounded interval
certificates of the two C_w upper bounds (used by the crossover gate).

## Tier 1 — PROOF CORE (the files Theorem 1.1 rests on)

Data:
- `certificate_exact_data.txt` — the exact rational certificate: the explicit list of the 29
  basis forms (v2; pivot order = the coordinate order), the 20 free coordinates of the dual
  vertex, the coordinate-to-form map, the exact rational b_0 and B(g) with outward decimal
  enclosures, the exact Eisenstein coefficients e_delta(g~), the exact binding combination
  r_3 / COEF18*r_3, and the rounding conventions.
- `guarded_cE.txt` — the zeta(17)-guarded per-class Eisenstein lower-bound constants
  c_E(gamma), both sides (paper eq. (guard)), as EXACT RATIONALS (v2) plus certified decimal
  displays and the rigorous integer crossovers; output of `code/d36_guarded_cE.py`.
- `SHA256SUMS` — checksums of the data files above and the receipts.

Code (pure Python: sympy/mpmath/fractions — no CAS; paths resolve relative to `code/`):
- `verify_certificate.py` — the single verification driver (see above).
- `ct_dual_d36.py`, `ct_dual_general.py` — eta-quotient/Eisenstein exact q-expansions, Fricke
  transform (i^k convention), value normalization.
- `eisen_projection.py` — exact Eisenstein projector on M_18(Gamma_0(24)) (Hecke-T5 spectral).
- `d36_newform_decomp.py` — exact echelon cusp basis (dim 64) + exact Hecke matrices T5,T7,U2,U3.
- `d36_cs_certificate.py` — newform/oldform decomposition, exact rational-block reconstruction
  gates, and the rigorous outward-rounded interval certification of C_w (`rigorous_C_S_w`:
  exact rational d=1 blocks + Sturm-interval algebraic blocks, sound interval Gaussian solve).
- `d36_guarded_cE.py` — exact guarded c_E constants + rigorous integer n0 (paper eq. (guard)).
- `d36_span_rank_certify.py` + `../receipt_d36_span_rank.txt` — exact (mod-p) certificate
  that the 3052 candidates span M_18(Gamma_0(24)) (rank 72; supports the Sec. 3.1 spanning
  statement; not needed for Theorem 1.1 itself).
- `requirements.txt` — pinned versions.

## Tier 2 — INDEPENDENT CROSS-CHECKS (not needed for the proof; reproduce its constants)

- `d36_tail_domination.py`, `d36_CS_crosschecks.py` — independent second implementation of the
  newform decomposition (high-precision simultaneous diagonalization, no shared code path with
  Tier 1's exact-charpoly route); reproduces both C_w to all reported digits and extends the
  exact sign scans to n=1100.
- `ct_dual_d36_mpmath800.py` — independent high-precision re-scan of the vertex.
- `d36_C_pari_indep.py` + `d36_export_cusp_targets.py` + `d36_cusp_targets_L150.json` +
  `../receipt_d36_pari_indep.txt` (v2) — THIRD, CAS-independent reproduction: PARI/GP 2.17.2
  native modular-symbols engine (mfinit/mfeigenbasis/mfbd/mfembed) rebuilds the 64-form
  newform/oldform lift basis of S_18(Gamma_0(24)) from scratch and reproduces both C_w to all
  12 reported digits (relative reconstruction residual 0 at 1536-bit embeddings), plus the
  newspace dimension table, the level-1 eigenvalues, and a Deligne-bound spot-check.
  Requires `cypari2` (optional CAS dependency; everything in Tier 1
  remains CAS-free). Values are numerical reproductions; the RIGOROUS upper bound on C_w
  remains the Tier-1 interval certificate.
- `d36_arb_check.py` + `d36_alg_system_export.json` + `../receipt_d36_arb_check.txt` (v3) —
  independent re-verification of the theorem-critical interval step in **Arb ball arithmetic**
  (python-flint): consumes the exact dyadic endpoints of the 33x33 system exported by
  `d36_cs_certificate.py`, re-solves it with Arb's certified ball solve (success certifies
  nonsingularity independently of the mpmath pivot log) and re-aggregates
  sum |lambda|/e^{8.5}; both sides agree with the receipt to all 12 digits, PASS. Scope: the
  solve + aggregation; the lift-vector enclosure CONSTRUCTION (exact Sturm data + mpmath.iv
  propagation) is corroborated by the Tier-2 reproductions above. Requires `python-flint`
  (optional).

## Tier 3 — CONSTRUCTION HISTORY (how the certificate was FOUND; not part of the proof)

- `exact_lp.py`, `ct_dual_d36_cutgen.py` — exact rational two-phase simplex + constraint
  generation (the search that produced the vertex). The floating-point heuristics documented
  in the paper (pivot-basis selection, candidate location) live here.
- `d36_cusp_reconstruct.py` — exact cusp components S = g - P_E(g), both sides; regenerates and
  caches the vertex (`python3 d36_cusp_reconstruct.py 800`, ~8 min).
- `receipt_d36_cutgen.txt` — construction receipt: constraint-generation iterations, exact
  optima (iter0 36.3470x -> iter1 32.9104x, both RELATIVE TO THE 29-FORM SUBSPACE, paper
  Sec. 3.1), exact sign scans to n=800. NOTE: as a historical log it also contains
  EXPLORATION-STAGE numbers superseded by the paper (e.g. the naive pre-lift-aware crossover
  estimates n0~783/827 and "CANDIDATE pending rigor" status lines); the paper's final constants
  are those of `receipt_d36_cs.txt` and `guarded_cE.txt`.
- `receipt_d36_cs.txt` — final verification receipt: newform/oldform decomposition of
  S_18(Gamma_0(24)) (dim 64, level table), exact reconstruction gates (deviation identically 0
  both sides), level-1 eigenvalue gate, per-class Eisenstein constants, and the RIGOROUS
  outward-rounded interval certification of C_w with soundness stress tests.
- The four cached `.pkl` intermediates (`cutgen_rows_cache.pkl`, `d36_iter1_vertex.pkl`,
  `d36_cusp_basis_hecke.pkl`, `eisen_projection_PROJ.pkl`) — ALL regenerable from the code;
  shipped only so that Tier 1/2 scripts run immediately. `verify_certificate.py` does not read
  them (it rebuilds from the published text data); `d36_cs_certificate.py` regenerates its
  inputs via `d36_cusp_reconstruct.py`/`d36_newform_decomp.py` if the caches are removed.

## Full reproduction order (from the `code/` directory)

    pip install -r requirements.txt
0. `python3 verify_certificate.py`            (Tier-1 driver, no caches; VERIFIED expected)
1. `python3 d36_cusp_reconstruct.py 800`      (regenerates the vertex from scratch; ~8 min)
2. `python3 d36_newform_decomp.py --selftest`
3. `python3 d36_cs_certificate.py`            (decomposition + gates + interval certification)
4. `python3 d36_guarded_cE.py`                (exact guarded c_E + rigorous integer n0)

See `code/README.md` for a per-module description.
