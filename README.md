# A dual linear programming bound for sphere packing in dimension 36

Code and exact certificate for the paper *"A dual linear programming bound for
sphere packing in dimension 36"* (R. Jumagulov, 2026).

**Result.** An explicit dual-feasible point for the Cohn–Elkies linear program in
dimension 36 shows

    delta_LP(36)  >=  146.1036...  =  32.9104... x (2^18 / 3^10),

i.e. the two-point LP bound in dimension 36 exceeds the density of the best packing
currently known (Kschischang–Pasupathy, center density 2^18/3^10) by a factor of at
least 32.91. In particular, no Cohn–Elkies auxiliary function can certify that
packing as optimal. To our knowledge this is the first such dual bound in any
dimension above 32.

The certificate is **exact and machine-checkable**: the dual vertex is a rational
vector, coefficient nonnegativity is verified by exact arithmetic up to n = 800, and
eventual positivity is proved via an explicit lift-aware Deligne-type tail bound
certified with outward-rounded interval arithmetic.

> **Paper:** `paper/d36_dual_bound.pdf` (source: `paper/d36_dual_bound.tex`).

## Repository layout

    paper/   the manuscript (LaTeX source + compiled PDF)
    code/    Python pipeline (exact construction + verification) + cached intermediates
    data/    exact certificate data, receipts, and the file manifest

- `code/` — 12 modules (pure Python) plus 4 `.pkl` caches, a pinned
  `requirements.txt`, and `code/README.md` describing each module.
- `data/` — `certificate_exact_data.txt`, `guarded_cE.txt`, the two construction /
  verification receipts, and `MANIFEST.md`.

## Requirements

Pure Python — **no computer-algebra system** (no Sage/Pari). Verified on
Python 3.11 with sympy 1.14.0, mpmath 1.4.1, numpy 2.4.6, scipy 1.17.1.

    cd code
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

The exact verification uses only `sympy` / `mpmath` / `fractions`; `numpy` and
`scipy` are used only by the floating-point LP heuristic in the construction step,
never in the exact rigor pass.

## Reproduce the certificate

From `code/` (the four `.pkl` caches are shipped, so steps 2–4 run immediately;
step 1 regenerates the dual vertex from scratch, ~8 min):

    python3 d36_cusp_reconstruct.py 800      # rebuild/cache the exact vertex
    python3 d36_newform_decomp.py --selftest # cusp basis + Hecke self-test
    python3 d36_cs_certificate.py            # decomposition + gates + interval certification
    python3 d36_guarded_cE.py                # exact guarded c_E constants (paper eq. (guard))

`d36_cs_certificate.py` prints `§36 VERDICT: FULLY PROVED` and writes a receipt
byte-identical to `data/receipt_d36_cs.txt`.

## How the result is checkable regardless of how it was produced

This work was carried out with substantial assistance from AI language-model tools.
Its correctness does not rest on trust in those tools: the certificate is exact and
every numerical claim is reproduced by re-running the verification above from
scratch. A second, independent implementation of the newform decomposition
(numerical simultaneous diagonalization, in `code/d36_tail_domination.py` and
`code/d36_CS_crosschecks.py`) reproduces the tail constants to all reported digits.

## Citation

See `CITATION.cff`, or cite the manuscript in `paper/`.

## License

MIT — see `LICENSE`.
