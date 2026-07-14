# A dual linear programming bound for sphere packing in dimension 36

Code and exact certificate for the paper *"A dual linear programming bound for
sphere packing in dimension 36"* (R. Jumagulov, 2026),
[arXiv:2607.11319](https://arxiv.org/abs/2607.11319).

**Result.** An explicit dual-feasible point for the Cohn–Elkies linear program in
dimension 36 shows

    delta_LP(36)  >=  B  =  32.91044... x (2^18 / 3^10)  >  146.1036,

i.e. the two-point LP bound in dimension 36 exceeds the density of the best packing
currently known (Kschischang–Pasupathy, center density 2^18/3^10) by a factor of at
least 32.91. In particular, no Cohn–Elkies auxiliary function can certify that
packing as optimal. To our knowledge this is the first such dual bound in any
dimension above 32. Here `B` is an explicit rational number (`data/certificate_exact_data.txt`).

The certificate is **exact and machine-checkable**: the dual vertex is a rational
vector, coefficient nonnegativity is verified by exact arithmetic up to n = 800, and
eventual positivity is proved via an explicit lift-aware Deligne-type tail bound
certified with outward-rounded interval arithmetic.

> **Paper:** `paper/d36_dual_bound.pdf` (source: `paper/d36_dual_bound.tex`).

## One-command verification

    cd code
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python3 verify_certificate.py        # prints VERIFIED iff every exact gate passes

`verify_certificate.py` rebuilds the certificate **from the published text data alone**
(`data/certificate_exact_data.txt` — it reads no `.pkl` cache) and checks, by exact
rational arithmetic: the 29 basis forms are independent (rank certificates); the vertex
satisfies `a_0 = 1`, `a_1..a_9 = 0`, and `a_n, b_n >= 0` for `n <= 800`; the published
exact `b_0`/`B` and their enclosures; the Eisenstein projection; and the guarded tail
crossovers `n_0 = 63 / 25`. The interval certification of the two tail constants `C_w`
is the separate step `python3 d36_cs_certificate.py` (below). Runtime: `verify_certificate.py`
~4 min; the full reproduction below ~10–15 min.

Data integrity: `shasum -c SHA256SUMS` (from the repository root) checks the exact
certificate data, receipts, and the two exported JSON inputs.

## Repository layout

    paper/   the manuscript (LaTeX source + compiled PDF)
    code/    Python pipeline: exact construction + verification + independent cross-checks
    data/    exact certificate data, receipts, and the file manifest (MANIFEST.md)
    SHA256SUMS   checksums for the proof-critical data files

- `code/` — pure-Python modules plus cached `.pkl` intermediates (all regenerable), a
  pinned `requirements.txt`, and `code/README.md` describing each module. The proof-core
  driver is `verify_certificate.py`; `data/MANIFEST.md` tiers every file
  (proof core / independent cross-checks / construction history).
- `data/` — `certificate_exact_data.txt` (the 29 basis forms, the 20-coordinate exact
  dual vertex, the coordinate-to-form map, exact `b_0`/`B`), `guarded_cE.txt` (exact
  per-class Eisenstein floor constants), the construction/verification/cross-check
  receipts, and `MANIFEST.md`.

## Requirements

Verification uses only **pure Python** (`sympy` / `mpmath` / `fractions`) — **no
computer-algebra system required**. Verified on Python 3.11 with sympy 1.14.0,
mpmath 1.4.1, numpy 2.4.6, scipy 1.17.1 (`numpy`/`scipy` are used only by the
floating-point LP heuristic in the *construction* step, never in the exact rigor pass).

Two **optional** independent cross-checks use a CAS: `d36_C_pari_indep.py` needs
`cypari2` (PARI/GP), and `d36_arb_check.py` needs `python-flint` (Arb). Neither is
required for the proof — they are independent reproductions of the tail constants.

## Full reproduction

From `code/` (the `.pkl` caches are shipped, so steps 2–4 run immediately; step 1
regenerates the dual vertex from scratch, ~8 min):

    python3 d36_cusp_reconstruct.py 800      # rebuild/cache the exact vertex
    python3 d36_newform_decomp.py --selftest # cusp basis + Hecke self-test
    python3 d36_cs_certificate.py            # decomposition + gates + interval certification of C_w
    python3 d36_guarded_cE.py                # exact guarded c_E constants + rigorous integer n_0

`d36_cs_certificate.py` prints `§36 VERDICT: FULLY PROVED` and writes a receipt
byte-identical to `data/receipt_d36_cs.txt` (it also prints the exact 33 determining
rows, the minimal pivot magnitude certifying nonsingularity, and dyadic-rational upper
bounds for `C_w`).

## Independent verification of the theorem-critical step

The interval solve for the tail constants `C_w` is reproduced **four independent ways**,
all agreeing to every reported digit:

- exact characteristic-polynomial factorization + `mpmath.iv` intervals (`d36_cs_certificate.py`);
- high-precision simultaneous diagonalization (`d36_tail_domination.py`, `d36_CS_crosschecks.py`);
- PARI/GP native modular-symbols basis rebuilt from scratch (`d36_C_pari_indep.py`, optional `cypari2`);
- Arb ball arithmetic, a certified-arithmetic re-solve of the 33×33 system (`d36_arb_check.py`, optional `python-flint`).

## How the result is checkable regardless of how it was produced

This work was carried out with substantial assistance from AI language-model tools.
Its correctness does not rest on trust in those tools: the certificate is exact and
machine-checkable, `verify_certificate.py` reproduces every exact gate from the
published data alone (and demonstrably rejects a corrupted certificate), and the
tail constants are reproduced by the four independent implementations above.

## Citation

See `CITATION.cff`, or cite the manuscript in `paper/`.

## License

MIT — see `LICENSE`.
