#!/usr/bin/env python3
r"""
§34 — d=36 CE-dual >record certificate attempt via CONSTRAINT GENERATION (cutting planes).

The instrument the referee (Sol, §33) named and §30's DENSE-LP escapes did NOT try. §30 died on the
full ~800-row degenerate LP (mpmath/HiGHS); this solver instead iteratively adds ONLY the actively
violated b_n>=0 rows to a WORKING SET, keeping every solve a small exact-rational simplex on the SAME
reduced basis (r=20 free variables from the M=28 nullspace, r fixed => NO §30 non-monotone-below-M=72
basis growth).

DECISIVE OPEN QUESTION (§30/§33): at N=24, T=10, once b_n>=0 is enforced pointwise through n0~783,
is  max{ b_0 }  STILL > record  delta_c(KP_36) = 2^18/3^10 = 4.4394316585  (ratio > 1)?
  - >record  => a CE dual point that certifies "CE 2-pt LP cannot certify KP_36 as optimal" at d=36,
               modulo the finite exact tail proof past n0 (Cohn-Triantafillou Sec 5: Eisenstein main
               term ~sigma_17(n) dominates the Deligne-bounded cusp remainder for n>n0). [weak form]
  - <=record => NO >record tail-safe certificate exists at N=24 (enforcing the tail drives the optimum
               below the record). Report the bound-vs-iteration curve + limiting bound.
BOTH are deliverables (§33: existence of a feasible >record point after enforcing the tail is UNKNOWN).

METHOD
  1. Reduced basis Bm = nullspace(Gm[1:T]) built ONCE at M=28 (r=20 free vars x). Exact.
  2. Exact q-expansions of g and Fricke image g~ to length L (<=800) computed ONCE (the only heavy
     sympy step); reduced rows GAr[m]=coeffs(a_m) . x, GBr[m]=coeffs(b_m) . x precomputed as Fractions.
  3. LP (exact rational two-phase simplex, exact_lp.solve_exact):
        maximize b_0 = GBr[0].x
        s.t. a_0 = GAr[0].x = 1 ;  a_n = GAr[n].x >= 0  (T<=n<=M=28)
             b_n = GBr[n].x >= 0  for n in working set W   (init W = {1..M})
  4. Evaluate b_n = GBr[n].x for ALL n<=L (exact Fractions). Collect violated {n : b_n < 0}. Add the
     most-violated (most-negative, normalized) up to ADD_PER_ITER of them to W. Re-solve. Iterate.
  5. Track bound = b_0 * (2/sqrt(N))^(d/2) * (sqrt(T)/2)^d  vs record, per iteration.
  6. TERMINATE when: (a) no violations up to L with bound>record  => CERTIFICATE candidate (then rigor
     pass: re-verify all a_n,b_n signs exactly to L + state the tail proof); or (b) LP becomes
     infeasible, or objective stabilizes <= record  => NO cert at N=24; or (c) time budget.

All LP arithmetic exact (fractions.Fraction). Reuses D.build_basis / Gg.build_basis / X.solve_exact;
NO nsimplify on any large rational (Rational(p,q) construction only). New file; modifies nothing.

Run:  python3 ct_dual_d36_cutgen.py [L] [TIME_BUDGET_SEC]   (defaults L=800, budget=1500s)
Pure-mathematics research; standard modular-forms / LP-dual jargon.
"""
from __future__ import annotations
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import sympy as sp

import ct_dual_d36 as D
import ct_dual_general as Gg
import exact_lp as X

N, K, d = 24, 18, 36
T, M = 10, 28
RECORD = sp.Rational(2**18, 3**10)          # KP_36 center density = 4.4394316585...
RECORD_F = float(RECORD)
CODE = os.path.dirname(os.path.abspath(__file__))  # resolve caches/receipts next to this file
CACHE = os.path.join(CODE, 'cutgen_rows_cache.pkl')


def bound_of(b0_fr):
    """center-density bound = b_0 * (2/sqrt N)^(d/2) * (sqrt T/2)^d  (float; b0 exact Fraction)."""
    return float(b0_fr) * (2 / N ** 0.5) ** (d / 2) * (T ** 0.5 / 2) ** d


def build_reduced_rows(L, out):
    """Build the reduced LP rows GAr[m], GBr[m] (exact Fraction vectors, len r) for m=0..L, ONCE.

    Returns (GAr, GBr, r, specs, Bm, n).  GAr/GBr are lists (len L+1) of Fraction-vectors (len r).
    Cached to disk keyed on L (the sympy q-expansion to length 800 is the only heavy step, ~minutes).
    """
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            C = pickle.load(f)
        if C['L'] >= L and C['N'] == N and C['K'] == K and C['T'] == T and C['M'] == M:
            p(out, f"  [cache] loaded reduced rows L={C['L']} (using first {L+1})")
            GAr = [[Fr(*t) for t in row] for row in C['GAr'][:L + 1]]
            GBr = [[Fr(*t) for t in row] for row in C['GBr'][:L + 1]]
            return GAr, GBr, C['r'], C['specs'], C['n']

    t0 = time.time()
    specs, gM, gtM, rank, DIM, nsol = D.build_basis(N, K, M, B=8, verbose=False)
    n = len(gM)
    Gm = sp.Matrix(M + 1, n, lambda m, j: gM[j][m])
    ns = Gm[1:T, :].nullspace()
    Bm = sp.Matrix.hstack(*ns)
    r = Bm.shape[1]
    p(out, f"  reduced basis: n={n} forms, r={r} free vars (nullspace of first {T-1} vanishing rows), "
           f"dim M_{K}(G0({N}))={DIM}  [{time.time()-t0:.0f}s]")

    # exact q-expansions of g and Fricke image g~ to length L via the FAST integer/Fraction helpers
    # (D._etaq_fast / D._Ek_fast / D._shift — pure Python int, EXACT; verified to match sympy
    # Gg.eta_quotient_qexp/E_k digit-for-digit at M=28). The sympy Rational q-expansion of 29 forms
    # to L=800 is ~minutes/spec; this is seconds total. Fricke constants C_j (29 of them, all clean
    # rationals for N=24,k=18 per §31) come from Gg once. g_j[m], g~_j[m] end up as exact Fractions.
    t1 = time.time()

    def to_fr(x):
        q = sp.Rational(x)
        return Fr(int(q.p), int(q.q))

    gg = []    # g_j : exact Fraction coeffs len L+1
    gtt = []   # g~_j = C_j * (Fricke q-expansion) : exact Fraction coeffs len L+1
    for s in specs:
        if s[0] == "eis":
            delta = s[1]
            base = D._Ek_fast(K, L)                      # Fraction coeffs of E_k
            gj = [Fr(0)] * (L + 1)
            for m in range(L + 1):
                if m % delta == 0:
                    gj[m] = base[m // delta]             # scale q->q^delta
            C, _ = Gg.fricke_eisen(delta, K, N, L)       # clean rational constant
            Cf = to_fr(C)
            himg = [Fr(0)] * (L + 1)                     # Fricke image E_k(q^{N/delta})
            dd = N // delta
            for m in range(L + 1):
                if m % dd == 0:
                    himg[m] = base[m // dd]
            gtj = [Cf * himg[m] for m in range(L + 1)]
        else:
            exps = s[1]
            gj = [to_fr(v) for v in D._etaq_fast(exps, L)]        # exact int -> Fraction
            C, _ = Gg.fricke_etaq(exps, K, N, L)
            Cf = to_fr(C)
            rev = {}                                     # reversed quotient delta -> N/delta
            for dl, rr in exps.items():
                rev[N // dl] = rev.get(N // dl, 0) + rr
            himg = [to_fr(v) for v in D._etaq_fast(rev, L)]
            gtj = [Cf * himg[m] for m in range(L + 1)]
        gg.append(gj); gtt.append(gtj)
    p(out, f"  exact q-expansions to L={L} (fast integer path): [{time.time()-t1:.0f}s]")

    # reduced rows: GA[m,l] = sum_j gg[j][m] * Bm[j,l] ; GB likewise with gtt. All exact Fractions.
    t2 = time.time()
    Bcol = [[to_fr(Bm[j, l]) for j in range(n)] for l in range(r)]  # Bm[:,l] as Fractions

    GAr, GBr = [], []
    for m in range(L + 1):
        gm = gg  # already Fractions
        rowA = [sum((gg[j][m] * Bcol[l][j] for j in range(n)), Fr(0)) for l in range(r)]
        rowB = [sum((gtt[j][m] * Bcol[l][j] for j in range(n)), Fr(0)) for l in range(r)]
        GAr.append(rowA); GBr.append(rowB)
        if m % 200 == 0 and m:
            p(out, f"    reduced rows m={m}/{L}  [{time.time()-t2:.0f}s]")
    p(out, f"  reduced rows built m=0..{L}: [{time.time()-t2:.0f}s]")

    with open(CACHE, 'wb') as f:
        pickle.dump({'L': L, 'N': N, 'K': K, 'T': T, 'M': M, 'r': r, 'n': n, 'specs': specs,
                     'GAr': [[(v.numerator, v.denominator) for v in row] for row in GAr],
                     'GBr': [[(v.numerator, v.denominator) for v in row] for row in GBr]}, f)
    return GAr, GBr, r, specs, n


import mpmath
import ct_dual_d36_mpmath800 as MP
DPS = 300   # §30 + this file's calibration: dps>=300 reproduces the EXACT vertex (a_0=1, 36.347x@iter0);
            # dps in {80,120,200} give PHANTOM vertices (a_0=0.9997/596e9). dps=300 mpmath == exact.


def _fr2mp(v):
    return mpmath.mpf(v.numerator) / mpmath.mpf(v.denominator)


def _exact_solve_active(GAr, GBr, r, W):
    """Exact-rational LP via the pure-Fraction two-phase simplex (audited exact_lp). SLOW but the
    ground truth; used for small W / final certification. Returns (status, x_exact, b0_exact)."""
    A_eq = [GAr[0]]
    b_eq = [Fr(1)]
    A_ub = [[-v for v in GAr[m]] for m in range(T, M + 1)]
    A_ub += [[-v for v in GBr[nn]] for nn in sorted(W)]
    b_ub = [Fr(0)] * len(A_ub)
    st, x, ob = X.solve_exact(GBr[0], A_eq, b_eq, A_ub, b_ub, r)
    if st != 'optimal':
        return st, None, None
    b0 = sum(GBr[0][l] * x[l] for l in range(r))
    return 'optimal', x, b0


def solve_working_set(GAr, GBr, r, W, exact=False):
    """max b_0 s.t. a_0=1, a_n>=0 (T<=n<=M), b_n>=0 (n in W).

    exact=True  -> pure-Fraction simplex (ground truth, slow).
    exact=False -> mpmath dps=300 simplex (validated == exact @iter0), then EXACT rational
                   reconstruction of the vertex from the mpmath-identified TIGHT constraint set
                   (so the returned x is EXACT and violations are checked exactly). On
                   reconstruction failure falls back to exact.
    Always returns (status, x_exact, b0_exact, tight_b) with tight_b = {n in W : b_n==0 exactly}.
    """
    if exact:
        st, x, b0 = _exact_solve_active(GAr, GBr, r, W)
        if st != 'optimal':
            return st, None, None, set()
        tight_b = set(nn for nn in W if sum(GBr[nn][l] * x[l] for l in range(r)) == 0)
        return 'optimal', x, b0, tight_b

    mpmath.mp.dps = DPS
    Wl = sorted(W)
    GAm = {m: [_fr2mp(GAr[m][l]) for l in range(r)] for m in range(0, M + 1)}
    GBm = {}
    for nn in Wl:
        GBm[nn] = [_fr2mp(GBr[nn][l]) for l in range(r)]
    GBm0 = [_fr2mp(GBr[0][l]) for l in range(r)]

    c = GBm0
    A_eq = [[GAm[0][l] for l in range(r)]]
    b_eq = [mpmath.mpf(1)]
    A_ub = [[-GAm[m][l] for l in range(r)] for m in range(T, M + 1)]
    A_ub += [[-GBm[nn][l] for l in range(r)] for nn in Wl]
    b_ub = [mpmath.mpf(0)] * len(A_ub)
    st, y, _, iters = MP.mp_solve(c, A_eq, b_eq, A_ub, b_ub, r)
    if st != 'optimal':
        return st, None, None, set()

    # identify tight constraints at the mpmath vertex (a_0=1 equality always tight)
    tol = mpmath.mpf(10) ** (-(DPS - 40))
    tight = []   # list of exact Fraction rows (the constraint value == its rhs); rhs=1 for a0 else 0
    tight.append((GAr[0], Fr(1)))
    for m in range(T, M + 1):
        val = sum(GAm[m][l] * y[l] for l in range(r))
        if abs(val) < tol:
            tight.append((GAr[m], Fr(0)))
    for nn in Wl:
        val = sum(GBm[nn][l] * y[l] for l in range(r))
        if abs(val) < tol:
            tight.append((GBr[nn], Fr(0)))

    # EXACT reconstruction: pick r independent tight rows, solve the exact r x r system.
    x_exact = _reconstruct_exact(tight, r)
    if x_exact is None:
        # fall back to the exact simplex (rare: degenerate vertex or rank shortfall)
        st2, x2, b02 = _exact_solve_active(GAr, GBr, r, W)
        return (st2, x2, b02, set(W)) if st2 == 'optimal' else (st2, None, None, set())
    b0 = sum(GBr[0][l] * x_exact[l] for l in range(r))
    # which b_n constraints (n in W) are TIGHT at the exact vertex (for cutting-plane pruning)
    tight_b = set()
    for nn in Wl:
        if sum(GBr[nn][l] * x_exact[l] for l in range(r)) == 0:
            tight_b.add(nn)
    return 'optimal', x_exact, b0, tight_b


def _reconstruct_exact(tight_rows, r):
    """Given tight constraint rows (row_vec, rhs) with row_vec exact Fractions, select r linearly
    independent ones and solve the exact rational linear system row.x = rhs. Returns x (list of
    Fraction) or None if fewer than r independent rows. Gaussian elimination over Fractions."""
    # build augmented matrix, do exact Gaussian elimination selecting pivot rows greedily
    rows = [([Fr(v) for v in rv], Fr(rhs)) for (rv, rhs) in tight_rows]
    chosen = []
    # work on a copy for elimination to test independence
    A = []  # reduced pivot rows [coeffs..., rhs]
    for (rv, rhs) in rows:
        vec = rv[:] + [rhs]
        # reduce against existing pivots
        for (pc, prow) in A:
            if vec[pc] != 0:
                f = vec[pc]
                vec = [vec[k] - f * prow[k] for k in range(r + 1)]
        # find a pivot column
        pc = next((k for k in range(r) if vec[k] != 0), None)
        if pc is None:
            continue  # dependent row
        prow = [vk / vec[pc] for vk in vec]
        A.append((pc, prow))
        chosen.append(pc)
        if len(A) == r:
            break
    if len(A) < r:
        return None
    # back-substitute: A has r pivot rows; solve. Build full system from pivots.
    # Each pivot row: prow[pc]=1, express. Do full reduced row echelon.
    # sort by pivot col
    A.sort(key=lambda t: t[0])
    # eliminate above pivots
    for i in range(len(A)):
        pc_i, row_i = A[i]
        for j in range(len(A)):
            if j != i:
                pc_j, row_j = A[j]
                if row_j[pc_i] != 0:
                    f = row_j[pc_i]
                    A[j] = (pc_j, [row_j[k] - f * row_i[k] for k in range(r + 1)])
    x = [Fr(0)] * r
    for (pc, row) in A:
        x[pc] = row[r]
    return x


def violations(GBr, x, L):
    """Exact b_n for n=1..L; return sorted list of (normalized_violation, n, b_n_float) for b_n<0.

    normalized_violation = -b_n / (1 + max|coeff in row|)  (scale-free, most-negative first).
    """
    viol = []
    for nn in range(1, L + 1):
        row = GBr[nn]
        bn = sum(row[l] * x[l] for l in range(len(x)))
        if bn < 0:
            scale = 1 + max((abs(v) for v in row), default=Fr(1))
            viol.append((float(-bn / scale), nn, float(bn)))
    viol.sort(reverse=True)
    return viol


def p(out, *args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    out.append(s)


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0
    # ADD_PER_ITER: rows added per round. DEFAULT = huge (add ALL violations) — empirically the
    # cutting plane then CONVERGES in ~1 round (iter0 max-b0 -> iter1 = full eventually-positive
    # optimum), because ALL b_n<0 live in the 4 gcd=3 classes and are enforced together. EXACT_MODE
    # (default True) uses the pure-Fraction simplex: SLOW per solve (~5 min at |W|~73) but the ONLY
    # trustworthy value (mpmath dps<=300 returns PHANTOM vertices a_0!=1 once b_n>=0 rows with 1e30+
    # coefficients are added — verified). Override: argv[3]=add_per_iter, argv[4]='mp' for mpmath.
    ADD_PER_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    EXACT_MODE = (len(sys.argv) <= 4) or (sys.argv[4].lower() != 'mp')
    t_start = time.time()
    out = []
    p(out, "=" * 80)
    p(out, "§34 d=36 CE-dual >record CONSTRAINT-GENERATION certificate attempt")
    p(out, "=" * 80)
    p(out, f"N={N} T={T} M={M} d={d} k={K};  record delta_c(KP_36)=2^18/3^10={RECORD_F:.10f}")
    p(out, f"L (tail enforced through) = {L};  time budget = {BUDGET:.0f}s;  add<= {ADD_PER_ITER}/iter;  "
           f"solver = {'EXACT Fraction (trustworthy)' if EXACT_MODE else 'mpmath dps=300 + exact reconstruction'}")
    p(out, f"scale factor (2/sqrtN)^(d/2)(sqrtT/2)^d = {(2/N**0.5)**(d/2)*(T**0.5/2)**d:.6f}")

    GAr, GBr, r, specs, n = build_reduced_rows(L, out)
    p(out, f"  => r={r} free vars; {n} forms; rows ready. [{time.time()-t_start:.0f}s]\n")

    # ---- iteration 0: baseline max-b0 with only W={1..M} (reproduces §29's 36.35x) ----
    # W = active b_n>=0 working set. CUTTING-PLANE HYGIENE: after each solve we PRUNE W to the
    # constraints that are actually TIGHT (b_n==0) + a few near-tight, then ADD the newly violated
    # rows. This keeps |W| ~ O(r)=O(20) so each mpmath dps=300 solve stays fast (a bloated W of
    # hundreds of slack rows is what makes the simplex degeneracy-crippled — the §30 wall). The
    # OBJECTIVE is still the true max over ALL enforced-so-far cuts because a dropped constraint had
    # positive slack at the optimum and is re-added immediately if a later vertex violates it; we
    # ALSO re-scan all n<=L every iteration, so nothing is silently skipped.
    W = set(range(1, M + 1))
    p(out, "-" * 80)
    p(out, "CONSTRAINT-GENERATION ITERATIONS (objective = center-density bound; record ratio in ())")
    p(out, "  (W = active b_n>=0 cut set, pruned to tight+recent each iter; ALL n<=L rescanned)")
    p(out, "-" * 80)
    curve = []   # (iter, |W|, bound, ratio, n_violations, worst_n, worst_bn)
    it = 0
    verdict = None
    last_added = list(range(1, M + 1))     # cuts added in the previous round (anti-cycle retention)
    ever_enforced = set(range(1, M + 1))   # union of every n ever put in W (audit trail)
    while True:
        st, x, b0, tight_b = solve_working_set(GAr, GBr, r, W, exact=EXACT_MODE)
        if st != 'optimal':
            p(out, f"[iter {it}] |W|={len(W)}  LP status = {st.upper()}")
            if st == 'infeasible':
                verdict = ('INFEASIBLE', None, len(W))
                p(out, "  => enforcing b_n>=0 on the working set makes the dual LP INFEASIBLE at N=24,T=10.")
                p(out, "     NO >record certificate at N=24 (no dual-feasible point with b_n>=0 through the set).")
            else:
                verdict = (st.upper(), None, len(W))
            break
        bd = bound_of(b0)
        ratio = bd / RECORD_F
        viol = violations(GBr, x, L)      # EXACT sign scan over ALL 1<=n<=L
        nv = len(viol)
        worst_n = viol[0][1] if viol else None
        worst_bn = viol[0][2] if viol else None
        curve.append((it, len(W), bd, ratio, nv, worst_n, worst_bn))
        tag = ">record" if bd > RECORD_F else "<=RECORD"
        wn = f"n={worst_n} b_n={worst_bn:.3e}" if worst_n is not None else "none"
        p(out, f"[iter {it:3d}] |W|={len(W):4d}  bound={bd:12.5f} ({ratio:8.4f}x) {tag}  "
               f"viol={nv:4d}  worst {wn}  |tight|={len(tight_b)}  [{time.time()-t_start:.0f}s]")

        # termination: no violations up to L
        if nv == 0:
            if bd > RECORD_F:
                verdict = ('CERTIFICATE_CANDIDATE', (b0, bd, ratio), len(W))
                p(out, f"\n  *** ALL b_n>=0 for n<=L={L} with bound {ratio:.4f}x record (>1) ***")
                p(out, f"      => CERTIFICATE CANDIDATE (pending exact rigor pass + tail proof past n0).")
            else:
                verdict = ('CLEAN_BELOW_RECORD', (b0, bd, ratio), len(W))
                p(out, f"\n  All b_n>=0 for n<=L but bound {ratio:.4f}x record (<=1): feasible tail-safe")
                p(out, f"      point exists but is NOT >record. NO >record certificate at N=24.")
            break

        # time budget (check BEFORE next solve)
        if time.time() - t_start > BUDGET:
            verdict = ('BUDGET', (b0, bd, ratio), len(W))
            p(out, f"\n  [budget {BUDGET:.0f}s reached] stopping. current bound {ratio:.4f}x record, "
                   f"{nv} violations remain (worst n={worst_n}).")
            break

        # ---- cutting-plane update: add the most-violated new rows ----
        # In EXACT_MODE we KEEP the full working set (union) — no pruning — because exact enforcement
        # is monotone and (add-all) converges in ~1 round; pruning only matters for the mpmath path
        # where a bloated W slows the simplex. Either way we rescan ALL n<=L, so nothing is skipped.
        if EXACT_MODE:
            newW = set(W)
        else:
            newW = set(tight_b) | set(last_added)   # mpmath: keep active + last round (anti-cycle)
        last_added = []
        for _, nn, _ in viol:                     # viol sorted most-violated first
            if nn not in newW:
                newW.add(nn); ever_enforced.add(nn); last_added.append(nn)
            if len(last_added) >= ADD_PER_ITER:
                break
        W = newW
        it += 1
        if it > 600:
            verdict = ('MAXITER', (b0, bd, ratio), len(W))
            p(out, "\n  max iterations reached.")
            break

    # ---- summary + curve ----
    p(out, "\n" + "=" * 80)
    p(out, "BOUND-vs-ITERATION CURVE")
    p(out, "=" * 80)
    p(out, f"{'iter':>4} {'|W|':>5} {'bound':>13} {'ratio':>9} {'viol':>6} {'worst_n':>8}")
    for (i, w, bd, ra, nv, wn, wb) in curve:
        p(out, f"{i:>4} {w:>5} {bd:>13.5f} {ra:>9.4f}x {nv:>6} {str(wn):>8}")

    p(out, "\n" + "=" * 80)
    p(out, "VERDICT")
    p(out, "=" * 80)
    kind = verdict[0] if verdict else 'UNKNOWN'
    if verdict and verdict[1]:
        b0v, bdv, rav = verdict[1]
    else:
        b0v = bdv = rav = None
    if kind == 'CERTIFICATE_CANDIDATE':
        p(out, f"CERTIFICATE CANDIDATE at N=24, T=10: bound = {bdv:.6f} = {rav:.4f}x record, |W|={verdict[2]}.")
        p(out, "Requires: (rigor R1) exact re-verification a_n>=0 [10..28] and b_n>=0 [1..L] as EXACT")
        p(out, "rationals at this vertex; (rigor R2) the finite tail proof for n>L (CT Sec5: Eisenstein")
        p(out, "main term |e1|*COEF18*sigma_17(n) dominates the |cusp| Deligne bound C_S*sigma_0(n)*n^8.5).")
        # attempt the exact rigor pass R1 immediately (pure-Fraction ground-truth simplex).
        # Enforce the FULL union of every n ever violated (ever_enforced) so the certified vertex is
        # the true full-range optimum, not the pruned working-set one.
        p(out, "\n[R1] exact re-verification of the final vertex signs (pure-Fraction simplex, "
               f"enforcing all {len(ever_enforced)} ever-violated n) ...")
        st, x, b0, _tb = solve_working_set(GAr, GBr, r, ever_enforced, exact=True)
        a_ok = all(sum(GAr[m][l] * x[l] for l in range(r)) >= 0 for m in range(T, M + 1))
        a0 = sum(GAr[0][l] * x[l] for l in range(r))
        b_neg = [nn for nn in range(1, L + 1) if sum(GBr[nn][l] * x[l] for l in range(r)) < 0]
        p(out, f"     a_0 == 1 exact: {a0 == 1};  a_n>=0 for {T}<=n<=28: {a_ok};  "
               f"b_n<0 count for 1<=n<={L}: {len(b_neg)} {'(NONE — R1 PASS)' if not b_neg else b_neg[:10]}")
    elif kind == 'CLEAN_BELOW_RECORD':
        p(out, f"Feasible point with b_n>=0 through n<=L exists but bound = {rav:.4f}x record (<=1).")
        p(out, "=> Enforcing the finite tail drives the optimum to/BELOW the record. NO >record")
        p(out, f"   certificate at N=24. Limiting bound = {bdv:.6f} = {rav:.4f}x record.")
    elif kind == 'INFEASIBLE':
        p(out, f"The working-set LP became INFEASIBLE at |W|={verdict[2]} (enforcing b_n>=0 through the")
        p(out,  "   generated set is inconsistent with a_0=1, a_n>=0). => NO >record dual point at N=24,T=10")
        p(out,  "   that is finitely positive on the enforced set.")
    elif kind == 'BUDGET':
        p(out, f"Budget reached mid-iteration. Current bound = {bdv:.6f} = {rav:.4f}x record, |W|={verdict[2]}.")
        p(out, "   Objective is monotone non-increasing in |W| (each added cut only tightens); the curve")
        p(out, "   above is the partial descent. Verdict-so-far: bound still")
        p(out, f"   {'ABOVE' if rav and rav>1 else 'AT/BELOW'} record after enforcing {verdict[2]} cuts.")
    else:
        p(out, f"Terminated: {kind}. bound = {rav}x record, |W|={verdict[2] if verdict else '?'}.")

    p(out, f"\n[total time {time.time()-t_start:.0f}s]")
    return out, verdict, curve


if __name__ == "__main__":
    out, verdict, curve = main()
    receipt = os.path.join(CODE, 'receipt_d36_cutgen.txt')
    with open(receipt, 'w') as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[written] {receipt}")
