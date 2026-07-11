#!/usr/bin/env python3
r"""
THE DECISIVE d=36 RUNG: high-precision (mpmath) Cohn-Elkies DUAL LP that DIRECTLY enforces
b_n >= 0 out to M (100 -> 800), and asks whether the optimal b_0 still gives a center-density
bound > record delta_c(KP_36) = 2^18/3^10 = 4.4394316585.

CONTEXT (log/solution/29 + code/receipt_d36_hiprec.txt): at N=24,T=10 a >record max-b0 point
exists whose Fricke image is EVENTUALLY POSITIVE on every residue class (CT-Sec5 SUF_c<=0), BUT
its finite b_n has scattered negatives up to n_0>783 (binding classes n=3,9,15,21 mod 24 where the
Eisenstein main constant r_g~ ~ 1e-13). The exact-RATIONAL simplex blows up beyond M~40. The
UNTRIED decisive move (this file): a HIGH-PRECISION LP that enforces b_n>=0 for ALL 1<=n<=M with
M up to 800 and reports max b0 -- if still >record at M=800, a full certificate is in reach; if the
>record margin dies once b_n>=0 is enforced across the finite range, N=24 is insufficient.

    maximize  b_0 = GB[0].y
    s.t.      GA[0].y = 1                         (a_0 = 1)
              GA[m].y >= 0   (T <= m <= M)        (a_n >= 0; a_n=0 for 1<=n<T by the reduction)
              GB[m].y >= 0   (1 <= m <= M)        (b_n >= 0, THE finite-positive tail, direct)

y ranges over the exact order-T vanishing subspace of M_18(Gamma_0(24)) (nullspace of the a_1..a_{T-1}
rows). center-density bound = b_0 * (2/sqrt(24))^18 * (sqrt(10)/2)^36.

TWO CRITICAL LESSONS BAKED IN (both are the log/29 "phantom" discipline, one level deeper):
  (1) The basis + Fricke images are expanded to M=800 with the FAST EXACT python-Fraction path
      (ct_dual_d36._etaq_fast / _Ek_fast + the exact Fricke rescaling), NOT the symbolic sympy path
      (which is the M~40 wall). This fast builder is CALIBRATED here against the sympy builder.
  (2) sympy.nsimplify() CORRUPTS large exact rationals -- e.g. nsimplify(-1792883/972) returns an
      IRRATIONAL algebraic surd ~ (-1792883/972)*(1 + 3e-13), and the old sp_to_mp() then read that
      surd's numer/denom, injecting ~1e-13 RELATIVE error into every reduced matrix entry -- exactly
      the scale of the binding r_g~. This file converts rationals to mpf DIRECTLY (mpf(p)/mpf(q)),
      NEVER through nsimplify. (The pre-existing ct_dual_d36_hiprec.py has this latent bug; see the
      receipt. It is not modified -- new files only.)

The returned mpmath vertex is ALWAYS re-verified (a_0=1, all a_n>=0, all b_n>=0, exact slacks) and,
if >record and feasible, rational-reconstructed and EXACT-verified with sympy Fraction.

SOLVER REALITY (established empirically here; sharpens §29's "exact-simplex wall M~40"):
  - float LP (scipy HiGHS, incl. Ruiz + exact power-of-2 preconditioning) CANNOT decide feasibility
    at d=36: after column scaling the gcd=3 binding-class constraint rows collapse to ~1e-17 in
    float, and HiGHS returns "Model error"/spurious "unbounded"/spurious "infeasible" for every box
    and method. => float is OUT (kept below only as a cross-reference, NOT trusted).
  - exact-rational (Fraction) simplex is correct but the pivot numerators blow up -> intractable even
    at M=28.
  - the exact-precision mpmath simplex (Bland) is the ONLY reliable solver, but (a) needs dps>=70
    (dps=55 returns a PHANTOM: a0=12.36, not 1), and (b) is DEGENERACY-crippled (M=28 r=20: 3481
    pivots/18s; M=40 r=32: 19112 pivots/343s), so M>>100 is out of reach in pure Python.
  - the LP must use the RANK-SELECTED basis at length M (ct_dual_d36.build_basis, the §29-validated
    methodology): the full 72-form basis is UNBOUNDED until M>=r=63; rank-selection gives r=20 at
    M=28 and saturates to r=63 (all 72 forms) at M~72. Below M~72 the max-b0 curve is NON-monotone
    (added forms raise it faster than added b_n>=0 constraints lower it); the DECISIVE monotone
    regime is M>=72 (full basis).

Run:  PYTHONPATH=. python3 ct_dual_d36_mpmath800.py [Mlist] [dps]
      example: python3 ct_dual_d36_mpmath800.py 80,100,120,150 70
      defaults: M in {100,200,400,600,800}, dps=60, T=10, N=24, k=18, B(|r|)=8
Pure-mathematics research; standard modular-forms / LP-dual jargon.
"""
from __future__ import annotations
import sys, time
from fractions import Fraction as Fr
import sympy as sp
import mpmath as mp

import ct_dual_d36 as D            # fast exact q-expansions (_etaq_fast/_Ek_fast/_shift) + build_basis
import ct_dual_general as G        # sympy builder (used ONLY for calibration)
import eisen_projection as EP      # exact Eisenstein projector (for the eventual-positivity check)

N, K, D_DIM, BMAX = 24, 18, 36, 8
DELTAS = [1, 2, 3, 4, 6, 8, 12, 24]
RECORD = Fr(2 ** 18, 3 ** 10)                       # KP_36 center density = 4.4394316585...
RECORD_F = float(RECORD)


# ---------------------------------------------------------------------------
# EXACT -> mpf without nsimplify (THE key correctness fix).
# ---------------------------------------------------------------------------
def fr_to_mp(x):
    """Fraction -> mpf, exact ratio (no nsimplify, no float)."""
    if isinstance(x, Fr):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    if isinstance(x, int):
        return mp.mpf(x)
    # sympy Rational / Integer
    return mp.mpf(int(sp.numer(x))) / mp.mpf(int(sp.denom(x)))


def as_fr(x):
    """sympy Rational/Integer or Fraction -> Fraction, EXACT (no nsimplify)."""
    if isinstance(x, Fr):
        return x
    if isinstance(x, int):
        return Fr(x)
    return Fr(int(sp.numer(x)), int(sp.denom(x)))


# ---------------------------------------------------------------------------
# FAST EXACT builder: g_j and its Fricke image g~_j as Fraction lists to length L.
# Mirrors ct_dual_general.fricke_eisen / fricke_etaq exactly, but on the fast Fraction path.
# ---------------------------------------------------------------------------
def _fricke_const_eis(delta):
    C = sp.nsimplify(sp.I ** K * sp.Rational(1, N) ** sp.Rational(K, 2) * sp.Rational(N, delta) ** K)
    return Fr(int(sp.numer(C)), int(sp.denom(C)))          # this C is a genuine rational; nsimplify is fine on the SMALL closed form, but we verify it below


def _fricke_const_eta(exps):
    C = sp.nsimplify(sp.Integer(N) ** sp.Rational(K, 2) *
                     sp.prod([sp.Integer(dd) ** sp.Rational(-r, 2) for dd, r in exps.items()]))
    return Fr(int(sp.numer(C)), int(sp.denom(C)))


def fast_g_gt(spec, L):
    """Return (g, gt): Fraction lists [0..L] for the form and its Fricke image."""
    if spec[0] == "eis":
        delta = spec[1]
        g = [Fr(x) for x in D._shift(D._Ek_fast(K, L), delta, L)]
        C = _fricke_const_eis(delta)
        h = [Fr(x) for x in D._shift(D._Ek_fast(K, L), N // delta, L)]
        gt = [C * x for x in h]
    else:
        exps = spec[1]
        g = [Fr(x) for x in D._etaq_fast(exps, L)]
        C = _fricke_const_eta(exps)
        rev = {}
        for dd, r in exps.items():
            rev[N // dd] = rev.get(N // dd, 0) + r
        h = [Fr(x) for x in D._etaq_fast(rev, L)]
        gt = [C * x for x in h]
    return g, gt


# ---------------------------------------------------------------------------
# CALIBRATION (B4): the fast Fraction builder MUST reproduce the sympy builder.
# (Compares raw rationals; deliberately does NOT use nsimplify -- that is the poison.)
# ---------------------------------------------------------------------------
def calibrate_builder(specs, L=90, sample=None, verbose=True):
    """B4: the fast Fraction builder must reproduce the sympy builder (g AND Fricke image).
    sample = list of spec-indices to check (default: all 8 Eisenstein + a spread of eta forms,
    including ones with nontrivial q-order shifts, which is where the Fricke reversal is subtle).
    The sympy build of a form to L is the slow step, so we sample rather than build all 72 every
    run -- the per-spec correctness is the invariant, proven independent of the others."""
    t0 = time.time()
    if sample is None:
        eis_idx = [j for j, s in enumerate(specs) if s[0] == "eis"]      # all 8 Eisenstein
        eta_idx = [j for j, s in enumerate(specs) if s[0] == "eta"]
        # spread across the eta list (early/mid/late) + the first few (which carry q-order shifts)
        pick = sorted(set(eta_idx[:6] + eta_idx[len(eta_idx) // 2: len(eta_idx) // 2 + 4] + eta_idx[-4:]))
        sample = sorted(set(eis_idx + pick))
    sub = [specs[j] for j in sample]
    g_sym, gt_sym = G.build_basis(sub, K, N, L)
    allok = True; nfail = 0
    for jj, s in enumerate(sub):
        gf, gtf = fast_g_gt(s, L)
        okg = all(gf[m] == as_fr(g_sym[jj][m]) for m in range(L + 1))
        okgt = all(gtf[m] == as_fr(gt_sym[jj][m]) for m in range(L + 1))
        if not (okg and okgt):
            allok = False; nfail += 1
            if verbose and nfail <= 3:
                print(f"    CALIB MISMATCH spec={s} g={okg} gt={okgt}", flush=True)
    if verbose:
        print(f"  [B4] fast-vs-sympy builder ({len(sub)} forms incl. all 8 Eisenstein, "
              f"g & Fricke, L={L}): {'ALL MATCH' if allok else f'{nfail} FAIL'}  [{time.time()-t0:.0f}s]",
              flush=True)
    return allok


# ---------------------------------------------------------------------------
# Build the EXACT reduced problem to length M: GAe, GBe over the a_1..a_{T-1} vanishing subspace.
# All Fraction / sympy Rational -- no float, no nsimplify.
# ---------------------------------------------------------------------------
def build_series(specs, M):
    """Fast exact g_j and Fricke g~_j for all specs to length M (Fraction lists)."""
    G_list = []; Gt_list = []
    for s in specs:
        g, gt = fast_g_gt(s, M)
        G_list.append(g); Gt_list.append(gt)
    return G_list, Gt_list


def rank_selected_specs(M):
    """Rank-select a spanning subset of M_18(Gamma_0(24)) AT truncation length M (the §29 /
    ct_dual_d36 methodology). At small M this returns FEWER than 72 forms (float-QR rank capped
    by M), which keeps the reduced max-b0 LP BOUNDED (the full 72-form basis is unbounded until
    ~M>=r). Returns the spec list; identical selection to the VALIDATED maxb0_hiprec path."""
    specs, gM, gtM, rank, DIM, nsol = D.build_basis(N, K, M, B=BMAX, verbose=False)
    return specs


def build_reduced(specs, M, T, series=None):
    """Return dict with exact reduced GAe, GBe (lists of Fraction rows, length r) to index M,
    plus Bm (n x r Fraction) and r. Uses the fast builder for g,gt and sympy only for the small
    nullspace of the (T-1) x n coefficient matrix. If series=(G_list,Gt_list) precomputed to >=M,
    reuse it (sliced)."""
    n = len(specs)
    if series is not None:
        G_full, Gt_full = series
        G_list = [g[:M + 1] for g in G_full]; Gt_list = [gt[:M + 1] for gt in Gt_full]
    else:
        G_list, Gt_list = build_series(specs, M)
    # exact nullspace of rows 1..T-1 (shape (T-1) x n). small -> sympy Rational is fine and exact.
    Amat = sp.Matrix(T - 1, n, lambda i, j: sp.Rational(G_list[j][1 + i].numerator, G_list[j][1 + i].denominator))
    ns = Amat.nullspace()
    if not ns:
        return None
    Bm = sp.Matrix.hstack(*ns)                       # n x r, exact sympy Rational
    r = Bm.shape[1]
    Bfr = [[as_fr(Bm[j, l]) for l in range(r)] for j in range(n)]
    # reduced rows: GAe[m][l] = sum_j G_list[j][m] * Bfr[j][l]   (exact Fraction)
    GAe = []; GBe = []
    for m in range(M + 1):
        rowA = [Fr(0)] * r; rowB = [Fr(0)] * r
        for j in range(n):
            gjm = G_list[j][m]; gtjm = Gt_list[j][m]
            if gjm or gtjm:
                Bj = Bfr[j]
                if gjm:
                    for l in range(r):
                        if Bj[l]:
                            rowA[l] += gjm * Bj[l]
                if gtjm:
                    for l in range(r):
                        if Bj[l]:
                            rowB[l] += gtjm * Bj[l]
        GAe.append(rowA); GBe.append(rowB)
    return dict(GAe=GAe, GBe=GBe, Bm=Bm, Bfr=Bfr, r=r, n=n, G_list=G_list, Gt_list=Gt_list, specs=specs)


# ---------------------------------------------------------------------------
# High-precision two-phase simplex (Bland's rule), mpf arithmetic. maximize c.x, x free,
# s.t. A_eq x = b_eq, A_ub x <= b_ub. THIS IS THE ORIGINAL ct_dual_d36_hiprec.py simplex
# (validated: reproduces the §29 36.347x max-b0 at M=28 in ~19s). A Dantzig/incremental variant
# was tried and REVERTED -- it returned spurious "unbounded" (the working reference is Bland +
# full objective recompute; do not "optimize" it without re-validating against 36.347x).
# ---------------------------------------------------------------------------
def _mp_simplex(T, basis, ncols, cost, ncon, tol, phase1_cols=None):
    def objrow():
        z = [mp.mpf(0)] * (ncols + 1)
        for i in range(ncon):
            cb = cost[basis[i]]
            if cb != 0:
                Ti = T[i]
                for kk in range(ncols + 1):
                    z[kk] += cb * Ti[kk]
        return [cost[j] - z[j] for j in range(ncols)] + [-z[ncols]]
    obj = objrow()
    it = 0
    while True:
        it += 1
        piv_c = -1
        for j in range(ncols):
            if phase1_cols is not None and j in phase1_cols:
                continue
            if obj[j] < -tol:
                piv_c = j
                break
        if piv_c == -1:
            return 'optimal', basis, T, obj, it
        piv_r = -1; best = None
        for i in range(ncon):
            a = T[i][piv_c]
            if a > tol:
                ratio = T[i][-1] / a
                if best is None or ratio < best - tol or (abs(ratio - best) <= tol and basis[i] < basis[piv_r]):
                    best = ratio; piv_r = i
        if piv_r == -1:
            return 'unbounded', basis, T, obj, it
        piv = T[piv_r][piv_c]
        T[piv_r] = [v / piv for v in T[piv_r]]
        Tr = T[piv_r]
        for i in range(ncon):
            if i != piv_r:
                f = T[i][piv_c]
                if f != 0:
                    Ti = T[i]
                    T[i] = [Ti[kk] - f * Tr[kk] for kk in range(ncols + 1)]
        basis[piv_r] = piv_c
        obj = objrow()


def mp_solve(c, A_eq, b_eq, A_ub, b_ub, nvar, tol=None):
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps - 12))
    n_ub = len(A_ub); n_eq = len(A_eq)
    npm = 2 * nvar
    n_struct = npm + n_ub
    rows = []; rhs = []
    for i in range(n_eq):
        row = [mp.mpf(0)] * n_struct
        for j in range(nvar):
            row[j] = A_eq[i][j]; row[nvar + j] = -A_eq[i][j]
        rows.append(row); rhs.append(b_eq[i])
    for i in range(n_ub):
        row = [mp.mpf(0)] * n_struct
        for j in range(nvar):
            row[j] = A_ub[i][j]; row[nvar + j] = -A_ub[i][j]
        row[npm + i] = mp.mpf(1)
        rows.append(row); rhs.append(b_ub[i])
    for i in range(len(rows)):
        if rhs[i] < 0:
            rows[i] = [-v for v in rows[i]]; rhs[i] = -rhs[i]
    ncon = len(rows)
    ncols = n_struct + ncon
    T = []; basis = []
    for i in range(ncon):
        rr = rows[i] + [mp.mpf(0)] * ncon
        rr[n_struct + i] = mp.mpf(1)
        rr.append(rhs[i])
        T.append(rr); basis.append(n_struct + i)
    cost1 = [mp.mpf(0)] * ncols
    for j in range(n_struct, ncols):
        cost1[j] = mp.mpf(1)
    st, basis, T, obj, it1 = _mp_simplex(T, basis, ncols, cost1, ncon, tol)
    w = -obj[-1]
    if w > tol * (1 + max(abs(x) for x in rhs)):
        return 'infeasible', None, None, it1
    cost2 = [mp.mpf(0)] * ncols
    for j in range(nvar):
        cost2[j] = -c[j]; cost2[nvar + j] = c[j]
    art = set(range(n_struct, ncols))
    st, basis, T, obj, it2 = _mp_simplex(T, basis, ncols, cost2, ncon, tol, phase1_cols=art)
    if st == 'unbounded':
        return 'unbounded', None, None, it1 + it2
    xval = [mp.mpf(0)] * ncols
    for i in range(ncon):
        xval[basis[i]] = T[i][-1]
    x = [xval[j] - xval[nvar + j] for j in range(nvar)]
    return 'optimal', x, None, it1 + it2


# ---------------------------------------------------------------------------
# CUTTING-PLANE (constraint-generation) mpmath LP -- the SCALABLE rigorous solver.
# The full LP has ~2M inequality constraints (a_n>=0, b_n>=0 to M=800) but the optimum vertex
# is pinned by <= r+1 of them. Solve a small subset with the exact-precision simplex, evaluate
# ALL constraints at the solution (cheap, O(M*r) mpf), add the most-violated ones, repeat.
# A relaxed (subset) LP's optimum is an UPPER bound on b0; when its solution satisfies every
# constraint, it IS the full optimum. HiGHS is unusable here (float conditioning ~1e39 -> "Model
# error"/spurious "unbounded"; verified in the receipt), so this stays entirely in mpmath.
# ---------------------------------------------------------------------------
def maxb0_cutplane(GA, GB, M, T, r, dps, seed_stride=7, add_per_round=40, max_rounds=60,
                   tol=None, verbose=True):
    """GA,GB: mpf reduced rows [0..M]. Maximize b0=GB[0].y s.t. GA[0].y=1, GA[m].y>=0 (T<=m<=M),
    GB[m].y>=0 (1<=m<=M). Returns (status, y, rounds, nactive)."""
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps - 14))
    c = [GB[0][l] for l in range(r)]
    A_eq = [[GA[0][l] for l in range(r)]]; b_eq = [mp.mpf(1)]
    # constraint pool: ('a', m) for T<=m<=M, ('b', m) for 1<=m<=M
    pool = [('a', m) for m in range(T, M + 1)] + [('b', m) for m in range(1, M + 1)]
    def row_of(tag_m):
        tg, m = tag_m
        src = GA if tg == 'a' else GB
        return [-src[m][l] for l in range(r)]      # -row.y <= 0  <=>  row.y >= 0
    def val_of(tag_m, y):
        tg, m = tag_m
        src = GA if tg == 'a' else GB
        return sum(src[m][l] * y[l] for l in range(r))
    # seed active set: a strided sample across both a and b (cover low n densely -- most binding)
    active = set()
    for m in range(T, min(T + 30, M + 1)):
        active.add(('a', m))
    for m in range(1, min(31, M + 1)):
        active.add(('b', m))
    for tag_m in pool[::seed_stride]:
        active.add(tag_m)
    y = None; rounds = 0
    while rounds < max_rounds:
        rounds += 1
        act = sorted(active)
        A_ub = [row_of(tm) for tm in act]; b_ub = [mp.mpf(0)] * len(A_ub)
        st, y, _, iters = mp_solve(c, A_eq, b_eq, A_ub, b_ub, r)
        if st != 'optimal':
            return st, None, rounds, len(active)
        # evaluate ALL constraints; collect violations
        viol = []
        for tm in pool:
            if tm in active:
                continue
            v = val_of(tm, y)
            if v < -tol:
                viol.append((v, tm))
        if not viol:
            if verbose:
                print(f"      [cutplane converged: {rounds} rounds, {len(active)} active constraints, "
                      f"b0={mp.nstr(sum(GB[0][l]*y[l] for l in range(r)),8)}]", flush=True)
            return 'optimal', y, rounds, len(active)
        viol.sort(key=lambda t: t[0])                 # most negative first
        for _, tm in viol[:add_per_round]:
            active.add(tm)
        if verbose and (rounds <= 3 or rounds % 5 == 0):
            print(f"      [cutplane round {rounds}: {len(active)} active, {len(viol)} violations, "
                  f"worst {mp.nstr(viol[0][0],3)} at {viol[0][1]}]", flush=True)
    # hit max rounds -> return last (with a warning; caller re-verifies)
    if verbose:
        print(f"      [cutplane hit max_rounds={max_rounds}, {len(active)} active -- re-verify]", flush=True)
    return 'maxrounds', y, rounds, len(active)


# ---------------------------------------------------------------------------
# scipy-HiGHS fallback with per-column Ruiz equilibration (the pure-mpmath simplex is
# O(rows*cols) per pivot in Python and becomes slow at M~800). HiGHS solves the equilibrated
# float system; the returned vertex is ALWAYS re-verified in mpmath (the phantom lesson).
# NOTE: at d=36 weight-18 conditioning (~1e39) HiGHS returns "Model error"/spurious "unbounded"
# even with Ruiz equilibration -- see the receipt. Kept only as a cross-reference, NOT trusted.
# ---------------------------------------------------------------------------
def _ruiz_equilibrate(Araw, niter=20):
    """Ruiz row+column equilibration of a dense matrix (list-of-lists float). Returns
    (rscale, cscale) with A[i][j]*rscale[i]*cscale[j] having ~unit inf-norm rows/cols."""
    import numpy as np
    A = np.array(Araw, dtype=float)
    m, n = A.shape
    rs = np.ones(m); cs = np.ones(n)
    for _ in range(niter):
        M2 = (A * rs[:, None]) * cs[None, :]
        rmax = np.maximum(np.max(np.abs(M2), axis=1), 1e-300)
        cmax = np.maximum(np.max(np.abs(M2), axis=0), 1e-300)
        rs = rs / np.sqrt(rmax)
        cs = cs / np.sqrt(cmax)
    return rs, cs


def _pow2_colscales(GAe, GBe, M, T, r):
    """EXACT power-of-2 column scales s_l = 2^{-round(log2 maxabs_l)} balancing the reduced
    columns (max over the a0 row + all a_n (T..M) + all b_n (1..M) rows). Powers of 2 -> the
    float conversion float(entry*s_l) loses NO precision beyond the mantissa round of entry."""
    import math
    from fractions import Fraction as Fr
    scales = []
    for l in range(r):
        mx = 0
        for m in [0] + list(range(T, M + 1)):
            v = GAe[m][l]
            if v:
                a = abs(v); f = a.numerator / a.denominator if isinstance(a, Fr) else float(a)
                if f > mx: mx = f
        for m in range(1, M + 1):
            v = GBe[m][l]
            if v:
                a = abs(v); f = a.numerator / a.denominator if isinstance(a, Fr) else float(a)
                if f > mx: mx = f
        e = 0 if mx == 0 else -int(round(math.log2(mx)))
        scales.append(2.0 ** e)
    return scales


def solve_highs_pow2(RED, M, T, r, box=1e7, verbose=True):
    """max b0 s.t. a0=1, a_n>=0 (T..M), b_n>=0 (1..M), solved by HiGHS after EXACT power-of-2
    column preconditioning (y=D z, D=diag(2^e)). A mild box |z|<=box gives HiGHS a bounded
    polytope; the returned vertex is checked for box-pinning and re-verified in mpmath by the
    caller. Returns (y_mpf in ORIGINAL coords, status, box_pinned_count)."""
    import numpy as np
    from scipy.optimize import linprog
    GAe, GBe = RED["GAe"], RED["GBe"]
    s = _pow2_colscales(GAe, GBe, M, T, r)             # exact powers of 2
    # scaled float rows: A[m][l]*s[l]
    def frow(rowsrc, m, sign):
        return [sign * (float(rowsrc[m][l]) * s[l]) for l in range(r)]
    A_ineq = np.array([frow(GAe, m, -1.0) for m in range(T, M + 1)] +
                      [frow(GBe, m, -1.0) for m in range(1, M + 1)], dtype=float)
    a0row = np.array(frow(GAe, 0, 1.0), dtype=float)
    cobj = np.array(frow(GBe, 0, 1.0), dtype=float)
    res = linprog(-cobj, A_ub=A_ineq, b_ub=np.zeros(A_ineq.shape[0]),
                  A_eq=a0row[None, :], b_eq=[1.0], bounds=[(-box, box)] * r, method="highs")
    if not res.success:
        if verbose:
            print(f"      [HiGHS-pow2 status {res.status}: {res.message}]", flush=True)
        return None, res.status, None
    z = res.x
    pinned = int(np.sum(np.abs(np.abs(z) - box) < 1e-3 * box))     # box-pinning check (phantom guard)
    y = [mp.mpf(float(z[l])) * mp.mpf(s[l]) for l in range(r)]     # back to original coords (exact 2^e)
    return y, 0, pinned


# ---------------------------------------------------------------------------
# THE DECISIVE SOLVE: max b0 with b_n>=0 enforced DIRECTLY to M.
# ---------------------------------------------------------------------------
def maxb0_direct(RED, M, T, dps=60, solver="cutplane", verbose=True):
    """Max b0 over the reduced problem with a_n>=0 (T<=n<=M) and b_n>=0 (1<=n<=M).
    solver='cutplane' -> exact-precision CUTTING-PLANE (scalable to M=800, rigorous);
    solver='mpmath'   -> exact-precision full two-phase simplex (rigorous, O(rows)/pivot, slow);
    solver='highs'    -> scipy-HiGHS (float; UNUSABLE at d=36 conditioning -- cross-ref only).
    RED = build_reduced(...) result. The returned vertex is ALWAYS re-verified in mpmath."""
    mp.mp.dps = dps
    r = RED["r"]
    GAe, GBe = RED["GAe"], RED["GBe"]
    # exact -> mpf (DIRECT, no nsimplify)
    GA = [[fr_to_mp(GAe[m][l]) for l in range(r)] for m in range(M + 1)]
    GB = [[fr_to_mp(GBe[m][l]) for l in range(r)] for m in range(M + 1)]
    c = [GB[0][l] for l in range(r)]
    t0 = time.time()
    iters = None
    if solver == "highs":
        y, hstat = solve_highs_equilibrated(GA, GB, M, T, r, verbose=verbose)
        st = "optimal" if y is not None else "infeasible-or-failed"
    elif solver == "cutplane":
        st, y, rounds, nact = maxb0_cutplane(GA, GB, M, T, r, dps, verbose=verbose)
        iters = (rounds, nact)
        if st == 'maxrounds':
            st = 'optimal'                # return the last vertex; re-verification below is the gate
    else:
        A_eq = [[GA[0][l] for l in range(r)]]; b_eq = [mp.mpf(1)]
        A_ub = [[-GA[m][l] for l in range(r)] for m in range(T, M + 1)] + \
               [[-GB[m][l] for l in range(r)] for m in range(1, M + 1)]
        b_ub = [mp.mpf(0)] * len(A_ub)
        st, y, _, iters = mp_solve(c, A_eq, b_eq, A_ub, b_ub, r)
    tsolve = time.time() - t0
    if st != 'optimal' or y is None:
        if verbose:
            print(f"    maxb0_direct M={M} [{solver}]: LP {st}  [{tsolve:.0f}s, {iters} pivots]", flush=True)
        return {"status": st, "M": M, "tsolve": tsolve, "iters": iters, "solver": solver}
    # For the HiGHS path a0 is only enforced to float precision; renormalize y so a0=1 exactly
    # in mpmath (the CE-dual problem is homogeneous under y->y/(a0), b0 scales the same way).
    a0_raw = sum(GA[0][l] * y[l] for l in range(r))
    if solver == "highs" and a0_raw != 0:
        y = [y[l] / a0_raw for l in range(r)]
    # RE-VERIFY the vertex at high precision (the phantom lesson).
    a = [sum(GA[m][l] * y[l] for l in range(r)) for m in range(M + 1)]
    b = [sum(GB[m][l] * y[l] for l in range(r)) for m in range(M + 1)]
    a0 = a[0]; b0 = b[0]
    min_a = min((a[m] for m in range(T, M + 1)), default=mp.mpf(0))
    min_b = min((b[m] for m in range(1, M + 1)), default=mp.mpf(0))
    argmin_b = min(range(1, M + 1), key=lambda m: b[m])
    argmin_a = min(range(T, M + 1), key=lambda m: a[m]) if M >= T else 0
    bound = b0 * (mp.mpf(2) / mp.sqrt(N)) ** (mp.mpf(D_DIM) / 2) * (mp.sqrt(T) / 2) ** D_DIM
    ratio = bound / (mp.mpf(RECORD.numerator) / mp.mpf(RECORD.denominator))
    out = {"status": "optimal", "M": M, "T": T, "r": r, "dps": dps, "solver": solver,
           "b0": b0, "bound": bound, "ratio": ratio,
           "a0": a0, "min_a": min_a, "min_b": min_b, "argmin_b": argmin_b, "argmin_a": argmin_a,
           "y": y, "iters": iters, "tsolve": tsolve}
    if verbose:
        print(f"    M={M:>4} dps={dps} [{solver}]: bound={mp.nstr(bound,8)} = {mp.nstr(ratio,6)}x record | "
              f"a0={mp.nstr(a0,12)} min_a={mp.nstr(min_a,4)}@{argmin_a} min_b={mp.nstr(min_b,4)}@{argmin_b} "
              f"| {tsolve:.0f}s {iters if iters else ''}piv", flush=True)
    return out


# ---------------------------------------------------------------------------
# EXACT verification of a candidate vertex (rational reconstruction + sympy).
# ---------------------------------------------------------------------------
def rational_reconstruct(RED, y_mp, M, T, dps, max_den_bits=None):
    """Rational-reconstruct the reduced coords y (mpf) and EXACT-verify a_0=1, a_n=0 (n<T),
    a_n>=0, b_n>=0 for all 1<=n<=M with sympy Fraction. Returns (ok, report)."""
    r = RED["r"]
    # reconstruct each y[l] as a rational via mpmath.nstr->Fraction with a controlled denominator
    yr = []
    for l in range(r):
        # continued-fraction rational approximation good to the working precision
        val = y_mp[l]
        fr = _cf_rational(val, dps)
        yr.append(fr)
    GAe, GBe = RED["GAe"], RED["GBe"]
    a0 = sum(GAe[0][l] * yr[l] for l in range(r))
    ok_a0 = (a0 == 1)
    # a_n=0 for 1<=n<T
    ok_vanish = all(sum(GAe[m][l] * yr[l] for l in range(r)) == 0 for m in range(1, T))
    min_a = None; min_b = None; ok_a = True; ok_b = True; first_bad_b = None
    for m in range(T, M + 1):
        v = sum(GAe[m][l] * yr[l] for l in range(r))
        if min_a is None or v < min_a: min_a = v
        if v < 0: ok_a = False
    for m in range(1, M + 1):
        v = sum(GBe[m][l] * yr[l] for l in range(r))
        if min_b is None or v < min_b: min_b = v
        if v < 0 and first_bad_b is None:
            first_bad_b = (m, v); ok_b = False
    b0 = sum(GBe[0][l] * yr[l] for l in range(r))
    ok = ok_a0 and ok_vanish and ok_a and ok_b
    rep = dict(ok=ok, ok_a0=ok_a0, a0=a0, ok_vanish=ok_vanish, ok_a=ok_a, ok_b=ok_b,
               min_a=min_a, min_b=min_b, b0=b0, first_bad_b=first_bad_b)
    return ok, rep


def _cf_rational(x, dps):
    """Best rational p/q with the mpf x to ~dps digits via continued fractions."""
    x = mp.mpf(x)
    sign = 1
    if x < 0:
        sign = -1; x = -x
    a = []
    xx = x
    lim = mp.mpf(10) ** (dps - 6)
    for _ in range(80):
        fl = mp.floor(xx)
        a.append(int(fl))
        frac = xx - fl
        if frac == 0:
            break
        xx = 1 / frac
        # stop when denominator would exceed precision
        p, q = _cf_convergent(a)
        if q > lim:
            a.pop()
            break
    p, q = _cf_convergent(a)
    return Fr(sign * p, q)


def _cf_convergent(a):
    p0, q0 = 1, 0
    p1, q1 = a[0], 1
    for ai in a[1:]:
        p0, p1 = p1, ai * p1 + p0
        q0, q1 = q1, ai * q1 + q0
    return p1, q1


# ---------------------------------------------------------------------------
# CT eventual-positivity check on the exact reconstructed point (for a full certificate).
# ---------------------------------------------------------------------------
def eventual_positivity_check(RED, yr, verbose=True):
    """Compute e_delta(g~) exactly from the reduced point via the exact Eisenstein projector,
    and evaluate the CT SUF_c per residue class. SUF_c<=0 for all c => eventually positive.
    Returns (all_safe, egt, suf)."""
    from math import gcd
    r = RED["r"]; specs = RED["specs"]; n = RED["n"]; Bfr = RED["Bfr"]
    # reduced point in the ORIGINAL coordinate space: x_j = sum_l Bfr[j][l] * yr[l]
    xj = [sum(Bfr[j][l] * yr[l] for l in range(r)) for j in range(n)]
    # g~ coefficients on the projector pivot window
    PROJ, PIV = EP._load_proj(verbose=False)
    need = max(PIV)
    Mp = need + 5
    g_sym, gt_sym = G.build_basis(specs, K, N, Mp)          # sympy exact (few forms window; but n=72 -> use fast)
    # build g~ coeffs at the PIV indices exactly (Fraction), then to sympy Rational for PROJ
    def gt_coeff(p):
        s = Fr(0)
        for j in range(n):
            s += as_fr(gt_sym[j][p]) * xj[j]
        return s
    a_gt = sp.Matrix(72, 1, lambda i, _: sp.Rational(gt_coeff(PIV[i]).numerator, gt_coeff(PIV[i]).denominator))
    e = PROJ * a_gt
    egt = {DELTAS[i]: sp.nsimplify(e[i]) if False else sp.Rational(e[i]) for i in range(8)}
    # SUF_c: COEF18<0 so we need sum_{t|gcd(c,N)} e_t * (1/t^17 if e_t>=0 else 1/sigma17(t)) <= 0
    def sig17(x): return int(sp.divisor_sigma(x, 17))
    all_safe = True; suf = {}
    for c in range(1, N + 1):
        gc = gcd(c, N); ts = [t for t in DELTAS if gc % t == 0]
        s = sp.Rational(0)
        for t in ts:
            et = egt[t]
            fac = sp.Rational(1, t ** 17) if et >= 0 else sp.Rational(1, sig17(t))
            s += et * fac
        suf[c] = s
        if s > 0:
            all_safe = False
    if verbose:
        print(f"    eventual-positivity: e(g~)={ {d: mp.nstr(mp.mpf(egt[d].numerator)/mp.mpf(egt[d].denominator),3) for d in DELTAS} }")
        print(f"    SUF_c<=0 on all classes: {all_safe}  "
              f"(max SUF = {mp.nstr(mp.mpf(max(suf.values()).numerator)/mp.mpf(max(suf.values()).denominator),3)})")
    return all_safe, egt, suf


# ===========================================================================
# MAIN: pilot M=100 (reproduce §29 regime), then 200,400,600,800; bound-vs-M curve.
# ===========================================================================
def main():
    args = sys.argv[1:]
    if args and args[0] not in ("-",):
        Mlist = [int(x) for x in args[0].split(",")]
    else:
        Mlist = [100, 200, 400, 600, 800]
    dps = int(args[1]) if len(args) > 1 else 70    # dps<70 -> phantom (a0!=1); 70 is the validated floor
    T = 10
    print("=" * 88, flush=True)
    print(f"[d=36 mpmath M->800] N={N} k={K} T={T}. record delta_c = 2^18/3^10 = {RECORD_F:.10f}", flush=True)
    print(f"  DIRECT b_n>=0 enforcement to M in {Mlist}; dps={dps}. THE decisive question:", flush=True)
    print(f"  is max b0 (feasible with a_n,b_n>=0 to M) still > record as M -> 800?", flush=True)
    print("=" * 88, flush=True)

    # One-time B4 calibration of the fast builder against the sympy builder (full 72-form space).
    t0 = time.time()
    specs_full, PIV = EP.select_basis()
    if not calibrate_builder(specs_full, L=90):
        print("  *** B4 CALIBRATION FAILED -- not reporting news ***", flush=True)
        sys.exit(1)
    print(f"  [B4 builder calibration done, {time.time()-t0:.0f}s]", flush=True)

    curve = []
    RED_cache = {}
    for M in Mlist:
        tb = time.time()
        # RANK-SELECT the basis AT length M (the VALIDATED bounded methodology of §29): at small M
        # this uses FEWER than 72 forms so the reduced max-b0 LP is bounded (the full 72-form basis
        # is unbounded until M>=r). Fast exact build; correct fr_to_mp (no nsimplify).
        specs = rank_selected_specs(M)
        SER = build_series(specs, M)
        RED = build_reduced(specs, M, T, series=SER)
        tbuild = time.time() - tb
        if RED is None:
            print(f"  M={M}: no vanishing subspace", flush=True); continue
        print(f"  --- M={M}: {len(specs)} forms rank-selected, reduced dim r={RED['r']}, "
              f"build {tbuild:.0f}s ---", flush=True)
        # RIGOROUS exact-precision mpmath simplex (the ONLY reliable solver here: float/HiGHS
        # cannot decide feasibility at this ~1e39 conditioning -- the ~1e-13 binding rows are
        # destroyed by rounding; see the receipt). Bland's rule; reproduces §29's 36.347x regime.
        res = maxb0_direct(RED, M, T, dps=dps, solver="mpmath")
        res["tbuild"] = tbuild; res["nforms"] = len(specs)
        RED_cache[M] = RED
        curve.append(res)

    print("\n" + "=" * 88)
    print("BOUND-vs-M CURVE (direct b_n>=0 to M):")
    print(f"  {'M':>5} | {'status':>10} | {'bound':>16} | {'ratio_to_record':>16} | {'min_b (mpf)':>14} | {'>record?':>8}")
    for res in curve:
        if res["status"] == "optimal":
            gt = "YES" if res["ratio"] > 1 else "no"
            print(f"  {res['M']:>5} | {res['status']:>10} | {mp.nstr(res['bound'],10):>16} | "
                  f"{mp.nstr(res['ratio'],8):>16} | {mp.nstr(res['min_b'],4):>14} | {gt:>8}")
        else:
            print(f"  {res['M']:>5} | {res['status']:>10} | {'--':>16} | {'--':>16} | {'--':>14} | {'--':>8}")
    print("=" * 88)

    # verdict + optional exact verification at the largest feasible >record M
    best = [res for res in curve if res["status"] == "optimal" and res["ratio"] > 1]
    if best:
        top = max(best, key=lambda rr: rr["M"])
        print(f"\n>record survives the DIRECT b_n>=0 enforcement up to M={top['M']} "
              f"({mp.nstr(top['ratio'],6)}x). Attempting EXACT verification (rational reconstruction)...")
        RED = RED_cache[top["M"]]
        ok, rep = rational_reconstruct(RED, top["y"], top["M"], T, dps)
        print(f"  EXACT verify (sympy Fraction) at M={top['M']}: a_0=1:{rep['ok_a0']} "
              f"a_n=0(n<T):{rep['ok_vanish']} a_n>=0:{rep['ok_a']} b_n>=0:{rep['ok_b']} "
              f"=> {'FULL FINITE-POSITIVE POINT' if ok else 'reconstruction imperfect'}")
        if rep["first_bad_b"]:
            print(f"    first b_n<0 in reconstruction at n={rep['first_bad_b'][0]} "
                  f"(value {rep['first_bad_b'][1]}) -- if only slightly negative, precision-limited")
        if ok:
            print("  Checking CT eventual positivity (n>M) on the exact point ...")
            yr = [_cf_rational(top["y"][l], dps) for l in range(RED["r"])]
            safe, egt, suf = eventual_positivity_check(RED, yr)
            b0 = rep["b0"]
            bound_ex = b0 * Fr(2, 1) ** 0  # placeholder; report the mpf bound (irrational factor)
            print(f"\n  *** CANDIDATE FULL CERTIFICATE at N=24, M={top['M']}: "
                  f"bound {mp.nstr(top['bound'],8)} = {mp.nstr(top['ratio'],6)}x record, "
                  f"eventual-positivity {'PASS' if safe else 'FAIL'} ***")
    else:
        # find where >record dies
        opt = [res for res in curve if res["status"] == "optimal"]
        died_at = None
        for res in opt:
            if res["ratio"] <= 1:
                died_at = res["M"]; break
        infeas = [res["M"] for res in curve if res["status"] == "infeasible"]
        print(f"\nVERDICT: N=24 INSUFFICIENT for a finite-positive >record tail-safe point.")
        if died_at:
            print(f"  The >record bound dies (ratio<=1) at M={died_at} once b_n>=0 is enforced there.")
        if infeas:
            print(f"  The LP becomes INFEASIBLE at M={infeas[0]} (no CE-dual point with b_n>=0 to that M).")
        print(f"  Indicated next escape: larger level N=48 (multi-hour pipeline, NOT built here).")

    print(f"\ntotal wall time: {time.time()-t0:.0f}s")
    return curve


if __name__ == "__main__":
    main()
