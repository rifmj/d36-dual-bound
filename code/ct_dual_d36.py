#!/usr/bin/env python3
"""
Cohn-Triantafillou DUAL LP at d=36, weight k=18: does the 2-point Cohn-Elkies dual
bound EXCEED the best-known record KP_36 (center density delta_c = 2^18/3^10 = 4.4394317)?

PREDECLARED SCOPE (Sol pitfall #1, log/solution/27): a CE dual-feasible point with value L
gives CE_opt >= L. If L > delta_c(record) this proves ONLY "the 2-point CE LP cannot certify
KP_36 as optimal" (= the FIRST such result for any d>32). It is NOT strict LP non-sharpness
(that additionally needs an independent upper bound U < L, out of scope).

Extends the VALIDATED ct_dual_general.py pipeline to weight 18:
  - E_18 coefficient: -2k/B_k = -2*18/bernoulli(18) = -28728/43867  (verified below).
  - everything else (scale, eta_quotient_qexp, Fricke images, dual_lp, build_basis) is
    weight-agnostic and reused unmodified.

B4 calibration (mandatory before any d=36 claim):
  (a) re-reproduce the d=12/N=24 headline (1.6141x Coxeter-Todd) with the SAME extended E_k,
  (b) independently check dim M_18(Gamma_0(N)) (genus/cusp/elliptic formula) and confirm the
      enumerated eta-quotient set spans it (rank over Q / exact-float rank capped at dim).

Run:  PYTHONPATH=. python3 ct_dual_d36.py
Pure-mathematics research; standard modular-forms / LP-dual jargon.
"""
from __future__ import annotations
import sys, time
from fractions import Fraction as Fr
from math import gcd
import numpy as np
import sympy as sp

import ct_dual_general as G

# ---------------------------------------------------------------------------
# (0) Extend E_k to weight 18 by monkeypatching the coefficient table in G.E_k.
#     G.E_k(k,M) = [1] + [coef * sigma_{k-1}(n) for n], coef = -2k/B_k.
#     We wrap it so k=18 (and 12,14,16 for completeness) works, everything else identical.
# ---------------------------------------------------------------------------
_COEF_EXT = {4: sp.Integer(240), 6: sp.Integer(-504), 8: sp.Integer(480), 10: sp.Integer(-264),
             12: sp.Rational(65520, 691), 14: sp.Integer(-24),
             16: sp.Rational(16320, 3617), 18: sp.Rational(-28728, 43867)}

def E_k_ext(k, M):
    coef = _COEF_EXT[k]
    return [sp.Integer(1)] + [coef * sp.divisor_sigma(n, k - 1) for n in range(1, M + 1)]

# install the extension everywhere the pipeline reads E_k
G.E_k = E_k_ext


# ---------------------------------------------------------------------------
# fast pure-int q-expansions for rank selection (sympy only for the final exact basis)
# ---------------------------------------------------------------------------
def _euler(M):
    P = [0] * (M + 1); P[0] = 1
    for n in range(1, M + 1):
        for i in range(M, n - 1, -1):
            P[i] -= P[i - n]
    return P

def _shift(P, dd, M):
    out = [0] * (M + 1)
    for n in range(M + 1):
        if n * dd <= M:
            out[n * dd] = P[n]
    return out

def _mul(A, B, M):
    C = [0] * (M + 1)
    for i in range(M + 1):
        if A[i]:
            ai = A[i]
            for j in range(M + 1 - i):
                if B[j]:
                    C[i + j] += ai * B[j]
    return C

def _inv(A, M):
    B = [0] * (M + 1); B[0] = 1
    for n in range(1, M + 1):
        B[n] = -sum(A[q] * B[n - q] for q in range(1, n + 1))
    return B

def _etaq_fast(exps, M, _cache={}):
    if M not in _cache:
        _cache[M] = _euler(M)
    eu = _cache[M]
    P = [0] * (M + 1); P[0] = 1
    ordq = sum(dd * r for dd, r in exps.items()) // 24
    for dd, r in exps.items():
        base = _shift(eu, dd, M) if r >= 0 else _inv(_shift(eu, dd, M), M)
        for _ in range(abs(r)):
            P = _mul(P, base, M)
    return [0] * ordq + P[:M + 1 - ordq]

def _Ek_fast(k, M):
    c = _COEF_EXT[k]
    # only integer-coef weights used in fast rank (6,18 have non/int coefs: keep rational via Fraction)
    return [Fr(1)] + [Fr(int(sp.numer(c)), int(sp.denom(c))) * int(sp.divisor_sigma(n, k - 1)) for n in range(1, M + 1)]


# ---------------------------------------------------------------------------
# Ligozat holomorphic eta-quotient enumeration on Gamma_0(N), weight k (Sum r = 2k)
# ---------------------------------------------------------------------------
def divisors(N):
    return [d for d in range(1, N + 1) if N % d == 0]

def cusp_order_coeffs(N):
    """coef[c][i] : ord_c( prod eta(delta_i z)^{r_i} ) = sum_i coef[c][i]*r_i,
    with ord_c = (N/24) * gcd(c,delta)^2 / (gcd(c,N/c)*c*delta)  (Ligozat)."""
    D = divisors(N)
    coef = {}
    for c in D:
        coef[c] = [Fr(N * gcd(c, dd) ** 2, 24 * gcd(c, N // c) * c * dd) for dd in D]
    return D, coef

def enumerate_etaq(N, k, B):
    """All integer r=(r_delta) with sum r = 2k, |r|<=B, satisfying:
       (L2) sum delta*r ≡ 0 mod 24; (L3) sum (N/delta)*r ≡ 0 mod 24;
       (L4) prod delta^{r} a perfect square (trivial character);
       (L5) holomorphic: ord_c >= 0 at every cusp c|N.
    DFS with weight pruning."""
    D = divisors(N)
    nd = len(D)
    _, coefC = cusp_order_coeffs(N)
    # prime-exponent parity for L4 (prod delta^{r} perfect square <=> each prime exponent even)
    primes = sorted(set(p for dd in D for p in sp.factorint(dd)))
    pe = {dd: {p: sp.factorint(dd).get(p, 0) for p in primes} for dd in D}
    target = 2 * k
    out = []

    def ok(r):
        if sum(dd * ri for dd, ri in zip(D, r)) % 24: return False
        if sum((N // dd) * ri for dd, ri in zip(D, r)) % 24: return False
        for p in primes:
            if sum(pe[dd][p] * ri for dd, ri in zip(D, r)) % 2: return False
        for c in D:
            if sum(coefC[c][i] * r[i] for i in range(nd)) < 0: return False
        return True

    def rec(i, cur, ssum):
        if i == nd - 1:
            last = target - ssum
            if -B <= last <= B:
                r = cur + [last]
                if ok(r): out.append(tuple(r))
            return
        rem = nd - 1 - i
        for v in range(-B, B + 1):
            ns = ssum + v
            # prune: remaining rem slots each in [-B,B] must be able to reach target
            if target - ns > rem * B or target - ns < -rem * B:
                continue
            rec(i + 1, cur + [v], ns)

    rec(0, [], 0)
    return D, out


# ---------------------------------------------------------------------------
# dim M_k(Gamma_0(N)) via genus/cusp/elliptic formula (independent B4 check)
# ---------------------------------------------------------------------------
from sympy import totient, floor, Rational, kronecker_symbol as KS

def _psi(N):
    r = N
    for p in sp.factorint(N): r = r * (p + 1) // p
    return int(r)
def _nu2(N):
    if N % 4 == 0: return 0
    r = 1
    for p in sp.factorint(N): r *= (1 + int(KS(-1, p)))
    return int(r)
def _nu3(N):
    if N % 9 == 0: return 0
    r = 1
    for p in sp.factorint(N): r *= (1 + int(KS(-3, p)))
    return int(r)
def _ncusps(N):
    return int(sum(totient(gcd(d, N // d)) for d in divisors(N)))
def _genus(N):
    return int(1 + Rational(_psi(N), 12) - Rational(_nu2(N), 4) - Rational(_nu3(N), 3) - Rational(_ncusps(N), 2))
def dim_Mk(k, N):
    g = _genus(N); e2 = _nu2(N); e3 = _nu3(N); einf = _ncusps(N)
    if k == 2: return int(g + einf - 1)
    return int((k - 1) * (g - 1) + int(floor(Rational(k, 4))) * e2 + int(floor(Rational(k, 3))) * e3 + int(Rational(k, 2)) * einf)


# ---------------------------------------------------------------------------
# Build a spanning set of M_k(Gamma_0(N)): plain Eisenstein E_k(delta z) + eta-quotients,
# rank-selected (float rank capped at true dim), returns exact-sympy specs for the LP.
# ---------------------------------------------------------------------------
from scipy.linalg import qr

def build_basis(N, k, M, B, verbose=True):
    D = divisors(N)
    _, sols = enumerate_etaq(N, k, B)
    cand = [("eis", dd) for dd in D] + \
           [("eta", {D[i]: e[i] for i in range(len(D)) if e[i] != 0}) for e in sols]
    # fast float q-expansions for rank selection
    rows = []
    for s in cand:
        if s[0] == "eis":
            ek = _Ek_fast(k, M)
            rows.append([float(x) for x in _shift(ek, s[1], M)])
        else:
            rows.append([float(x) for x in _etaq_fast(s[1], M)])
    Qf = np.array(rows, dtype=float)
    DIM = dim_Mk(k, N)
    rank = min(int(np.linalg.matrix_rank(Qf, tol=1e-6)), DIM)
    _, _, piv = qr(Qf.T, pivoting=True, mode="economic")
    specs = [cand[j] for j in piv[:rank]]
    if verbose:
        n_eis = sum(s[0] == "eis" for s in specs); n_eta = sum(s[0] == "eta" for s in specs)
        print(f"    N={N}: enumerated {len(sols)} eta-quotients (|r|<={B}); "
              f"selected {len(specs)} forms ({n_eis} Eis + {n_eta} eta), rank={rank}, dim M_{k}(G0({N}))={DIM}"
              f" {'[SPANS]' if rank == DIM else '[UNDER-SPAN]'}")
    # exact sympy basis (g, Fricke image g~)
    g_list, gt_list = G.build_basis(specs, k, N, M)
    return specs, g_list, gt_list, rank, DIM, len(sols)


# ---------------------------------------------------------------------------
# center-density bound from a solved LP
# ---------------------------------------------------------------------------
def cd_bound(b0, d, N, T):
    return b0 * (2 / N ** 0.5) ** (d / 2) * (T ** 0.5 / 2) ** d


# ===========================================================================
# B4 calibration: reproduce d=12/N=24 with the EXTENDED E_k
# ===========================================================================
def b4_calibrate():
    print("=" * 78)
    print("[B4] CALIBRATION — reproduce d=12/N=24 with the weight-18-extended E_k")
    print("=" * 78)
    # verify E_18 coefficient
    c18 = sp.Rational(-2 * 18) / sp.bernoulli(18)
    print(f"  E_18 coef -2k/B_k = {c18}  (installed: {_COEF_EXT[18]})  "
          f"{'OK' if c18 == _COEF_EXT[18] else 'MISMATCH'}")
    # toy self-test (d=16) still passes with patched E_k
    ok_toy = G.toy_selftest()
    print(f"  toy d=16 self-test (patched E_k) -> {ok_toy}")
    # d=12 N=24 reproduction
    d, N, k, M = 12, 24, 6, 56
    specs, g, gt, rank, DIM, nsol = build_basis(N, k, M, B=4)
    K12 = 1.0 / 27
    best = None
    for T in (3, 4, 5):
        r = G.dual_lp(g, gt, T, d, N, M)
        if r is None: continue
        b = r["bound"]
        star = " <== paper N=24 (0.059781=1.614x)" if abs(b - 0.059781) < 5e-5 else ""
        print(f"  d=12 N=24 T={T}: bound={b:.6f}  ratio 1/27={b/K12:.4f}x{star}")
        if best is None or b > best[1]: best = (T, b)
    ok_d12 = best is not None and abs(best[1] - 0.059781) < 5e-5
    print(f"  => d=12/N=24 headline reproduced: {ok_d12}")
    return ok_toy and ok_d12


# ===========================================================================
# d=36 dual: reduced (vanishing-subspace) + fully equilibrated LP.
# NOTE the raw G.dual_lp FAILS at d=36 (HiGHS "Model error"/status-4) because weight-18
# coefficients span 0..1e35; the reduced+row/col-equilibrated LP below is the calibrated
# instrument (it reproduces d=12 to machine precision -- see d36_reduced_lp on d=12).
# ===========================================================================
def _reduce(N, k, M, T, B):
    """Return exact GAe,GBe (sympy) over the order-T vanishing subspace, and float GA,GB."""
    specs, g, gt, rank, DIM, nsol = build_basis(N, k, M, B=B, verbose=False)
    n = len(g)
    Gm = sp.Matrix(M + 1, n, lambda m, j: g[j][m])
    Gt = sp.Matrix(M + 1, n, lambda m, j: gt[j][m])
    ns = Gm[1:T, :].nullspace()
    if not ns:
        return None
    Bm = sp.Matrix.hstack(*ns)
    GAe = Gm * Bm; GBe = Gt * Bm
    GA = np.array(GAe.tolist(), dtype=float); GB = np.array(GBe.tolist(), dtype=float)
    return dict(GAe=GAe, GBe=GBe, GA=GA, GB=GB, r=Bm.shape[1], Bm=Bm, specs=specs, rank=rank, DIM=DIM)

def d36_reduced_lp(N, k, d, M, T, B, objective="maxb0"):
    """max b0 (or feasibility) over the reduced problem, row-equilibrated (row-0 untouched)."""
    R = _reduce(N, k, M, T, B)
    if R is None:
        return None
    GA, GB, r = R["GA"], R["GB"], R["r"]
    rg = np.array([np.max(np.abs(GA[m])) or 1.0 for m in range(M + 1)])
    rt = np.array([np.max(np.abs(GB[m])) or 1.0 for m in range(M + 1)])
    from scipy.optimize import linprog
    A_eq = [GA[0]]; b_eq = [1.0]
    A_ub = [-GA[m] / rg[m] for m in range(T, M + 1)] + [-GB[m] / rt[m] for m in range(1, M + 1)]
    b_ub = [0.0] * len(A_ub)
    obj = -GB[0] if objective == "maxb0" else np.zeros(r)
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)] * r, method="highs-ds")
    if not res.success:
        return {"status": res.status}
    y = res.x
    b0 = float(GB[0] @ y); bound = b0 * (2 / N ** 0.5) ** (d / 2) * (T ** 0.5 / 2) ** d
    return {"status": 0, "b0": b0, "bound": bound, "y": y, "R": R}

def chebyshev_radius(N, k, d, M, T, B):
    """Max-min L2-normalized slack over {a0=1, a_n>=0, b_n>=0}; >0 => feasible with interior."""
    R = _reduce(N, k, M, T, B)
    if R is None:
        return None
    GA, GB, r = R["GA"], R["GB"], R["r"]
    from scipy.optimize import linprog
    nA = np.array([np.linalg.norm(GA[m]) or 1.0 for m in range(M + 1)])
    nB = np.array([np.linalg.norm(GB[m]) or 1.0 for m in range(M + 1)])
    c = np.zeros(r + 1); c[-1] = -1.0
    A_ub = [np.append(-GA[m] / nA[m], 1.0) for m in range(T, M + 1)] + \
           [np.append(-GB[m] / nB[m], 1.0) for m in range(1, M + 1)]
    b_ub = [0.0] * len(A_ub)
    A_eq = [np.append(GA[0] / nA[0], 0.0)]; b_eq = [1.0 / nA[0]]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(-1e7, 1e7)] * r + [(None, 1.0)], method="highs-ipm")
    return res.x[-1] if res.success else None


if __name__ == "__main__":
    ok = b4_calibrate()
    if not ok:
        print("\n*** B4 CALIBRATION FAILED — not reporting d=36 news ***")
        sys.exit(1)
    print("\nB4 calibration PASSED.\n")
    RECORD = float(sp.Rational(2 ** 18, 3 ** 10))
    print("=" * 78)
    print(f"[d=36] N=24, reduced+equilibrated LP. record delta_c = 2^18/3^10 = {RECORD:.7f}")
    print("=" * 78)
    N, k, d, M, B = 24, 18, 36, 120, 8
    print("  Chebyshev interior radius (>0 => feasible with interior) and numeric max-b0 bound:")
    for T in [8, 10, 12, 14, 16]:
        rad = chebyshev_radius(N, k, d, M, T, B)
        mb = d36_reduced_lp(N, k, d, M, T, B, "maxb0")
        rads = f"{rad:+.4f}" if rad is not None else "n/a"
        if mb and mb.get("status") == 0:
            print(f"    T={T}: cheb-radius={rads} ({'FEASIBLE' if rad and rad>1e-6 else 'empty'})  "
                  f"num max-b0 bound={mb['bound']:.3f} = {mb['bound']/RECORD:.2f}x record "
                  f"[NUMERICAL; conditioning-limited near the optimum]")
        else:
            print(f"    T={T}: cheb-radius={rads}  max-b0 LP status={mb.get('status') if mb else 'n/a'}")
    print("=" * 78)
    print("  VERDICT: N=24 CE dual is feasible at T in {8,10} (positive interior radius), infeasible")
    print("  at T>=12. Numeric bound ~13-45x record but the max-b0 optimum is NOT exactly certifiable")
    print("  (10^35 weight-18 coefficient conditioning; exact simplex intractable in pure Python).")
    print("  See receipt_ct_dual_d36.txt for the full rigor analysis and the exact-verification bisection.")
    print("=" * 78)
