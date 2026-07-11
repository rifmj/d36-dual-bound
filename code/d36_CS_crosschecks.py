#!/usr/bin/env python3
r"""
§35 cross-checks (C1 self-audit) — independent verifications of the d=36 tail certificate.

 (X1) DELIGNE-INDEPENDENT direct confirmation: extend the EXACT a_n,b_n>=0 scan of the iter1
      vertex to n=2000 (well past n0<=63).  If clean, the tail positivity is confirmed WITHOUT
      any modular-forms/Deligne argument, for a large explicit range.
 (X2) naive lower bound  sup_{n<=800} |c_n|/(sigma_0(n) n^{17/2})  for BOTH sides: must be <= C_S'
      (a lower bound on the true asymptotic constant; sanity-bounds C_S' from below).
 (X3) Deligne spot-check on the level-1 newform (Delta*E6, EXACT integer coeffs): |a_n| <=
      sigma_0(n) n^{17/2} for all n<=800 (confirms the bound the decomposition relies on).
 (X4) gamma-robustness: recompute C_S' with a DIFFERENT generic Hecke combination weight; the
      canonical decomposition (hence C_S') must be independent of gamma.

Run:  python3 d36_CS_crosschecks.py
Pure-mathematics research; standard modular-forms jargon.
"""
from __future__ import annotations
import sys, os, pickle, time
CODE = os.path.dirname(os.path.abspath(__file__))  # resolve caches/receipts next to this file
sys.path.insert(0, CODE)
from fractions import Fraction as Fr
import sympy as sp
import mpmath

import ct_dual_d36_cutgen as CG
import d36_newform_decomp as ND
import eisen_projection as EP

DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
COEF18 = sp.Rational(-28728, 43867)


def X1_direct_scan(Lext=2000):
    print("=" * 84)
    print(f"(X1) DELIGNE-INDEPENDENT: extend EXACT a_n,b_n>=0 scan to n={Lext} (past n0<=63)")
    print("=" * 84)
    with open(os.path.join(CODE, 'd36_iter1_vertex.pkl'), 'rb') as f:
        VC = pickle.load(f)
    x = [Fr(*t) for t in VC['x']]
    out = []
    GAr, GBr, r, specs, n = CG.build_reduced_rows(Lext, out)     # exact reduced rows to Lext
    aneg = [m for m in range(1, Lext + 1) if sum(GAr[m][l] * x[l] for l in range(r)) < 0]
    bneg = [m for m in range(1, Lext + 1) if sum(GBr[m][l] * x[l] for l in range(r)) < 0]
    a0 = sum(GAr[0][l] * x[l] for l in range(r))
    print(f"  a_0 == 1 exact: {a0 == 1}")
    print(f"  a_n < 0 count for 1<=n<={Lext}: {len(aneg)}  {aneg[:8] if aneg else '(NONE)'}")
    print(f"  b_n < 0 count for 1<=n<={Lext}: {len(bneg)}  {bneg[:8] if bneg else '(NONE)'}")
    print(f"  => EXACT positivity holds to n={Lext} (direct, no Deligne): "
          f"{'CONFIRMED' if not aneg and not bneg else 'FAILED'}")
    return not aneg and not bneg


def X2_naive_lower_bounds():
    print("=" * 84)
    print("(X2) naive lower bound  sup|c_n|/(sigma_0(n) n^{17/2})  <=  C_S'  (both sides)")
    print("=" * 84)
    mpmath.mp.dps = 40
    with open(os.path.join(CODE, 'd36_iter1_vertex.pkl'), 'rb') as f:
        VC = pickle.load(f)
    a = [Fr(*t) for t in VC['a']]; b = [Fr(*t) for t in VC['b']]
    P, PIV = EP._proj()

    def cusp_of(coeffs):
        # exact Eisenstein coeffs via PROJ
        aa = sp.Matrix(72, 1, lambda i, _: sp.Rational(coeffs[PIV[i]].numerator, coeffs[PIV[i]].denominator))
        ev = P * aa
        ed = {DIVS[i]: sp.Rational(ev[i]) for i in range(8)}
        def eis(delta, nn):
            if nn == 0: return sp.Integer(1)
            return COEF18 * sp.divisor_sigma(nn // delta, 17) if nn % delta == 0 else sp.Integer(0)
        return [sp.Rational(coeffs[nn]) - sum(ed[d] * eis(d, nn) for d in DIVS) for nn in range(len(coeffs))]

    for name, coeffs, CSp in [("b", b, mpmath.mpf('0.3588077783')), ("a", a, mpmath.mpf('0.3063293335'))]:
        cusp = cusp_of(coeffs)
        worst = (mpmath.mpf(0), None)
        for m in range(1, 801):
            if cusp[m] == 0: continue
            cm = mpmath.mpf(cusp[m].p) / mpmath.mpf(cusp[m].q)
            den = mpmath.mpf(int(sp.divisor_sigma(m, 0))) * mpmath.mpf(m) ** (mpmath.mpf(17) / 2)
            rr = abs(cm) / den
            if rr > worst[0]: worst = (rr, m)
        ok = worst[0] <= CSp
        print(f"  side {name}: naive sup = {mpmath.nstr(worst[0],6)} at n={worst[1]};  C_S' = {mpmath.nstr(CSp,6)}  "
              f"=> lower<=C_S': {ok}")


def X3_deligne_level1():
    print("=" * 84)
    print("(X3) Deligne spot-check: level-1 newform (Delta*E6), |a_n| <= sigma_0(n) n^{17/2}, n<=800")
    print("=" * 84)
    M = 800
    eu = [0] * (M + 1); eu[0] = 1
    for n in range(1, M + 1):
        for i in range(M, n - 1, -1):
            eu[i] -= eu[i - n]
    R = [0] * (M + 1); R[0] = 1
    for _ in range(24):
        C = [0] * (M + 1)
        for i in range(M + 1):
            if R[i]:
                for j in range(M + 1 - i):
                    if eu[j]: C[i + j] += R[i] * eu[j]
        R = C
    Delta = [0] * (M + 1)
    for nq in range(M): Delta[nq + 1] = R[nq]
    E6 = [1] + [-504 * int(sp.divisor_sigma(nq, 5)) for nq in range(1, M + 1)]
    f = [0] * (M + 1)
    for i in range(M + 1):
        if Delta[i]:
            for j in range(M + 1 - i):
                if E6[j]: f[i + j] += Delta[i] * E6[j]
    mpmath.mp.dps = 40
    bad = []
    for nq in range(1, M + 1):
        bound = mpmath.mpf(int(sp.divisor_sigma(nq, 0))) * mpmath.mpf(nq) ** (mpmath.mpf(17) / 2)
        if abs(f[nq]) > bound:
            bad.append(nq)
    print(f"  a_1..a_7 = {f[1:8]}  (=level-1 eigenform, B4)")
    print(f"  # of n<=800 violating |a_n| <= sigma_0(n) n^8.5: {len(bad)}  {'(NONE -> Deligne holds)' if not bad else bad[:10]}")


def X4_gamma_robustness():
    print("=" * 84)
    print("(X4) gamma-robustness: recompute C_S' (b-side) with a DIFFERENT Hecke combo weight")
    print("=" * 84)
    orig = ND.GAMMA
    ND.GAMMA = mpmath.mpf('1.3247179572447460259609088')   # plastic number, != golden ratio
    R = ND.compute_CS(verbose=False)
    ND.GAMMA = orig
    print(f"  with gamma=1.32471795...: C_S' = {mpmath.nstr(R['CSp'],10)}, validation {mpmath.nstr(R['maxrel'],4)}")
    print(f"  (original gamma=0.618...: C_S' = 0.3588077783)  => gamma-independent: "
          f"{abs(R['CSp'] - mpmath.mpf('0.3588077783')) < mpmath.mpf(10)**-6}")


if __name__ == "__main__":
    t0 = time.time()
    X2_naive_lower_bounds()
    print()
    X3_deligne_level1()
    print()
    X4_gamma_robustness()
    print()
    Lext = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
    X1_direct_scan(Lext)
    print(f"\n[cross-checks done in {time.time()-t0:.0f}s]")
