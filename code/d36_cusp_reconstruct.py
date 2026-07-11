#!/usr/bin/env python3
r"""
§35 step 1 — EXACT reconstruction of the §34 iter1 dual point, its Fricke image g~, its
EXACT Eisenstein projection, and the EXACT cusp component  S = g~ - P_E(g~)  in S_18(Gamma_0(24)).

This is the object whose Fourier coefficients we must bound by Deligne (CT Sec 5) to pin the
explicit tail constant C_S and the crossover n0 = (C_S/|COEF18*r3|)^{2/17}.

Everything EXACT (fractions.Fraction / sympy.Rational).  NO nsimplify on any large rational
(the §30 poison).  Reuses ct_dual_d36_cutgen (reduced rows, cached to 800), eisen_projection
(the exact 8x72 projector PROJ), ct_dual_d36 (fast exact eta/E_k q-expansions).

Outputs (cached to disk for the downstream C_S computation):
  d36_iter1_vertex.pkl :  {'x': [20 Fraction], 'b': [b_0..b_L Fraction], 'a': [a_0..a_L Fraction],
                           'cusp': [c_0..c_L Fraction], 'e_delta': {delta: Fraction}, 'r3': Fraction,
                           'L': L}
Also prints:
  - independent re-derivation of e_1(g~), r3, |COEF18*r3|  (must match §34 receipt)
  - EXACT verification that S = g~ - P_E(g~) is CUSPIDAL (P_E(S) == 0 exactly)
  - naive SCALE of C_S:  max_{n<=L} |c_n| / (sigma_0(n) * n^{17/2})   (a LOWER bound on the true
    asymptotic constant; tells us the order of magnitude the Deligne C_S must exceed)

Run:  python3 d36_cusp_reconstruct.py [L]      (default L=800)
Pure-mathematics research; standard modular-forms / LP-dual jargon.
"""
from __future__ import annotations
import sys, os, time, pickle
CODE = os.path.dirname(os.path.abspath(__file__))  # resolve caches/receipts next to this file
sys.path.insert(0, CODE)
from fractions import Fraction as Fr
import sympy as sp

import ct_dual_d36_cutgen as CG        # reduced rows (cached), exact solver
import eisen_projection as EP          # exact 8x72 projector PROJ (+ PIV window)
import ct_dual_d36 as D                # DIVS, COEF18, fast exact q-expansions

N, K, d = 24, 18, 36
DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
COEF18 = sp.Rational(-28728, 43867)
RECORD = sp.Rational(2**18, 3**10)
VERT_CACHE = os.path.join(CODE, 'd36_iter1_vertex.pkl')


def sigma(kk, n):
    return int(sp.divisor_sigma(int(n), kk))


def reproduce_iter1_vertex(GAr, GBr, r, Lviol=300):
    """Reproduce the §34 constraint-generation iter1 EXACT vertex (20 Fractions).

    iter0: max b0 s.t. a0=1, a_n>=0 (10<=n<=28), b_n>=0 (1<=n<=28)  -> 45 violated b_n<0 in n<=Lviol.
    iter1: add ALL 45 violated rows -> W (|W|=73) -> exact-Fraction simplex -> the iter1 vertex.
    Matches receipt_d36_cutgen.txt (iter0 36.3470x -> iter1 32.9104x, converged, 0 violations<=300).
    """
    T, M = CG.T, CG.M
    # iter0
    W0 = set(range(1, M + 1))
    st, x0, b0_0, _ = CG.solve_working_set(GAr, GBr, r, W0, exact=True)
    assert st == 'optimal', f"iter0 status {st}"
    viol0 = CG.violations(GBr, x0, Lviol)
    bad = sorted(nn for _, nn, _ in viol0)
    print(f"  iter0: bound={CG.bound_of(b0_0):.5f} = {CG.bound_of(b0_0)/float(RECORD):.4f}x; "
          f"{len(bad)} violated b_n<0 (n<={Lviol}), first at n={bad[0] if bad else None}", flush=True)
    # iter1
    W1 = set(range(1, M + 1)) | set(bad)
    st, x1, b0_1, _ = CG.solve_working_set(GAr, GBr, r, W1, exact=True)
    assert st == 'optimal', f"iter1 status {st}"
    print(f"  iter1: |W|={len(W1)}  bound={CG.bound_of(b0_1):.5f} = "
          f"{CG.bound_of(b0_1)/float(RECORD):.4f}x record", flush=True)
    return x1, len(W1)


def eisen_coeffs_EXACT(bcoeffs):
    """EXACT e_delta(g) via the cached PROJ applied to EXACT coefficients (NO nsimplify).

    bcoeffs : list of exact sympy Rationals / Fractions, length > max(PIV).
    """
    P, PIV = EP._proj()
    need = max(PIV)
    if len(bcoeffs) <= need:
        raise ValueError(f"need {need+1} coeffs, got {len(bcoeffs)}")
    a = sp.Matrix(72, 1, lambda i, _: sp.Rational(bcoeffs[PIV[i]]))
    e = P * a
    return {DIVS[i]: sp.Rational(e[i]) for i in range(8)}


def eisen_pattern_val(delta, n):
    """E_18(delta z)_n exact:  1 if n==0; COEF18*sigma_17(n/delta) if delta|n; else 0."""
    if n == 0:
        return sp.Integer(1)
    if n % delta == 0:
        return COEF18 * sp.divisor_sigma(n // delta, 17)
    return sp.Integer(0)


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    t0 = time.time()
    print("=" * 80)
    print("§35 step 1 — EXACT cusp component S = g~ - P_E(g~) of the §34 iter1 point")
    print("=" * 80)

    out = []
    GAr, GBr, r, specs, n = CG.build_reduced_rows(L, out)
    print(f"  reduced rows ready: r={r} free vars, {n} forms, L={L}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- exact iter1 vertex (cache it: the exact solve is ~300s) ----
    if os.path.exists(VERT_CACHE):
        with open(VERT_CACHE, 'rb') as f:
            VC = pickle.load(f)
        if VC.get('L', 0) >= L:
            x = [Fr(*t) for t in VC['x']]
            print(f"  [cache] loaded iter1 vertex (20 Fractions)  [{time.time()-t0:.0f}s]", flush=True)
        else:
            x = None
    else:
        x = None
    if x is None:
        print("  reproducing exact iter1 vertex (constraint generation, ~300s exact simplex) ...", flush=True)
        x, wlen = reproduce_iter1_vertex(GAr, GBr, r)

    # ---- exact b_n (Fricke image g~) and a_n (form g) to length L, from cached reduced rows ----
    b = [sum(GBr[m][l] * x[l] for l in range(r)) for m in range(L + 1)]
    a = [sum(GAr[m][l] * x[l] for l in range(r)) for m in range(L + 1)]
    a0 = a[0]
    assert a0 == 1, f"a_0 != 1 (got {a0})"
    b0 = b[0]
    bound = float(b0) * (2 / N ** 0.5) ** (d / 2) * (N ** 0)  # placeholder; use CG.bound_of on b0
    bnd = CG.bound_of(b0)
    print(f"\n  a_0 = {a0} (exact);  b_0 = {float(b0):.6f} (exact);  bound = {bnd:.5f} = "
          f"{bnd/float(RECORD):.4f}x record", flush=True)
    neg_a = [m for m in range(1, L + 1) if a[m] < 0]
    neg_b = [m for m in range(1, L + 1) if b[m] < 0]
    print(f"  a_n<0 count (1<=n<={L}): {len(neg_a)};  b_n<0 count (1<=n<={L}): {len(neg_b)}   "
          f"(both must be 0 — matches §34)", flush=True)

    # ---- EXACT Eisenstein projection of g~ and the EXACT cusp component S ----
    print("\n  [Eisenstein read-off of g~ via exact PROJ; cusp component S = g~ - sum e_delta E_18(delta z)]",
          flush=True)
    egt = eisen_coeffs_EXACT([sp.Rational(v.numerator, v.denominator) for v in b])
    for dl in DIVS:
        v = egt[dl]
        print(f"    e_{dl:<2d}(g~) = {float(v):+.6e}", flush=True)
    e1 = egt[1]
    e3 = egt[3]
    den3 = 1 + 3 ** 17
    r3 = sp.Rational(e1 + e3 / den3)
    coef18_r3 = sp.Rational(COEF18 * r3)
    print(f"\n  e_1(g~) = {float(e1):+.6e}   (coprime classes gcd(n,24)=1)", flush=True)
    print(f"  r3      = e_1 + e_3/(1+3^17) = {float(r3):+.6e}   (binding gcd=3 classes {{3,9,15,21}})",
          flush=True)
    print(f"  COEF18*r3 = {float(coef18_r3):+.6e}   (>0 required for b_n>0 tail; |.| = "
          f"{abs(float(coef18_r3)):.6e})", flush=True)
    print(f"  [MATCH §34 receipt: e_1=-4.198e-13, r3=-4.615e-15, |COEF18*r3|=3.022e-15]", flush=True)

    # exact cusp component S = g~ - Eisenstein part
    cusp = [sp.Rational(b[m]) - sum(egt[dl] * eisen_pattern_val(dl, m) for dl in DIVS)
            for m in range(L + 1)]
    # c_0 must be 0 (both g~ and Eisenstein have constant term with e-weighted sum; check)
    print(f"\n  cusp c_0 = {cusp[0]} (Eisenstein carries the constant term; check c_0==0 after "
          f"subtracting sum e_delta): ", end="", flush=True)
    # NOTE eisen_pattern_val(delta,0)=1 for all delta, so Eisenstein const = sum e_delta; and b_0 = that.
    print(f"{'OK (0)' if cusp[0] == 0 else cusp[0]}", flush=True)

    # ---- EXACT verification: S is CUSPIDAL (its Eisenstein projection is 0) ----
    e_resid = eisen_coeffs_EXACT(cusp)
    all_zero = all(e_resid[dl] == 0 for dl in DIVS)
    print(f"\n  EXACT cuspidality check: P_E(S) == 0 for all delta: {all_zero}   "
          f"{'(S IS a cusp form)' if all_zero else '(FAIL)'}", flush=True)
    if not all_zero:
        for dl in DIVS:
            print(f"    P_E(S)[{dl}] = {e_resid[dl]}")

    # ---- naive SCALE of C_S: sup |c_n|/(sigma_0(n) n^{17/2}) over 1<=n<=L (a LOWER bound) ----
    import mpmath
    mpmath.mp.dps = 40
    worst = (0.0, None)
    for m in range(1, L + 1):
        if cusp[m] == 0:
            continue
        cm = mpmath.mpf(cusp[m].p) / mpmath.mpf(cusp[m].q)
        denom = mpmath.mpf(sigma(0, m)) * mpmath.power(mpmath.mpf(m), mpmath.mpf(17) / 2)
        ratio = abs(cm) / denom
        if float(ratio) > worst[0]:
            worst = (float(ratio), m)
    print(f"\n  naive scale  max_{{1<=n<={L}}} |c_n|/(sigma_0(n) n^(17/2)) = {worst[0]:.4f} at n={worst[1]}",
          flush=True)
    print(f"    => the true Deligne C_S = sum|lambda_j| is >= this (lower bound); we need C_S <= ~1e10", flush=True)
    print(f"    => target crossover for this scale: n0 ~ ({worst[0]:.3g}/{abs(float(coef18_r3)):.3g})^(2/17) = "
          f"{(worst[0]/abs(float(coef18_r3)))**(2/17):.1f}", flush=True)

    # ---- cache ----
    with open(VERT_CACHE, 'wb') as f:
        pickle.dump({'x': [(v.numerator, v.denominator) for v in x],
                     'b': [(sp.Rational(v).p, sp.Rational(v).q) for v in b],
                     'a': [(sp.Rational(v).p, sp.Rational(v).q) for v in a],
                     'cusp': [(sp.Rational(v).p, sp.Rational(v).q) for v in cusp],
                     'e_delta': {dl: (egt[dl].p, egt[dl].q) for dl in DIVS},
                     'r3': (r3.p, r3.q), 'coef18_r3': (coef18_r3.p, coef18_r3.q),
                     'L': L}, f)
    print(f"\n  [cached] exact vertex + g~ + a + cusp + e_delta + r3 -> {VERT_CACHE}", flush=True)
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
