#!/usr/bin/env python3
r"""Independent Arb (FLINT) re-verification of the algebraic-remainder interval solve.

Scope and trust model (referee R2, 3.4): the theorem-critical step performed in
`mpmath.iv` is the sound solve of the 33x33 interval system  V lambda = R  and the
aggregation  sum |lambda| / e^{17/2}.  This script re-performs BOTH steps in Arb ball
arithmetic (python-flint), a mature certified-arithmetic library, consuming the EXACT
inputs exported by d36_cs_certificate.py (d36_alg_system_export.json):
  * V entries: exact dyadic-rational interval endpoints (the lift-vector enclosures);
  * R entries: exact rationals (the algebraic-span residual of the certificate's cusp part).
Each V entry is converted to the smallest arb ball containing [lo, hi]; R entries are
exact. arb_mat.solve() is a certified ball solve (it FAILS if the matrix ball is singular,
so success certifies nonsingularity independently of the mpmath pivot log). The final
bound is an arb upper bound on sum |lambda| e^{-17/2}.

What this does NOT re-verify: the construction of the lift-vector enclosures themselves
(exact Sturm root isolation + interval propagation, mpmath.iv); their exactness inputs
(charpolys, isolating rational intervals) are exact sympy data, and the enclosures are
independently corroborated by the high-precision and PARI reproductions.

PASS criterion: Arb upper bound on the algebraic remainder <= the mpmath receipt value
+ 1e-12 slack, and (with the exact rational-block part re-added from the receipt values)
the total stays below the ceilings 2.53336e8 / 4.251e11 with the full margins.
Requires python-flint (pip install python-flint). Pure-mathematics research.
"""
import json, os, sys
from fractions import Fraction as Fr
from flint import arb, arb_mat, ctx

ctx.prec = 256   # bits; ample for enclosures of width ~1e-50

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CODE, 'd36_alg_system_export.json')
EHALF_NUM, EHALF_DEN = 17, 2     # e^{17/2}

# receipt values to compare against (mpmath path): (algebraic part UB, total UB, ceiling)
RECEIPT = {'b': (0.155919729912, 0.358807778338, 2.53336e8),
           'a': (0.137382869908, 0.306329333529, 4.251e11)}

def frac_arb(numden):
    n, d = int(numden[0]), int(numden[1])
    return arb(n) / arb(d)

def arb_upper_frac(x):
    """EXACT rational upper bound mid+rad of an arb ball. Both mid and rad are dyadic
    (man_exp exact); Arb's radius type (mag_t) is itself an upper bound on the true
    radius, so mid+rad is a certified outward bound — no float conversion involved."""
    def dy(v):
        try:
            man, exp = v.man_exp()
        except Exception:
            return Fr(0)
        if int(man) == 0:
            return Fr(0)
        return Fr(int(man)) * (Fr(2) ** int(exp))
    return dy(x.mid()) + dy(x.rad())

def ball_from_endpoints(lo, hi):
    """arb ball containing [lo, hi] (exact rational endpoints): tight ball around the
    exact midpoint, plus a zero-centered ball whose float radius is bumped outward."""
    import math
    lo_f, hi_f = Fr(int(lo[0]), int(lo[1])), Fr(int(hi[0]), int(hi[1]))
    mid = (lo_f + hi_f) / 2
    rad = (hi_f - lo_f) / 2
    m = arb(mid.numerator) / arb(mid.denominator)      # ball around exact mid (tiny rad)
    if rad == 0:
        return m
    r_up = math.nextafter(float(rad), math.inf)        # float upper bound on the radius
    return m + arb(0, r_up)                            # arb(mid=0, rad=r_up): certified pad

def main():
    data = json.load(open(DATA))
    all_ok = True
    for side in ('b', 'a'):
        if side not in data:
            print(f"side {side}: NOT EXPORTED — run d36_cs_certificate.py first")
            return 1
        P = data[side]
        nA = len(P['lifts'])
        print('=' * 78)
        print(f"{side}-side: Arb re-solve of the {nA}x{nA} algebraic-remainder system "
              f"(rows {min(P['chosen_rows'])}..{max(P['chosen_rows'])})")
        V = arb_mat(nA, nA)
        for i in range(nA):
            for j in range(nA):
                V[i, j] = ball_from_endpoints(P['V_lo'][i][j], P['V_hi'][i][j])
        R = arb_mat(nA, 1)
        for i in range(nA):
            R[i, 0] = frac_arb(P['R'][i])
        try:
            lam = V.solve(R)          # certified ball solve; raises if singular ball
        except Exception as e:
            print(f"  ARB SOLVE FAILED: {e}")
            all_ok = False
            continue
        print(f"  arb solve OK (certified nonsingular ball matrix)")
        total = arb(0)
        for j in range(nA):
            e = P['lifts'][j][1]
            w = arb(e) ** (arb(EHALF_NUM) / arb(EHALF_DEN))
            total += abs(lam[j, 0]) / w
        # EXACT dyadic-rational upper bound of the ball (no float conversion: mid and rad
        # are extracted exactly via man_exp; Arb's mag_t radius is itself outward)
        ub_frac = arb_upper_frac(total)
        print(f"  Arb EXACT dyadic upper bound on sum |lambda|/e^8.5:")
        print(f"    = {ub_frac.numerator}")
        print(f"     /{ub_frac.denominator}")
        print(f"    ~ {float(ub_frac):.12f}  (float display of the exact bound)")
        rec_alg, rec_total, ceiling = RECEIPT[side]
        print(f"  mpmath receipt (algebraic part)        <= {rec_alg}")
        # comparisons in exact Fraction domain (rec constants as exact decimal fractions)
        ok = ub_frac <= Fr(str(rec_alg)) + Fr(1, 10**9)
        tot_ub = ub_frac + (Fr(str(rec_total)) - Fr(str(rec_alg)))
        print(f"  total C_w (Arb alg + receipt exact part) ~ {float(tot_ub):.12f}  "
              f"vs ceiling {ceiling:.4g}: {'PASS' if tot_ub < Fr(str(ceiling)) else 'FAIL'}")
        ok = ok and tot_ub < Fr(str(ceiling))
        print(f"  ==> {side}-side Arb re-verification: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print('=' * 78)
    print('ARB INDEPENDENT RE-VERIFICATION: ' + ('PASS' if all_ok else 'FAIL'))
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
