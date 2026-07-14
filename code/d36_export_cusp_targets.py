#!/usr/bin/env python3
r"""Export the EXACT rational cusp parts S = (form) - (Eisenstein part), both sides, to JSON,
as input for the independent PARI/GP reproduction (d36_C_pari_indep.py).

S_a = g - P_E(g), S_b = g~ - P_E(g~); coefficients n=0..NCHECK as exact (num, den) pairs.
The Eisenstein part is computed from the exact projector (eisen_projection):
Eis(n) = kappa_18 * sum_{delta | gcd(n,24)} e_delta * sigma_17(n/delta) for n>=1,
Eis(0) = sum_delta e_delta.  All arithmetic exact rational.
Pure-mathematics research; standard modular-forms jargon.
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import sympy as sp
import eisen_projection as EP

NCHECK = 150
DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
KAPPA = Fr(-28728, 43867)
CODE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(CODE, 'd36_iter1_vertex.pkl'), 'rb') as f:
    VC = pickle.load(f)
P, PIV = EP._proj()

def sigma17(m):
    return sum(d**17 for d in range(1, m + 1) if m % d == 0)

def side(key):
    coeffs = [Fr(p, q) for p, q in VC[key]]
    v = sp.Matrix(72, 1, lambda i, _: sp.Rational(coeffs[PIV[i]].numerator,
                                                  coeffs[PIV[i]].denominator))
    e = P * v
    ed = {DIVS[i]: Fr(int(sp.Rational(e[i]).p), int(sp.Rational(e[i]).q)) for i in range(8)}
    S = []
    for n in range(NCHECK + 1):
        if n == 0:
            eis = sum(ed.values())
        else:
            eis = KAPPA * sum(ed[dl] * sigma17(n // dl) for dl in DIVS if n % dl == 0)
        s = coeffs[n] - eis
        S.append([str(s.numerator), str(s.denominator)])
    return {'S': S, 'e_delta': {str(dl): [str(ed[dl].numerator), str(ed[dl].denominator)]
                                for dl in DIVS}}

out = {'NCHECK': NCHECK, 'a_side': side('a'), 'b_side': side('b')}
path = os.path.join(CODE, 'd36_cusp_targets_L150.json')
with open(path, 'w') as f:
    json.dump(out, f)
print(f"wrote {path}")
# quick sanity: S must vanish at n=0 (both forms have Eisenstein constant term = a_0-part)
for k in ('a_side', 'b_side'):
    n0 = out[k]['S'][0]
    print(f"  {k}: S(0) = {n0[0]}/{n0[1]}  (cusp form => expect 0)")
