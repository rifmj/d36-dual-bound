#!/usr/bin/env python3
"""
gen-5 J2 — GENERAL Cohn–Triantafillou dual-LP builder (arbitrary d, level N), NO modular-forms CAS.
Validated by reproducing the §17 toy case (d=16, N=4) with FULLY COMPUTED Atkin–Lehner (Fricke) images
(not the hardcoded permutation): T=3 -> center-density bound 3^8/2^16 = 0.1001129 exactly.
Pure-mathematics research; standard modular-forms / LP-bound jargon.

Building blocks (all exact, sympy):
  E_k(delta z)   : Eisenstein E_k = 1 - (2k/B_k) sum sigma_{k-1}(n) q^n, scaled q->q^delta.
  eta_quotient   : prod_{delta|N} eta(delta z)^{r_delta}, exact truncated integer power series.
  Fricke images (i^k g|_k w_N),  w_N = (0,-1;N,0)/sqrt(N):
    Eisenstein  E_k(delta z)  ->  i^k * N^{-k/2} (N/delta)^k * E_k((N/delta) z)
    eta-quotient prod eta(delta z)^{r_delta} -> N^{k/2} prod delta^{-r_delta/2} * (reversed quotient, delta->N/delta)
    (calibrated on the toy: E_8(z)->256 E_8(4z), E_8(2z)->E_8(2z), f=eta^8 eta2^8 -> 16 f(2z)).
Dual LP (weak duality): g=sum x_j g^j, a_n=coeffs(g), b_n=coeffs(g~).  maximize b_0 s.t.
  a_0=1, a_n=0 (1<=n<T), a_n>=0 (T<=n<=M), b_n>=0 (1<=n<=M).  center-density bound = b_0 (2/sqrt N)^{d/2}(sqrt T/2)^d.

Run:  python3 ct_dual_general.py   (self-test = reproduce the d=16 toy via computed Fricke)
"""
from __future__ import annotations
import sympy as sp

# ---------- exact q-expansions ----------
def E_k(k, M):
    coef = {4: 240, 6: -504, 8: 480, 10: -264}[k]      # -2k/B_k
    return [sp.Integer(1)] + [coef * sp.divisor_sigma(n, k - 1) for n in range(1, M + 1)]

def scale(c, d, M):
    out = [sp.Integer(0)] * (M + 1)
    for n in range(M + 1):
        if n * d <= M:
            out[n * d] = c[n]
    return out

def _prod_1mq_delta_pow(P, delta, r, M):
    for n in range(1, M // delta + 1):
        kk = delta * n
        Q = [sp.Integer(0)] * (M + 1)
        j = 0
        while j * kk <= M:
            coef = sp.binomial(r, j) * (-1) ** j
            for i in range(M + 1 - j * kk):
                Q[i + j * kk] += coef * P[i]
            j += 1
        P = Q
    return P

def eta_quotient_qexp(exps, M):
    """prod eta(delta z)^{r_delta} as coeffs of q^0..M (includes the leading q^{ord})."""
    num = sum(d * r for d, r in exps.items())
    assert num % 24 == 0, f"q-order {num}/24 not integer for {exps}"
    ord_ = num // 24
    P = [sp.Integer(0)] * (M + 1); P[0] = sp.Integer(1)
    for d, r in exps.items():
        P = _prod_1mq_delta_pow(P, d, r, M)
    return [sp.Integer(0)] * ord_ + P[:M + 1 - ord_]

# ---------- Fricke images ----------
def fricke_eisen(delta, k, N, M):
    C = sp.nsimplify(sp.I ** k * sp.Rational(1, N) ** sp.Rational(k, 2) * sp.Rational(N, delta) ** k)
    return C, scale(E_k(k, M), N // delta, M)

def fricke_etaq(exps, k, N, M):
    C = sp.nsimplify(sp.Integer(N) ** sp.Rational(k, 2) * sp.prod([sp.Integer(d) ** sp.Rational(-r, 2) for d, r in exps.items()]))
    rev = {}
    for d, r in exps.items():
        rev[N // d] = rev.get(N // d, 0) + r
    return C, eta_quotient_qexp(rev, M)

# ---------- dual LP ----------
def dual_lp(g_list, gt_list, T, d, N, M):
    import numpy as np
    from scipy.optimize import linprog
    n = len(g_list)
    G = np.array([[float(g_list[j][m]) for j in range(n)] for m in range(M + 1)])
    Gt = np.array([[float(gt_list[j][m]) for j in range(n)] for m in range(M + 1)])
    A_eq = [G[0]] + [G[m] for m in range(1, T)]
    b_eq = [1.0] + [0.0] * (T - 1)
    A_ub = [-G[m] for m in range(T, M + 1)] + [-Gt[m] for m in range(1, M + 1)]
    b_ub = [0.0] * len(A_ub)
    res = linprog(-Gt[0], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)] * n, method="highs")
    if not res.success:
        return None
    b0 = float(-res.fun)
    return {"b0": b0, "bound": b0 * (2 / N ** 0.5) ** (d / 2) * (T ** 0.5 / 2) ** d, "x": res.x}

def build_basis(specs, k, N, M):
    """specs: list of ('eis', delta) or ('eta', {delta:r,...}).  Returns (g_list, gt_list)."""
    g_list, gt_list = [], []
    for s in specs:
        if s[0] == "eis":
            g_list.append(scale(E_k(k, M), s[1], M)); C, h = fricke_eisen(s[1], k, N, M)
        else:
            g_list.append(eta_quotient_qexp(s[1], M)); C, h = fricke_etaq(s[1], k, N, M)
        gt_list.append([C * x for x in h])
    return g_list, gt_list


def toy_selftest():
    d, N, k, M = 16, 4, 8, 12
    specs = [("eis", 1), ("eis", 2), ("eis", 4), ("eta", {1: 8, 2: 8}), ("eta", {2: 8, 4: 8})]
    g, gt = build_basis(specs, k, N, M)
    print("Toy d=16,N=4 self-test (computed Fricke, all 5 basis elts):")
    ok = True
    for T, want in [(2, sp.Rational(1, 16)), (3, sp.Rational(3 ** 8, 2 ** 16)), (4, sp.Rational(1, 16))]:
        r = dual_lp(g, gt, T, d, N, M)
        match = abs(r["bound"] - float(want)) < 1e-9
        ok = ok and match
        print(f"  T={T}: bound={r['bound']:.7f}  (want {float(want):.7f})  {'OK' if match else 'MISMATCH'}")
    print(f"  => general builder validated: {ok}")
    return ok


if __name__ == "__main__":
    toy_selftest()
