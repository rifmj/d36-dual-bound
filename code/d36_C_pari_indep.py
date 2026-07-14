#!/usr/bin/env python3
r"""Independent PARI/GP reproduction of the d=36 lift-aware Deligne constants C_w, both sides.

DISJOINT from the campaign pipeline (which is sympy eta-quotient echelon + exact charpoly +
Sturm intervals / high-precision diagonalization): this script uses PARI 2.17.2's native
modular-symbols mf engine (mfinit / mfeigenbasis / mffields / mfbd / mfembed) to build the
newform/oldform lift basis of S_18(Gamma_0(24)) from scratch, decomposes the certificate's
exact cusp parts (input: d36_cusp_targets_L150.json, produced by d36_export_cusp_targets.py),
and recomputes C_w = sum |lambda_{f,e}| / e^{17/2}.

Cross-checks against the paper / receipt_d36_cs.txt:
  * full cusp dim 64; newspace dims (1,1,3,2,3,4,2,9) for M=(1,2,3,4,6,8,12,24)
  * 64 individual lift q-series spanning S_18(Gamma_0(24))
  * reconstruction residual ~0 over all rows n=1..150 (64 determining rows used for the solve)
  * C_w(g~) vs receipt <= 0.358807778338, C_w(g) vs receipt <= 0.306329333529
  * level-1 newform (Delta*E6) eigenvalues a_2=-528, a_3=-4284, a_5=-1025850, a_7=3225992
  * Deligne bound |a_p| <= 2 p^{17/2} spot-checked on all newform embeddings, p<=97

NOTE (same status as the d=10 PARI reproduction, §43/§44): PARI reproduces the VALUES
numerically at 80-digit precision from an independently built eigenbasis; the RIGOROUS
outward-rounded upper bound on C_w remains the interval certificate of d36_cs_certificate.py.
Requires cypari2 (pip install cypari2); this optional cross-check is the one script in the
bundle that uses a CAS.  Pure-mathematics research; standard modular-forms jargon.
"""
import json, os, sys
from collections import defaultdict
import cypari2

pari = cypari2.Pari()
pari.default('parisize', '2G')
# cypari2 gotcha: default('realprecision') does NOT reach library-mode mf* calls — the
# embedding precision must be passed explicitly (in BITS) to mfembed. The 64x64 lift-basis
# solve is ill-conditioned (~10^30; coefficients grow like n^17), so give it real room:
PREC_BITS = 1536          # ~460 decimal digits
pari.set_real_precision(200)

CODE = os.path.dirname(os.path.abspath(__file__))
N, K = 24, 18
DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
EHALF = (K - 1) / 2.0          # 8.5 : oldform-lift normalization e^{(k-1)/2}

def divisors24(n):
    return [d for d in DIVS if n % d == 0]

def embed_rows(f, gco, rdeg):
    e = pari.mfembed(f, gco, precision=PREC_BITS)
    return [e] if rdeg == 1 else [e[i] for i in range(rdeg)]

def build_columns(Ncheck):
    """All 64 individual newform-lift q-series of S_18(Gamma_0(24)) (complex embeddings)."""
    cols, labels = [], []
    newdims = {}
    lvl1_eigs = None
    for M in DIVS:
        mfM = [M, K]
        dnew = int(pari.mfdim(mfM, 0))
        newdims[M] = dnew
        if dnew == 0:
            continue
        mf0 = pari.mfinit(mfM, 0)
        eig = pari.mfeigenbasis(mf0)
        dims = [int(pari.poldegree(P)) for P in pari.mffields(mf0)]
        assert sum(dims) == dnew
        for oi, f in enumerate(eig):
            rdeg = dims[oi]
            if M == 1:
                co = pari.mfcoefs(f, 7)
                lvl1_eigs = [int(co[2]), int(co[3]), int(co[5]), int(co[7])]
            for e in divisors24(N // M):
                g = pari.mfbd(f, e) if e > 1 else f
                gco = pari.mfcoefs(g, Ncheck)
                for row in embed_rows(f, gco, rdeg):
                    cols.append([row[n] for n in range(1, Ncheck + 1)])
                    labels.append((M, oi, e))
    return cols, labels, newdims, lvl1_eigs

def deligne_check(cols, labels, Ncheck):
    """|a_f(p)| <= 2 p^{8.5} on the pure newform embeddings (e=1 columns), p prime <= min(97,Ncheck)."""
    viol = 0
    primes = [p for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
              if p <= Ncheck]
    for j, (M, oi, e) in enumerate(labels):
        if e != 1:
            continue
        for p in primes:
            if abs(float(pari.abs(cols[j][p - 1]))) > 2 * p**EHALF + 1e-6:
                viol += 1
    return viol

def solve_and_C(cols, labels, target, Ncheck, dim=64):
    fullcols = [pari([cols[j][i] for i in range(Ncheck)]).Col() for j in range(dim)]
    Mfull = pari.matconcat(fullcols)
    rank = int(pari.matrank(Mfull))
    Mt = pari.mattranspose(Mfull)
    idx = pari.matindexrank(Mt)
    rowsel = [int(x) for x in idx[1]]
    assert len(rowsel) >= dim, f"only {len(rowsel)} independent rows"
    rowsel = rowsel[:dim]
    Msq = pari.matconcat([pari([cols[j][r - 1] for r in rowsel]).Col() for j in range(dim)])
    rhs = pari([target(r) for r in rowsel]).Col()
    lam = pari.matsolve(Msq, rhs)
    lamcol = pari([lam[j] for j in range(dim)]).Col()
    rhsfull = pari([target(i + 1) for i in range(Ncheck)]).Col()
    resid = Mfull * lamcol - rhsfull
    # RELATIVE residual: the coefficients grow like n^17 (~1e37 at n=150), so the
    # meaningful reconstruction metric is |residual| / max(1, |target|) per row.
    mx = max(float(pari.abs(resid[i])) / max(1.0, abs(float(rhsfull[i])))
             for i in range(Ncheck))
    C = 0.0
    bylev = defaultdict(float)
    for j in range(dim):
        aj = float(pari.abs(lam[j]))
        e = labels[j][2]
        C += aj / e**EHALF
        bylev[(labels[j][0], e)] += aj / e**EHALF
    return C, mx, rank, bylev

def main():
    data = json.load(open(os.path.join(CODE, 'd36_cusp_targets_L150.json')))
    Ncheck = data['NCHECK']
    print('=' * 78)
    print('d36_C_pari_indep.py — independent PARI/GP reproduction of C_w, both sides')
    print('=' * 78)
    print(f'PARI version: {pari.version()}   Ncheck={Ncheck}')
    dim_cusp = int(pari.mfdim([N, K], 1))
    print(f'full cusp dim mfdim([24,18],1) = {dim_cusp}   (paper: 64)')
    cols, labels, newdims, lvl1 = build_columns(Ncheck)
    print(f'newspace dims per level M: {newdims}   (paper: 1,1,3,2,3,4,2,9)')
    print(f'# individual lift q-series = {len(labels)}   (paper: 64)')
    print(f'level-1 newform eigenvalues a_2,a_3,a_5,a_7 = {lvl1}')
    print(f'   (paper: -528, -4284, -1025850, 3225992)')
    viol = deligne_check(cols, labels, Ncheck)
    print(f'Deligne |a_p| <= 2 p^8.5 spot-check on newform embeddings, p<=97: violations = {viol}')

    ok = (dim_cusp == 64 and len(labels) == 64
          and [newdims[M] for M in DIVS] == [1, 1, 3, 2, 3, 4, 2, 9]
          and lvl1 == [-528, -4284, -1025850, 3225992] and viol == 0)

    for name, key, receipt in [('g~ (Fricke side)', 'b_side', 0.358807778338),
                               ('g  (a side)',      'a_side', 0.306329333529)]:
        S = data[key]['S']
        def tf(i, S=S):
            return pari(f'{S[i][0]}/{S[i][1]}')
        C, mx, rank, bylev = solve_and_C(cols, labels, tf, Ncheck)
        print('-' * 78)
        print(f'{name}:')
        print(f'  basis rank over rows n=1..{Ncheck}: {rank}  (want 64)')
        print(f'  reconstruction max RELATIVE residual over ALL rows n=1..{Ncheck}: '
              f'{mx:.3e}  (want ~0)')
        print(f'  ==> C_w = sum |lambda|/e^8.5 = {C:.12f}')
        print(f'      receipt (interval-certified upper bound): {receipt}')
        agree = abs(C - receipt) < 5e-11
        print(f'      agreement to ~1e-11: {agree}')
        dom = max(bylev.items(), key=lambda kv: kv[1])
        print(f'      dominant (level, lift): M={dom[0][0]}, e={dom[0][1]} '
              f'-> {dom[1]:.6f} of C_w')
        ok = ok and rank == 64 and mx < 1e-60 and agree

    print('=' * 78)
    print('PARI INDEPENDENT REPRODUCTION: ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
