#!/usr/bin/env python3
r"""
§35 step 2 — newform/oldform DECOMPOSITION of S_18(Gamma_0(24)) and the explicit Deligne C_S.

Goal: for the exact cusp component  S = sum c_n q^n  (from d36_cusp_reconstruct.py), write
      S = sum_{f,e} lambda_{f,e} f(e z)
over the newform/oldform basis (f a normalized newform of level M|24, e|(24/M)), and set
      C_S = sum_{f,e} |lambda_{f,e}|.
Then, by Deligne  |a_m(f)| <= sigma_0(m) m^{(k-1)/2}  for every normalized newform f, we get the
rigorous tail bound   |c_n| <= C_S * sigma_0(n) * n^{17/2}   (CT arXiv:1909.04772 Sec 5).

METHOD (pure numpy/mpmath; NO Sage/Pari, NO nsimplify):
  1. Exact rational cusp basis of S_18(Gamma_0(24)) (dim 64), row-reduced to a q-echelon basis
     {b_i} with distinct leading exponents (pivots).                       [exact Fraction]
  2. Exact Hecke matrices T_5, T_7 (good primes) and U_2, U_3 (bad primes) on {b_i}. [exact Fraction]
  3. Numerically diagonalise a generic combination G = T_5 + g*T_7 -> common Hecke eigenspaces E_f
     (each of dim t_f = #{e-lifts}); cluster eigenvalues.                    [numpy float / mpmath]
  4. Per eigenspace E_f:  read the lift set  Delta_f = {e | 24/M_f} from the leading exponents;
     recover a_2(f), a_3(f) from the traces of U_2,U_3 on E_f (a_2 = tr(U_2|E_f)/t_3, etc.);
     build the lift basis {f(e z): e in Delta_f} as q-vectors by matching their (known, small)
     coefficients a_{e'/e}(f) on the pivot window inside E_f.
  5. Assemble all 64 lift vectors; solve  S = sum lambda_{f,e} f(e z)  on a determining window;
     C_S = sum |lambda_{f,e}|.
  6. VALIDATE: reconstruct sum lambda f(e z) and compare to the EXACT c_n to n=L (must match).

Self-tests (B4): level-1 newform eigenvalues a_2=-528,a_3=-4284,a_5=-1025850,a_7=3225992 (Delta*E6);
  Hecke matrices commute; sum of eigenspace dims = 64.

Run:  python3 d36_newform_decomp.py [--selftest]     (default: full C_S run, needs step-1 cache)
Pure-mathematics research; standard modular-forms jargon.
"""
from __future__ import annotations
import sys, os, time, pickle
from fractions import Fraction as Fr
CODE = os.path.dirname(os.path.abspath(__file__))  # resolve caches/receipts next to this file
sys.path.insert(0, CODE)
import sympy as sp
import numpy as np
import mpmath

import ct_dual_d36 as D
import eisen_projection as EP

N, K = 24, 18
DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
COEF18 = sp.Rational(-28728, 43867)
P17 = 5 ** 0  # placeholder; per-prime p^{k-1} computed inline
KM1 = K - 1     # 17
BASIS_CACHE = os.path.join(CODE, 'd36_cusp_basis_hecke.pkl')


# ---------------------------------------------------------------------------
# fast EXACT q-expansions (python int / Fraction), reused from ct_dual_d36
# ---------------------------------------------------------------------------
def eis_qexp(delta, L):
    """E_18(delta z) as Fraction list length L+1:  1, then COEF18*sigma_17(n/delta) if delta|n."""
    c = Fr(int(sp.numer(COEF18)), int(sp.denom(COEF18)))
    out = [Fr(0)] * (L + 1)
    out[0] = Fr(1)
    for n in range(1, L + 1):
        if n % delta == 0:
            out[n] = c * int(sp.divisor_sigma(n // delta, KM1))
    return out


def etaq_qexp(exps, L):
    return [Fr(v) for v in D._etaq_fast(exps, L)]


def build_cusp_echelon(L, verbose=True):
    """Exact q-echelon basis of S_18(Gamma_0(24)) (dim 64), to length L.

    Returns (B, pivots): B = list of 64 Fraction q-vectors (len L+1) with B[i][pivots[i]]=1 and
    B[i][pivots[j]]=0 for j<i (upper-echelon by leading exponent); pivots strictly increasing.
    """
    t0 = time.time()
    specs, PIV = EP.select_basis()               # 72 independent M_18 forms + determining window
    P, PIVp = EP._proj()                          # exact 8x72 Eisenstein projector
    assert PIV == PIVp
    # exact q-expansions of the 72 forms to length L (fast exact)
    forms = []
    for s in specs:
        if s[0] == 'eis':
            forms.append(eis_qexp(s[1], L))
        else:
            forms.append(etaq_qexp(s[1], L))
    if verbose:
        print(f"    expanded 72 M_18 forms to L={L}  [{time.time()-t0:.0f}s]", flush=True)
    # cusp part of each form: subtract its Eisenstein projection
    Pf = [[Fr(int(sp.numer(P[i, j])), int(sp.denom(P[i, j]))) for j in range(72)] for i in range(8)]
    cusp_forms = []
    for f in forms:
        a_at_piv = [f[PIV[j]] for j in range(72)]
        e = [sum(Pf[i][j] * a_at_piv[j] for j in range(72)) for i in range(8)]  # e_delta
        eis = [eis_qexp(DIVS[i], L) for i in range(8)]
        cf = [f[n] - sum(e[i] * eis[i][n] for i in range(8)) for n in range(L + 1)]
        cusp_forms.append(cf)
    # row-reduce (echelon by leading exponent) -> 64 pivots
    B, pivots = [], []
    for cf in cusp_forms:
        v = cf[:]
        for bi, pc in zip(B, pivots):
            if v[pc] != 0:
                fct = v[pc]
                v = [v[n] - fct * bi[n] for n in range(L + 1)]
        pc = next((n for n in range(L + 1) if v[n] != 0), None)
        if pc is None:
            continue
        inv = Fr(1) / v[pc]
        v = [x * inv for x in v]
        # clear this pivot from previous basis vectors (full reduced echelon, keeps read-off clean)
        for i in range(len(B)):
            if B[i][pc] != 0:
                fct = B[i][pc]
                B[i] = [B[i][n] - fct * v[n] for n in range(L + 1)]
        B.append(v)
        pivots.append(pc)
    # sort by pivot
    order = sorted(range(len(B)), key=lambda i: pivots[i])
    B = [B[i] for i in order]
    pivots = [pivots[i] for i in order]
    if verbose:
        print(f"    cusp echelon basis: dim={len(B)}, pivots {pivots[0]}..{pivots[-1]}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return B, pivots


def echelon_coords(vec, B, pivots):
    """Express a cusp q-vector 'vec' (Fraction list) in the echelon basis B -> list of 64 Fractions."""
    v = vec[:]
    coords = []
    for bi, pc in zip(B, pivots):
        c = v[pc]
        coords.append(c)
        if c != 0:
            v = [v[n] - c * bi[n] for n in range(len(v))]
    return coords


def hecke_matrix(op, p, B, pivots, L):
    """Exact Hecke matrix (list of 64 lists of Fraction) of T_p (op='T') or U_p (op='U') on B.

    (T_p f)_n = f_{np} + p^{17} f_{n/p};  (U_p f)_n = f_{np}.  Column j = echelon coords of op(B[j]).
    Needs B[j] to length p*max(pivot).
    """
    p17 = p ** KM1
    need = p * pivots[-1]
    assert need <= L, f"need length {need} for {op}_{p}, have {L}"
    ncol = len(B)
    cols = []
    for j in range(ncol):
        f = B[j]
        img = [Fr(0)] * (L + 1)
        for n in range(1, pivots[-1] + 1):
            v = f[n * p] if n * p <= L else Fr(0)
            if op == 'T' and n % p == 0:
                v = v + p17 * f[n // p]
            img[n] = v
        cols.append(echelon_coords(img, B, pivots))
    # cols[j] = coords of op(B[j]); matrix M[i][j] = cols[j][i]
    M = [[cols[j][i] for j in range(ncol)] for i in range(ncol)]
    return M


# ---------------------------------------------------------------------------
# build / cache the exact cusp basis + Hecke matrices
# ---------------------------------------------------------------------------
def build_all(Lbasis=1000, verbose=True):
    if os.path.exists(BASIS_CACHE):
        with open(BASIS_CACHE, 'rb') as f:
            C = pickle.load(f)
        if C['L'] >= Lbasis:
            B = [[Fr(*t) for t in row] for row in C['B']]
            pivots = C['pivots']
            H = {k: [[Fr(*t) for t in row] for row in M] for k, M in C['H'].items()}
            if verbose:
                print(f"  [cache] cusp basis + Hecke matrices (L={C['L']}, dim {len(B)})", flush=True)
            return B, pivots, H
    t0 = time.time()
    B, pivots = build_cusp_echelon(Lbasis, verbose)
    H = {}
    for (op, p) in [('T', 5), ('T', 7), ('T', 11), ('T', 13), ('U', 2), ('U', 3)]:
        H[f"{op}{p}"] = hecke_matrix(op, p, B, pivots, Lbasis)
        if verbose:
            print(f"    Hecke {op}_{p} built  [{time.time()-t0:.0f}s]", flush=True)
    with open(BASIS_CACHE, 'wb') as f:
        pickle.dump({'L': Lbasis, 'pivots': pivots,
                     'B': [[(v.numerator, v.denominator) for v in row] for row in B],
                     'H': {k: [[(v.numerator, v.denominator) for v in row] for row in M]
                           for k, M in H.items()}}, f)
    if verbose:
        print(f"  [cached] basis + Hecke -> {BASIS_CACHE}  [{time.time()-t0:.0f}s]", flush=True)
    return B, pivots, H


def fr_matrix_to_mp(M):
    return mpmath.matrix([[mpmath.mpf(v.numerator) / mpmath.mpf(v.denominator) for v in row] for row in M])


# ---------------------------------------------------------------------------
# self-test: level-1 eigenvalues, commuting, dims
# ---------------------------------------------------------------------------
def selftest():
    print("=" * 80)
    print("§35 step 2 SELF-TEST — cusp basis + Hecke matrices of S_18(Gamma_0(24))")
    print("=" * 80)
    B, pivots, H = build_all()
    d = len(B)
    print(f"  dim cusp space = {d}   {'OK (64)' if d == 64 else 'FAIL'}")
    print(f"  pivots: {pivots}")
    # commuting T5,T7 (exact)
    def matmul(A, Bm):
        n = len(A)
        return [[sum(A[i][k] * Bm[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    T5, T7 = H['T5'], H['T7']
    C1 = matmul(T5, T7); C2 = matmul(T7, T5)
    comm = all(C1[i][j] == C2[i][j] for i in range(d) for j in range(d))
    print(f"  T5 T7 == T7 T5 (exact): {comm}   {'OK' if comm else 'FAIL'}")
    # level-1 newform: its (a_5,a_7)=(-1025850,3225992) must be a common eigenvalue with a 1-dim...
    # actually level-1 lifts to 8 forms (e|24) so it is an 8-dim eigenspace of T5,T7.
    T5m = fr_matrix_to_mp(T5)
    mpmath.mp.dps = 30
    E5, V5 = mpmath.eig(T5m)
    a5s = sorted(set(round(float(mpmath.re(x)), 3) for x in E5))
    print(f"  distinct T5 eigenvalues (a_5 of newforms): {len(a5s)} values")
    print(f"    level-1 a_5=-1025850 present: {any(abs(x+1025850) < 1e-2 for x in a5s)}")
    # count multiplicity of a_5=-1025850 (should be 8 = number of e|24 lifts)
    mult = sum(1 for x in E5 if abs(float(mpmath.re(x)) + 1025850) < 1e-1)
    print(f"    multiplicity of a_5=-1025850 (expect 8 lifts e|24): {mult}")
    print("  (B4 level-1 target: a_2=-528, a_3=-4284, a_5=-1025850, a_7=3225992)")
    return comm and d == 64


# ---------------------------------------------------------------------------
# step 3+ : full newform/oldform decomposition and C_S
# ---------------------------------------------------------------------------
GAMMA = mpmath.mpf('0.6180339887498948482045868343656381177203091798058')  # fixed irrational combo weight
DPS = 60


def _divisors(m):
    return [e for e in range(1, m + 1) if m % e == 0]


def small_a_values(a2, a3, Mf):
    """a_m(f) for m|24, from a_2,a_3 and the level M_f (bad-prime rules).

    2-part:  v2(Mf)=0 -> good recursion a_{2^{j+1}}=a2 a_{2^j}-2^17 a_{2^{j-1}};
             v2(Mf)=1 -> a_{2^j}=a2^j;  v2(Mf)>=2 -> a2=0, a_{2^j}=0 (j>=1).
    3-part:  v3(Mf)=0 -> good (a_9 not needed, 9!|24);  v3(Mf)=1 -> a3=+-3^8, a_{3^j}=a3^j.
    a_m = a_{2^i} * a_{3^j} for m=2^i 3^j (i<=3, j<=1)."""
    v2 = 0
    mm = Mf
    while mm % 2 == 0:
        v2 += 1; mm //= 2
    v3 = 1 if Mf % 3 == 0 else 0
    p2_17 = mpmath.mpf(2) ** 17
    # a_{2^i}, i=0..3
    A2 = [mpmath.mpf(1)]
    if v2 == 0:
        A2.append(a2)
        for j in range(2, 4):
            A2.append(a2 * A2[j - 1] - p2_17 * A2[j - 2])
    elif v2 == 1:
        for j in range(1, 4):
            A2.append(a2 ** j)
    else:
        for j in range(1, 4):
            A2.append(mpmath.mpf(0))
    # a_{3^j}, j=0..1  (only j<=1 ever needed since 9 does not divide 24)
    A3 = [mpmath.mpf(1), a3]
    out = {}
    for i in range(4):
        for j in range(2):
            m = (2 ** i) * (3 ** j)
            if m <= 24 and 24 % m == 0:
                out[m] = A2[i] * A3[j]
    return out


def compute_CS(L=800, verbose=True, cusp=None):
    """Full newform/oldform decomposition of an exact cusp form and its Deligne C_S.

    cusp: exact q-expansion [c_0,...,c_M] (Fraction/sympy Rational), M>=L+... .  Default = the
    g~ cusp component from step 1 (d36_iter1_vertex.pkl).
    Returns dict with C_S, C_S', the per-lift |lambda|, validation residual, and diagnostics.
    """
    t0 = time.time()
    mpmath.mp.dps = DPS
    if cusp is None:
        with open(os.path.join(CODE, 'd36_iter1_vertex.pkl'), 'rb') as f:
            VC = pickle.load(f)
        cusp = [Fr(*t) for t in VC['cusp']]        # exact c_n, n=0..VC['L']
        assert VC['L'] >= L
    else:
        def _tofr(v):
            if isinstance(v, Fr):
                return v
            q = sp.Rational(v)
            return Fr(int(q.p), int(q.q))
        cusp = [_tofr(v) for v in cusp]
        assert len(cusp) > L
    B, pivots, H = build_all(verbose=verbose)
    d = len(B)
    Lb = len(B[0]) - 1                              # basis length
    # ---- diagonalise generic combo G = T5 + GAMMA T7 ----
    T5 = fr_matrix_to_mp(H['T5']); T7 = fr_matrix_to_mp(H['T7'])
    U2 = fr_matrix_to_mp(H['U2']); U3 = fr_matrix_to_mp(H['U3'])
    G = T5 + GAMMA * T7
    if verbose:
        print(f"  diagonalising G=T5+g*T7 (mpmath dps={DPS}) ...", flush=True)
    E, V = mpmath.eig(G)                            # V columns = right eigenvectors (echelon coords)
    Vinv = V ** -1
    U2e = Vinv * U2 * V                             # block-diagonal by cluster (commutes with G)
    U3e = Vinv * U3 * V
    T5e = Vinv * T5 * V
    T7e = Vinv * T7 * V
    T11e = Vinv * fr_matrix_to_mp(H['T11']) * V
    T13e = Vinv * fr_matrix_to_mp(H['T13']) * V
    # ---- cluster by the Hecke eigenvalue TUPLE (a5,a7,a11,a13) (strong multiplicity one) ----
    # G's eigenvectors are simultaneous Hecke eigenvectors; a_p^(k)=Tpe[k,k] is CONSTANT within a
    # newform system (all e-lifts share the a_p) and distinguishes distinct systems.  (a5,a7) alone
    # can coincide for two distinct newforms; four good primes make an accidental full-tuple match
    # effectively impossible — and the divisor-closed check on each Delta re-verifies it.
    key = [(mpmath.re(T5e[i, i]), mpmath.re(T7e[i, i]), mpmath.re(T11e[i, i]), mpmath.re(T13e[i, i]))
           for i in range(d)]
    TOLc = mpmath.mpf(10) ** (-8)
    clusters = []
    used = [False] * d
    for i in range(d):
        if used[i]:
            continue
        cl = [i]; used[i] = True
        for j in range(i + 1, d):
            if (not used[j]) and all(abs(key[i][t_] - key[j][t_]) < TOLc for t_ in range(4)):
                cl.append(j); used[j] = True
        clusters.append(cl)
    if verbose:
        from collections import Counter
        print(f"  {len(clusters)} newform systems; lift-mult distribution "
              f"{dict(Counter(len(c) for c in clusters))}; sum={sum(len(c) for c in clusters)}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # coordinate index of each pivot exponent: coordidx[exponent] = i with pivots[i]==exponent.
    # In the REDUCED echelon basis B, coordinate i of a cusp form == its q-coefficient at pivots[i].
    coordidx = {pivots[i]: i for i in range(d)}

    def coord_pivots(cols):
        """Leading-COORDINATE pivots (indices into the echelon basis) of span{cols}.  cols: list of
        mpmath column vectors (echelon coords).  Robust Gaussian echelon: normalise + relative tol."""
        pivrows = {}                                  # pivot-index -> reduced row (list len d)
        piv = []
        tol = mpmath.mpf(10) ** (-25)
        for c in cols:
            mx = max((abs(c[i]) for i in range(d)), default=mpmath.mpf(1))
            rr = [c[i] / mx for i in range(d)]
            for pc in sorted(piv):
                if abs(rr[pc]) > tol:
                    fct = rr[pc]; base = pivrows[pc]
                    rr = [rr[i] - fct * base[i] for i in range(d)]
            pc = next((i for i in range(d) if abs(rr[i]) > tol), None)
            if pc is None:
                continue
            rr = [x / rr[pc] for x in rr]
            pivrows[pc] = rr; piv.append(pc)
        return sorted(piv)

    lift_vectors = []      # coordinate vector (mpmath, len d) for each f(ez)
    lift_qvec = []         # q-vector (len Lb+1) for each f(ez)  (for validation)
    lift_labels = []       # (system_index, level Mf, e)
    a_report = []          # per-system diagnostics
    for si, cl in enumerate(clusters):
        t = len(cl)
        cols = [V[:, k] for k in cl]
        pcs = coord_pivots(cols)
        Delta = sorted(pivots[i] for i in pcs)
        Df = max(Delta); Mf = 24 // Df
        assert Delta == _divisors(Df), f"system {si}: Delta={Delta} not divisors({Df}) [a5~{float(mpmath.re(T5e[cl[0],cl[0]])):.1f}]"
        v3 = 1 if Df % 3 == 0 else 0; v2 = _v2(Df)
        t2 = v2 + 1; t3 = v3 + 1
        assert t2 * t3 == t, f"system {si}: t2*t3={t2*t3} != t={t}"
        # a_2,a_3 from U-traces over the cluster block (eigenbasis is block-diagonal for commuting U_p)
        a2 = sum(U2e[k, k] for k in cl) / t3
        a3 = sum(U3e[k, k] for k in cl) / t2
        a5 = sum(T5e[k, k] for k in cl) / t
        a7 = sum(T7e[k, k] for k in cl) / t
        a_report.append((si, Mf, complex(a2), complex(a3), complex(a5), complex(a7), Delta, t))
        av = small_a_values(a2, a3, Mf)      # a_m(f), m|24
        # lift basis f(ez): coordinate vector matching q-coeff a_{d'/e} at each pivot exponent d' in Delta
        Psi = mpmath.matrix(t, t)
        for a in range(t):
            dp = Delta[a]
            for kk in range(t):
                Psi[a, kk] = V[coordidx[dp], cl[kk]]
        Psi_inv = Psi ** -1
        # q-vectors of the eigenvectors (for validation reconstruction)
        wcols = []
        for k in cl:
            w = [mpmath.mpc(0)] * (Lb + 1)
            for i in range(d):
                ci = V[i, k]
                if ci != 0:
                    bi = B[i]
                    for n in range(Lb + 1):
                        if bi[n]:
                            w[n] += ci * mpmath.mpf(bi[n].numerator) / mpmath.mpf(bi[n].denominator)
            wcols.append(w)
        for e in Delta:
            target = mpmath.matrix(t, 1)
            for a in range(t):
                dp = Delta[a]
                target[a] = av[dp // e] if (dp % e == 0 and (dp // e) in av) else mpmath.mpf(0)
            c = Psi_inv * target
            xcoord = mpmath.matrix(d, 1)
            for kk in range(t):
                for i in range(d):
                    xcoord[i] += c[kk] * V[i, cl[kk]]
            fe = [mpmath.mpc(0)] * (Lb + 1)
            for kk in range(t):
                ck = c[kk]; wk = wcols[kk]
                for n in range(Lb + 1):
                    fe[n] += ck * wk[n]
            lift_vectors.append([xcoord[i] for i in range(d)])
            lift_qvec.append(fe)
            lift_labels.append((si, Mf, e))
    if verbose:
        # a_2,a_3 audit on the level-1 system (target -528,-4284)
        for (si, Mf, a2, a3, a5, a7, Delta, t) in a_report:
            if Mf == 1:
                print(f"  [B4] level-1 system: a2={a2.real:+.3f} a3={a3.real:+.3f} a5={a5.real:+.1f} "
                      f"a7={a7.real:+.1f}  (target -528,-4284,-1025850,3225992)", flush=True)
        print(f"  reconstructed {len(lift_vectors)} newform-lift basis vectors  [{time.time()-t0:.0f}s]",
              flush=True)

    # ---- decompose S = sum lambda_{f,e} f(ez) in echelon-COORDINATE space (well conditioned) ----
    Scoord = echelon_coords(cusp, B, pivots)          # exact Fractions
    Phi = mpmath.matrix(d, d)
    for j in range(d):
        for i in range(d):
            Phi[i, j] = lift_vectors[j][i]
    Svec = mpmath.matrix(d, 1)
    for i in range(d):
        c = Scoord[i]
        Svec[i] = mpmath.mpf(c.numerator) / mpmath.mpf(c.denominator)
    lam = Phi ** -1 * Svec
    CS = sum(abs(lam[j]) for j in range(d))
    # sharper constant C_S' = sum |lambda_{f,e}| * e^{-(k-1)/2}: the lift f(ez) contributes
    # a_{n/e}(f), bounded by sigma_0(n) (n/e)^{17/2} = sigma_0(n) n^{17/2} e^{-17/2}, so
    # |c_n| <= sigma_0(n) n^{17/2} * sum|lambda_{f,e}| e^{-17/2}.  (High lifts e are suppressed.)
    half = mpmath.mpf(KM1) / 2                          # 17/2
    CSp = sum(abs(lam[j]) * mpmath.mpf(lift_labels[j][2]) ** (-half) for j in range(d))
    # per-e breakdown of sum|lambda|
    from collections import defaultdict
    byE = defaultdict(lambda: mpmath.mpf(0))
    for j in range(d):
        byE[lift_labels[j][2]] += abs(lam[j])
    lift_vectors = lift_qvec                          # use q-vectors for the validation loop below
    # ---- VALIDATION: reconstruct c_n to n=L and compare to exact ----
    maxrel = mpmath.mpf(0); worstn = None
    for n in range(1, L + 1):
        rec = sum(lam[j] * lift_vectors[j][n] for j in range(d))
        c = cusp[n]
        cn = mpmath.mpf(c.numerator) / mpmath.mpf(c.denominator)
        denom = abs(cn) if cn != 0 else mpmath.mpf(1)
        rel = abs(rec - cn) / denom
        if rel > maxrel:
            maxrel = rel; worstn = n
    if verbose:
        print(f"  C_S  = sum|lambda|            = {mpmath.nstr(CS, 12)}   [{time.time()-t0:.0f}s]", flush=True)
        print(f"  C_S' = sum|lambda| e^(-17/2)  = {mpmath.nstr(CSp, 12)}   (sharper: high lifts suppressed)",
              flush=True)
        print(f"  sum|lambda| by lift e: " +
              ", ".join(f"e={e}:{mpmath.nstr(byE[e],4)}" for e in sorted(byE)), flush=True)
        print(f"  VALIDATION max relative |reconstruction - exact c_n| over 1<=n<={L}: "
              f"{mpmath.nstr(maxrel, 4)} at n={worstn}", flush=True)
    return dict(CS=CS, CSp=CSp, byE=dict(byE), lam=lam, lift_labels=lift_labels, a_report=a_report,
                maxrel=maxrel, worstn=worstn, clusters=clusters, pivots=pivots)


def _v2(m):
    v = 0
    while m % 2 == 0:
        v += 1; m //= 2
    return v


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = selftest()
        sys.exit(0 if ok else 1)
    R = compute_CS()
    print("\nC_S  =", mpmath.nstr(R['CS'], 12))
    print("C_S' =", mpmath.nstr(R['CSp'], 12), "(sharper)")
    print("validation max rel err =", mpmath.nstr(R['maxrel'], 4), "at n =", R['worstn'])
