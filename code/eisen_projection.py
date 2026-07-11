#!/usr/bin/env python3
r"""
EXACT Eisenstein projection  P_E : M_18(Gamma_0(24)) -> E_18 = span{E_18(delta z): delta|24}.

Given a modular form g in M_18(Gamma_0(24)) by its exact q-expansion [a_0,...,a_M]
(sympy Integers/Rationals, M >= 72), returns the 8 EXACT Eisenstein coefficients
    e = (e_1,e_2,e_3,e_4,e_6,e_8,e_12,e_24)
such that  g - sum_delta e_delta * E_18(delta z)  is a CUSP form (lies in S_18).

Here  E_18(z) = 1 + COEF18 * sum_{n>=1} sigma_17(n) q^n,  COEF18 = -2*18/B_18 = -28728/43867,
and E_18(delta z) is the q -> q^delta rescaling.

------------------------------------------------------------------------------------------
METHOD -- HECKE-OPERATOR SPECTRAL PROJECTION (exact, sympy Rational). ---------------------
------------------------------------------------------------------------------------------
  dim M_18(G0(24)) = 72,  dim Eisenstein = 8 (= #cusps),  dim S_18 = 64.
  E_18(delta z) and cusp forms are BOTH Hecke-eigenstable.  For a prime p coprime to 24,
  T_p acts on q-expansions by   (T_p f)_n = a_{np} + p^{17} a_{n/p}   (a_{n/p}=0 if p!|n).
  Every E_18(delta z) is a T_p-eigenform with eigenvalue  1 + p^{17}; every cusp eigenform
  has |eigenvalue| <= 2 p^{17/2} (Deligne).  With p = 5 (smallest prime coprime to 24):
        Eisenstein eigenvalue = 1 + 5^17 = 762939453126        (~7.6e11)
        cusp eigenvalues       <= 2 * 5^8.5 ~ 7.6e5             -- a clean spectral gap.
  Hence  E_18 = ker(T_5 - (1+5^17) I)  is EXACTLY the 8-dim Eisenstein subspace, and the
  spectral projector onto it (along the complementary Hecke-stable cusp space S_18) is P_E.

  Realisation (all EXACT / rational -- floats are useless here: weight-18 coefficients span
  ~10^35 and float rank/inverse silently mis-rank, cf. ct_dual_d36_hiprec.py):
   1. CANDIDATES = the 8 Eisenstein forms E_18(delta z) + all holomorphic weight-18
      eta-quotients on Gamma_0(24) with |r|<=8 (Ligozat; D.enumerate_etaq).  Expanded with
      the FAST EXACT (python-Fraction) path (D._etaq_fast / D._Ek_fast) to length L=100.
   2. Pick 72 EXACTLY-independent forms by exact rational Gauss elimination (candidate order),
      recording the PIVOT coefficient-indices PIV (the coordinate window that determines a
      form).  |PIV| = 72; here max(PIV) = 72 (the Sturm bound floor(18*[SL2:G0(24)]/12)=72
      guarantees a determining window exists; the exact pivots need not be {0..71}).
   3. Expand ONLY those 72 chosen specs to length  M = 5*max(PIV)+1  with EXACT sympy
      Rationals (needed so (T_5 f)_n = a_{5n}+5^17 a_{n/5} is available on the window).
   4. PHI = 72x72 exact matrix of the 72 forms on the window PIV (invertible by step 2).
      T5 = PHIinv * IMG (IMG = images (T_5 form_j) on the window) is the exact T_5 matrix on
      the coordinate basis.
   5. E_18 in coordinate space is spanned by the 8 UNIT coordinate vectors of the Eisenstein
      basis forms (each E_18(delta z) is a basis form == a unit coordinate vector, and is a
      T_5-eigenvector with the Eisenstein eigenvalue).  The EISENSTEIN-COEFFICIENT extractor
      is the matching set of LEFT eigenvectors: rows L_delta of an 8x72 matrix with
      L * (T5) = (1+5^17) * L   (== nullspace of (T5^T - eig I)), normalised so
      L_delta . (unit vector of E_18(delta' z)) = kronecker(delta,delta').  Left eigenvectors
      for the Eisenstein eigenvalue automatically annihilate the cusp space (left/right
      eigenspaces for distinct eigenvalues are orthogonal) -- so L is the exact P_E read-off.
   6. Compose with the coordinate map (x = PHIinv . a):  PROJ = Lnorm * PHIinv  is the exact
      8 x 72 rational matrix with   e_delta = (PROJ . a[PIV])_delta.

  The build runs ONCE (~130s, dominated by the exact sympy expansion of the 72 forms to
  M=361; the linear algebra is <1s) and PROJ (+ PIV) is cached to disk (~18 KB of exact
  rationals).  eisen_coeffs() then only multiplies an 8x72 rational matrix by 72 coefficients
  -- fully exact; a cached self-verification run finishes in ~6s.

Run:   PYTHONPATH=<...C1.../code> python3 eisen_projection.py            # load/build + self-test
       PYTHONPATH=<...C1.../code> python3 eisen_projection.py --rebuild  # force rebuild

Pure-mathematics research; standard modular-forms jargon.
"""
from __future__ import annotations
import os, sys, time, pickle
from fractions import Fraction as Fr
import sympy as sp

import ct_dual_general as Gg
import ct_dual_d36 as D          # importing D monkeypatches Gg.E_k to the weight-18 table

# ---------------------------------------------------------------------------
# problem constants
# ---------------------------------------------------------------------------
N        = 24
K        = 18
DELTAS   = [1, 2, 3, 4, 6, 8, 12, 24]      # divisors of 24 = D.divisors(24)
DIMM     = 72                              # dim M_18(Gamma_0(24))
HECKE_P  = 5                               # smallest prime coprime to 24
BMAX     = 8                               # |r| bound in the eta-quotient enumeration
LSEL     = 100                             # coefficient length used for exact form-selection
COEF18   = sp.Rational(-28728, 43867)      # -2*18/B_18   (E_18 leading coefficient)
EIS_EIG  = 1 + HECKE_P ** (K - 1)          # 1 + 5^17 = Eisenstein T_5 eigenvalue

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eisen_projection_PROJ.pkl")


# ---------------------------------------------------------------------------
# Eisenstein "pattern" coefficients:  E_18(delta z)_n  (exact, any length, cheap)
#   n=0 -> 1 ;  else COEF18 * sigma_17(n/delta) if delta|n else 0
# ---------------------------------------------------------------------------
def eisen_pattern(delta, M):
    out = [sp.Integer(0)] * (M + 1)
    out[0] = sp.Integer(1)
    for n in range(1, M + 1):
        if n % delta == 0:
            out[n] = COEF18 * sp.divisor_sigma(n // delta, K - 1)
    return out


# ---------------------------------------------------------------------------
# Step 1-2: exact selection of 72 independent forms + their pivot coefficient-indices.
# Uses ONLY the fast exact python-Fraction q-expansions (no floats, no sympy rank).
# ---------------------------------------------------------------------------
def _fast_qexp(spec, L):
    if spec[0] == "eis":
        return [Fr(x.p, x.q) if hasattr(x, "p") else Fr(x) for x in D._shift(D._Ek_fast(K, L), spec[1], L)]
    return [Fr(x) for x in D._etaq_fast(spec[1], L)]

def select_basis(L=LSEL, want=DIMM):
    """Return (specs, PIV): 'want' exactly-independent forms (as specs) and the sorted list
    of pivot coefficient-indices that determine a form (a coordinate window)."""
    Ddiv = D.divisors(N)
    _, sols = D.enumerate_etaq(N, K, BMAX)
    cand = [("eis", dd) for dd in Ddiv] + \
           [("eta", {Ddiv[i]: e[i] for i in range(len(Ddiv)) if e[i] != 0}) for e in sols]
    chosen, pivcols, redrows = [], [], []       # redrows: reduced (echelon) Fraction rows
    for idx, s in enumerate(cand):
        v = _fast_qexp(s, L)
        for b, pc in zip(redrows, pivcols):
            if v[pc] != 0:
                f = v[pc]
                v = [v[i] - f * b[i] for i in range(len(v))]
        pc = next((i for i in range(len(v)) if v[i] != 0), None)
        if pc is None:
            continue
        inv = Fr(1) / v[pc]
        v = [x * inv for x in v]
        redrows.append(v); pivcols.append(pc); chosen.append(s)
        if len(chosen) == want:
            break
    if len(chosen) != want:
        raise RuntimeError(f"only found {len(chosen)} independent forms (< {want})")
    return chosen, sorted(pivcols)


# ---------------------------------------------------------------------------
# Step 3-6: build the exact 8 x 72 projection matrix PROJ (one-time, cached).
# ---------------------------------------------------------------------------
def build_proj(verbose=True):
    t0 = time.time()
    specs, PIV = select_basis()
    assert len(specs) == DIMM and len(PIV) == DIMM
    Mexp = HECKE_P * max(PIV) + 1
    # index within specs of each Eisenstein basis form
    eis_index = {s[1]: j for j, s in enumerate(specs) if s[0] == "eis"}
    assert set(eis_index) == set(DELTAS), f"Eisenstein forms not all selected: {eis_index}"
    if verbose:
        print(f"  selected 72 independent forms (max pivot {max(PIV)}); "
              f"exact expansion length M={Mexp}", flush=True)

    # exact sympy q-expansions of the 72 chosen specs to length Mexp
    t = time.time()
    g_list, _ = Gg.build_basis(specs, K, N, Mexp)
    if verbose:
        print(f"  exact expansion of 72 forms to M={Mexp} in {time.time()-t:.1f}s", flush=True)

    # PHI: coordinate matrix on the pivot window.  PHI[a, j] = (form_j)_{PIV[a]}
    t = time.time()
    PHI = sp.Matrix(DIMM, DIMM, lambda a, j: g_list[j][PIV[a]])
    PHIinv = PHI.inv()
    if verbose:
        print(f"  coordinate inverse (72x72) in {time.time()-t:.1f}s", flush=True)

    # exact T_5 matrix on the coordinate basis:
    #   (T_5 form_j)_n = a_{5n} + 5^17 a_{n/5}.  IMG[a,j] = image coeff at index PIV[a].
    t = time.time()
    p, p17 = HECKE_P, HECKE_P ** (K - 1)

    def Tp_image_coeff(gj, n):
        v = gj[n * p] if n * p <= Mexp else sp.Integer(0)
        if n % p == 0:
            v = v + p17 * gj[n // p]
        return v

    IMG = sp.Matrix(DIMM, DIMM, lambda a, j: Tp_image_coeff(g_list[j], PIV[a]))
    T5 = PHIinv * IMG                          # columns = coords of T_5(form_j)
    if verbose:
        print(f"  exact T_5 matrix (72x72) in {time.time()-t:.1f}s", flush=True)

    # sanity: the 8 Eisenstein basis forms are unit coordinate vectors and T_5-eigenvectors
    for d in DELTAS:
        col = T5[:, eis_index[d]]
        assert col == EIS_EIG * sp.eye(DIMM)[:, eis_index[d]], \
            f"E_18({d}z) is not a clean T_5-eigenvector (basis not aligned)"

    # LEFT eigenvectors for the Eisenstein eigenvalue = nullspace of (T5^T - eig I).
    t = time.time()
    Lrows = (T5.T - EIS_EIG * sp.eye(DIMM)).nullspace()
    assert len(Lrows) == 8, f"left Eisenstein eigenspace dim = {len(Lrows)} (expected 8)"
    Lmat = sp.Matrix.hstack(*Lrows).T          # 8 x 72
    # normalise so  Lmat . (unit vec of E_18(delta z)) = kronecker  =>
    #   Lmat[:, eis_cols] is 8x8; multiply by its inverse.
    eis_cols = [eis_index[d] for d in DELTAS]
    Gsub = Lmat[:, eis_cols]                    # 8 x 8
    Lnorm = Gsub.inv() * Lmat                   # rows now dual to the eis unit vectors
    if verbose:
        print(f"  left-eigenvector read-off (P_E) in {time.time()-t:.1f}s", flush=True)

    # PROJ : e = Lnorm . x = Lnorm . (PHIinv . a[PIV])  =>  PROJ = Lnorm * PHIinv  (8 x 72)
    PROJ = Lnorm * PHIinv
    PROJ = sp.Matrix(PROJ)
    if verbose:
        print(f"  PROJ (8x72) assembled; total build {time.time()-t0:.1f}s", flush=True)
    return PROJ, PIV


def _load_proj(rebuild=False, verbose=True):
    if (not rebuild) and os.path.exists(_CACHE):
        with open(_CACHE, "rb") as f:
            data = pickle.load(f)
        PROJ = sp.Matrix([[sp.Rational(x) for x in row] for row in data["PROJ"]])
        PIV = list(data["PIV"])
        return PROJ, PIV
    PROJ, PIV = build_proj(verbose=verbose)
    with open(_CACHE, "wb") as f:
        pickle.dump({"PROJ": [[str(PROJ[i, j]) for j in range(PROJ.cols)] for i in range(PROJ.rows)],
                     "PIV": PIV}, f)
    return PROJ, PIV


# module-level lazy singleton
_PROJ = None
_PIV = None
def _proj():
    global _PROJ, _PIV
    if _PROJ is None:
        _PROJ, _PIV = _load_proj()
    return _PROJ, _PIV


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def eisen_coeffs(gcoeffs, M=None):
    """EXACT Eisenstein coefficients of g in M_18(Gamma_0(24)).

    gcoeffs : list [a_0, a_1, ..., a_M] of exact sympy Integers/Rationals.
    returns : {delta: e_delta}  (delta in {1,2,3,4,6,8,12,24}, sympy Rationals) with
              g - sum_delta e_delta E_18(delta z)  a cusp form.
    """
    P, PIV = _proj()
    need = max(PIV)
    if len(gcoeffs) <= need:
        raise ValueError(f"need at least {need + 1} coefficients (indices 0..{need}); "
                         f"got {len(gcoeffs)}")
    a = sp.Matrix(DIMM, 1, lambda a_, _: sp.nsimplify(gcoeffs[PIV[a_]]))
    e = P * a
    return {DELTAS[i]: sp.nsimplify(e[i]) for i in range(8)}


# ---------------------------------------------------------------------------
# SELF-VERIFICATION
# ---------------------------------------------------------------------------
def _find_cuspidal_spec():
    """A holomorphic eta-quotient with ord_c > 0 at EVERY cusp (=> a genuine cusp form)."""
    Ddiv, coefC = D.cusp_order_coeffs(N)
    _, sols = D.enumerate_etaq(N, K, BMAX)
    for e in sols:
        exps = {Ddiv[i]: e[i] for i in range(len(Ddiv)) if e[i] != 0}
        if all(sum(coefC[c][i] * e[i] for i in range(len(Ddiv))) > 0 for c in Ddiv):
            orders = [int(sum(coefC[c][i] * exps.get(Ddiv[i], 0) for i in range(len(Ddiv)))) for c in Ddiv]
            return exps, Ddiv, orders
    raise RuntimeError("no all-cusp-positive eta-quotient found")


def _selftest():
    print("=" * 78)
    print("SELF-VERIFICATION of eisen_coeffs (EXACT, sympy Rational)")
    print("=" * 78)
    _, PIV = _proj()
    Mv = max(PIV) + 20                        # comfortably above the window
    ok_all = True

    # ---- (a) identity on each E_18(delta z): e = unit vector ----
    print("\n(a) eisen_coeffs(E_18(delta z)) == unit vector e_delta:")
    a_ok = True
    for d in DELTAS:
        e = eisen_coeffs(eisen_pattern(d, Mv), Mv)
        want = {dd: (sp.Integer(1) if dd == d else sp.Integer(0)) for dd in DELTAS}
        good = all(e[dd] == want[dd] for dd in DELTAS)
        a_ok = a_ok and good
        nz = {dd: e[dd] for dd in DELTAS if e[dd] != 0}
        print(f"    delta={d:2d}: nonzero coords = {nz}   {'OK' if good else 'FAIL'}")
    print(f"  (a) => {'PASS' if a_ok else 'FAIL'}")
    ok_all = ok_all and a_ok

    # ---- (b) a genuine CUSP form -> all zeros ----
    print("\n(b) eisen_coeffs(cuspidal eta-quotient) == all zeros:")
    cusp_spec, Ddiv, orders = _find_cuspidal_spec()
    hcoeffs = Gg.eta_quotient_qexp(cusp_spec, Mv)
    eb = eisen_coeffs(hcoeffs, Mv)
    b_ok = all(eb[d] == 0 for d in DELTAS)
    print(f"    cuspidal eta-quotient exps = {cusp_spec}")
    print(f"    ord_c at the 8 cusps {Ddiv} = {orders}  (all > 0 => cuspidal)")
    print(f"    eisen_coeffs = {dict(eb)}")
    print(f"  (b) => {'PASS' if b_ok else 'FAIL'}")
    ok_all = ok_all and b_ok

    # ---- (c) linearity: eisen_coeffs(2f+3h) == 2 eisen(f) + 3 eisen(h) ----
    print("\n(c) linearity  eisen_coeffs(2f+3h) == 2*eisen(f)+3*eisen(h):")
    f = eisen_pattern(2, Mv)                          # Eisenstein input
    h = hcoeffs                                       # cusp input
    mix = [2 * f[n] + 3 * h[n] for n in range(Mv + 1)]
    em = eisen_coeffs(mix, Mv); ef = eisen_coeffs(f, Mv); eh = eisen_coeffs(h, Mv)
    c_ok = all(em[d] == 2 * ef[d] + 3 * eh[d] for d in DELTAS)
    print(f"    eisen(2f+3h)          = {dict(em)}")
    print(f"    2*eisen(f)+3*eisen(h) = { {d: 2*ef[d]+3*eh[d] for d in DELTAS} }")
    print(f"  (c) => {'PASS' if c_ok else 'FAIL'}")
    ok_all = ok_all and c_ok

    # ---- (extra) planted mixture recovered exactly (non-cusp input, mixed deltas) ----
    print("\n(extra) g = 7*E_18(z) - 5*E_18(6z) + 11*E_18(24z) + (cusp):")
    g = [7 * eisen_pattern(1, Mv)[n] - 5 * eisen_pattern(6, Mv)[n]
         + 11 * eisen_pattern(24, Mv)[n] + h[n] for n in range(Mv + 1)]
    eg = eisen_coeffs(g, Mv)
    want = {1: sp.Integer(7), 6: sp.Integer(-5), 24: sp.Integer(11)}
    x_ok = all(eg[d] == want.get(d, 0) for d in DELTAS)
    print(f"    recovered = {dict(eg)}   {'OK' if x_ok else 'FAIL'}")
    ok_all = ok_all and x_ok

    # ---- (extra2) the residual g - sum e_delta E_18(delta z) really is a cusp form:
    #      it must be expressible as a combination of the (all-cusp-positive) cusp space,
    #      equivalently its Eisenstein projection is 0. Re-run eisen_coeffs on the residual.
    print("\n(extra2) residual of the planted mixture is a cusp form (its P_E == 0):")
    resid = [g[n] - sum(int(want.get(d, 0)) * eisen_pattern(d, Mv)[n] for d in DELTAS)
             for n in range(Mv + 1)]
    er = eisen_coeffs(resid, Mv)
    r_ok = all(er[d] == 0 for d in DELTAS)
    print(f"    P_E(residual) = {dict(er)}   {'OK (cuspidal)' if r_ok else 'FAIL'}")
    ok_all = ok_all and r_ok

    print("\n" + "=" * 78)
    print(f"OVERALL: {'ALL PASS' if ok_all else 'FAILURE'}   "
          f"[(a) {'PASS' if a_ok else 'FAIL'}, (b) {'PASS' if b_ok else 'FAIL'}, "
          f"(c) {'PASS' if c_ok else 'FAIL'}]")
    print("=" * 78)
    return ok_all


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    t0 = time.time()
    if rebuild and os.path.exists(_CACHE):
        os.remove(_CACHE)
    cached = os.path.exists(_CACHE)
    print(f"{'Loading' if cached else 'Building'} exact projector PROJ (8x72) ...", flush=True)
    tp = time.time()
    _proj()
    print(f"PROJ ready in {time.time()-tp:.1f}s ({'from cache' if cached else 'built + cached to disk'}).\n")
    ok = _selftest()
    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    sys.exit(0 if ok else 1)
