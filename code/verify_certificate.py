#!/usr/bin/env python3
r"""verify_certificate.py — single-driver rigorous verification of the d=36 dual certificate.

Reconstructs everything from the PUBLISHED TEXT DATA (../certificate_exact_data.txt) alone:
the 29 basis forms are re-expanded exactly from the printed form list (NOT from any cache),
the reduced basis is the exact rational nullspace of the printed 9 vanishing conditions, and
the certificate form g is rebuilt from the printed 20 coordinates. All gates are exact
rational comparisons. Prints VERIFIED only if every gate passes.

Gates:
  G1  form list parses to 29 forms; nullspace of (a_1..a_9) has exact dimension r=20
  G1b the 29 selected forms are linearly independent (rank 29 mod 10^9+7 on n<=100
      => exact rank 29); rank certificates: rank(a_1..a_9)=9 (from G1), and the
      10x29 matrix of (a_0..a_9) has exact rank 10 (a_0 independent of the conditions)
  G2  a_0 = 1 and a_1 = ... = a_9 = 0 exactly
  G3  a_n >= 0 for 10 <= n <= 800 (exact, zero negatives)
  G4  b_n >= 0 for  0 <= n <= 800 (exact, zero negatives)
  G5  b_0 and B = b_0 * 5^18/(2^27*3^9) equal the published exact fractions;
      B > 146.1036 (exact comparison); published decimal enclosures contain b_0, B
  G6  exact Eisenstein projection of g~ reproduces the eight published e_delta(g~)
      identically; r3 = e_1 + e_3/(1+3^17) matches the published exact value
  G7  guarded per-class Eisenstein constants c_E(gamma) (paper eq. (guard)) are positive
      for all 8 classes on BOTH sides; rigorous integer crossovers n0 = 63 (g~) / 25 (g)
      with the certified C_w upper bounds 358807778339/10^12, 306329333530/10^12

The heavy interval certification of the C_w upper bounds themselves is a separate step:
run  python3 d36_cs_certificate.py  (see ../MANIFEST.md, "proof core").

Usage:  python3 verify_certificate.py      (~ a few minutes, pure sympy/fractions)
Pure-mathematics research; standard modular-forms / LP-dual jargon.
"""
import os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import sympy as sp

import ct_dual_d36 as D
import ct_dual_general as Gg
import eisen_projection as EP

N, K, T, M, L = 24, 18, 10, 28, 800
HERE = os.path.dirname(os.path.abspath(__file__))
# Locate the published certificate data, robust to bundle layout:
#   ancillary/code/ + ancillary/certificate_exact_data.txt   (this repo's ancillary bundle), or
#   code/ + data/certificate_exact_data.txt                  (flat paper/code/data layout).
def _find_cert():
    for rel in ('../certificate_exact_data.txt',
                '../data/certificate_exact_data.txt',
                'certificate_exact_data.txt'):
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, '..', 'certificate_exact_data.txt')  # default (will error clearly)
CERT = _find_cert()
DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
KAPPA = sp.Rational(-28728, 43867)
Z17 = sp.Integer(1) + sp.Rational(1, 2**17) + sp.Rational(1, 2**16 * 16)

FAIL = []
def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)

# ---------------------------------------------------------------- parse published data
def parse_cert(path):
    txt = open(path).read()
    forms = []
    for m in re.finditer(r"^form\[\s*(\d+)\] = (Eis  E_18\(q\^(\d+)\)|eta  (\{[^}]*\}))",
                         txt, re.M):
        j = int(m.group(1))
        if m.group(2).startswith("Eis"):
            forms.append((j, ("eis", int(m.group(3)))))
        else:
            forms.append((j, ("eta", {int(k): int(v) for k, v in
                                      re.findall(r"(\d+):\s*(-?\d+)", m.group(4))})))
    forms.sort()
    assert [j for j, _ in forms] == list(range(len(forms)))
    specs = [s for _, s in forms]

    xs = {}
    for m in re.finditer(r"^x\[(\d+)\] = (-?\d+)/(\d+)$", txt, re.M):
        xs[int(m.group(1))] = Fr(int(m.group(2)), int(m.group(3)))
    x = [xs[i] for i in range(len(xs))]

    ed = {}
    for m in re.finditer(r"^e_(\d+)\(g~\) = (-?\d+)/(\d+)$", txt, re.M):
        ed[int(m.group(1))] = Fr(int(m.group(2)), int(m.group(3)))

    def one(pat):
        m = re.search(pat, txt, re.M)
        return Fr(int(m.group(1)), int(m.group(2))) if m else None
    b0_pub = one(r"^b_0 = (-?\d+)/(\d+)$")
    r3_pub = one(r"^r3 = (-?\d+)/(\d+)$")
    mB = re.search(r"^B\(g\) = b_0.*\n\s*= (-?\d+)/(\d+)$", txt, re.M)
    B_pub = Fr(int(mB.group(1)), int(mB.group(2))) if mB else None
    return specs, x, ed, b0_pub, B_pub, r3_pub

t0 = time.time()
print("=" * 78)
print("verify_certificate.py — d=36 dual certificate, verification from published data")
print("=" * 78)
specs, x, ed_pub, b0_pub, B_pub, r3_pub = parse_cert(CERT)
print(f"parsed: {len(specs)} forms, {len(x)} coordinates, {len(ed_pub)} e_delta, "
      f"b_0/B/r3 fractions: {b0_pub is not None}/{B_pub is not None}/{r3_pub is not None}")

# ------------------------------------------------- exact q-expansions of the 29 forms
def to_fr(v):
    q = sp.Rational(v)
    return Fr(int(q.p), int(q.q))

gg, gtt = [], []
for s in specs:
    if s[0] == "eis":
        delta = s[1]
        base = D._Ek_fast(K, L)
        gj = [Fr(0)] * (L + 1)
        for m in range(L + 1):
            if m % delta == 0:
                gj[m] = to_fr(base[m // delta])
        C, _ = Gg.fricke_eisen(delta, K, N, L)
        Cf = to_fr(C)
        himg = [Fr(0)] * (L + 1)
        dd = N // delta
        for m in range(L + 1):
            if m % dd == 0:
                himg[m] = to_fr(base[m // dd])
        gtj = [Cf * himg[m] for m in range(L + 1)]
    else:
        exps = s[1]
        gj = [to_fr(v) for v in D._etaq_fast(exps, L)]
        C, _ = Gg.fricke_etaq(exps, K, N, L)
        Cf = to_fr(C)
        rev = {}
        for dl, rr in exps.items():
            rev[N // dl] = rev.get(N // dl, 0) + rr
        himg = [to_fr(v) for v in D._etaq_fast(rev, L)]
        gtj = [Cf * himg[m] for m in range(L + 1)]
    gg.append(gj); gtt.append(gtj)
print(f"exact q-expansions of the {len(specs)} forms and Fricke images to L={L} "
      f"[{time.time()-t0:.0f}s]")

# ------------------------------------------------------- G1: exact reduced nullspace
n = len(specs)
Gm = sp.Matrix(T - 1, n, lambda i, j: sp.Rational(gg[j][i + 1].numerator,
                                                  gg[j][i + 1].denominator))
ns = Gm.nullspace()
Bm = sp.Matrix.hstack(*ns)
gate("G1: 29 forms, nullspace(a_1..a_9) has dim r=20",
     n == 29 and Bm.shape == (29, 20), f"n={n}, Bm={Bm.shape}")

# G1b: exact independence of the 29 forms (mod-p rank on n<=100: rank_p <= rank_Q <= 29,
# so rank_p = 29 proves exact rank 29) + exact rank 10 of the (a_0..a_9) matrix.
PMOD = 1000000007
def modval(fr):
    return (fr.numerator % PMOD) * pow(fr.denominator % PMOD, -1, PMOD) % PMOD
rows_p = [[modval(gg[j][m]) for m in range(101)] for j in range(n)]
rank_p, pr = 0, 0
for col in range(101):
    piv = next((i for i in range(pr, n) if rows_p[i][col]), None)
    if piv is None:
        continue
    rows_p[pr], rows_p[piv] = rows_p[piv], rows_p[pr]
    inv = pow(rows_p[pr][col], -1, PMOD)
    rows_p[pr] = [(v * inv) % PMOD for v in rows_p[pr]]
    for i in range(n):
        if i != pr and rows_p[i][col]:
            f = rows_p[i][col]
            rows_p[i] = [(rows_p[i][k] - f * rows_p[pr][k]) % PMOD for k in range(101)]
    pr += 1
    rank_p = pr
G10 = sp.Matrix(T, n, lambda i, j: sp.Rational(gg[j][i].numerator, gg[j][i].denominator))
rank10 = G10.rank()
gate("G1b: 29 forms independent (rank 29 mod p => exact); rank(a_0..a_9) = 10",
     rank_p == 29 and rank10 == 10, f"rank_p={rank_p}, rank(a0..a9)={rank10}")

# form coefficients y = Bm x (exact Fractions)
y = []
for i in range(n):
    acc = Fr(0)
    for jj in range(Bm.shape[1]):
        c = Bm[i, jj]
        if c != 0:
            acc += Fr(int(sp.Rational(c).p), int(sp.Rational(c).q)) * x[jj]
    y.append(acc)

a = [sum(y[j] * gg[j][m] for j in range(n)) for m in range(L + 1)]
b = [sum(y[j] * gtt[j][m] for j in range(n)) for m in range(L + 1)]
print(f"certificate q-expansions assembled [{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------------- G2-G4: gates
gate("G2: a_0 = 1 and a_1..a_9 = 0 exactly",
     a[0] == 1 and all(a[m] == 0 for m in range(1, T)))
neg_a = [m for m in range(T, L + 1) if a[m] < 0]
gate("G3: a_n >= 0 for 10 <= n <= 800 (exact)", not neg_a,
     f"negatives: {len(neg_a)}")
neg_b = [m for m in range(0, L + 1) if b[m] < 0]
gate("G4: b_n >= 0 for 0 <= n <= 800 (exact)", not neg_b,
     f"negatives: {len(neg_b)}")

# ------------------------------------------------------------------------- G5: value
scale = Fr(5**18, 2**27 * 3**9)
Bval = b[0] * scale
ok5 = (b[0] == b0_pub and Bval == B_pub and Bval > Fr(1461036, 10000)
       and Fr(1011817608012, 10**10) < b[0] < Fr(1011817608013, 10**10)
       and Fr(1461036734821, 10**10) < Bval < Fr(1461036734822, 10**10))
gate("G5: b_0, B match published exact fractions; B > 146.1036; enclosures", ok5)

# -------------------------------------------------- G6: exact Eisenstein projection
P, PIV = EP._proj()
def project(coeffs):
    v = sp.Matrix(72, 1, lambda i, _: sp.Rational(coeffs[PIV[i]].numerator,
                                                  coeffs[PIV[i]].denominator))
    e = P * v
    return {DIVS[i]: sp.Rational(e[i]) for i in range(8)}
ed_b = project(b)
ed_a = project(a)
ok6 = all(Fr(int(ed_b[dl].p), int(ed_b[dl].q)) == ed_pub[dl] for dl in DIVS)
r3 = ed_b[1] + ed_b[3] / (1 + sp.Integer(3)**17)
ok6 = ok6 and (Fr(int(sp.Rational(r3).p), int(sp.Rational(r3).q)) == r3_pub)
gate("G6: Eisenstein projection of g~ reproduces published e_delta and r3 exactly", ok6)

# --------------------------------------------- G7: guarded c_E + rigorous crossover
def guarded(ed):
    out = {}
    for g_ in DIVS:
        pos = sum(KAPPA * ed[dl] / sp.Integer(dl)**17
                  for dl in DIVS if g_ % dl == 0 and KAPPA * ed[dl] > 0)
        neg = sum(KAPPA * ed[dl] / sp.Integer(dl)**17
                  for dl in DIVS if g_ % dl == 0 and KAPPA * ed[dl] < 0)
        out[g_] = pos + Z17 * neg
    return out
ok7 = True
for side, ed_side, CwR, n0_expect in [
        ("g~", ed_b, sp.Rational(358807778339, 10**12), 63),
        ("g",  ed_a, sp.Rational(306329333530, 10**12), 25)]:
    cE = guarded(ed_side)
    allpos = all(cE[g_] > 0 for g_ in DIVS)
    mn = min(cE.items(), key=lambda kv: kv[1])
    if mn[1] <= 0:
        # corrupted-input backstop: the crossover loop below requires a positive floor
        print(f"    {side}-side: min guarded c_E is NOT positive — gate fails")
        ok7 = False
        continue
    nn = 1
    while mn[1] * sp.Integer(nn)**8 <= 2 * CwR:
        nn += 1
        if nn > 10**6:   # unreachable on valid data (n0=63/25); backstop only
            break
    print(f"    {side}-side: all 8 classes positive: {allpos}; min at gamma={mn[0]}; "
          f"rigorous n0={nn} (expected {n0_expect})")
    ok7 = ok7 and allpos and mn[0] == 3 and nn == n0_expect
gate("G7: guarded c_E positive (16/16), binding gamma=3, rigorous n0 = 63/25", ok7)

print("-" * 78)
if not FAIL:
    print(f"VERIFIED  — all gates passed  [{time.time()-t0:.0f}s]")
    print("(The C_w upper bounds used in G7 carry their own interval certificate:")
    print(" run  python3 d36_cs_certificate.py  and see receipt_d36_cs.txt.)")
else:
    print(f"NOT VERIFIED — failed gates: {FAIL}")
    sys.exit(1)
