#!/usr/bin/env python3
r"""EXACT certification that the candidate list (8 Eisenstein + 3044 eta-quotients, |r|<=8)
spans M_18(Gamma_0(24)):  rank of the truncated coefficient matrix mod p = 72.

Logic: rank_{F_p}(A mod p) <= rank_Q(A) <= dim M_18(Gamma_0(24)) = 72 (dimension formula).
So a mod-p rank of 72 proves exact rank 72, i.e. spanning. Truncation (n<=100) can only
LOWER the rank, never inflate it, so the certificate is one-sided-safe.
E_18 columns are scaled by the denominator of kappa_18 (scaling preserves rank).
Replaces the construction-stage floating-point rank check with an exact certificate
(paper Sec 3.1; v2 referee pass). Pure-mathematics research; standard jargon.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ct_dual_d36 as D

N, K, B, L = 24, 18, 8, 100
P = 1000000007
t0 = time.time()

_, sols = D.enumerate_etaq(N, K, B)
DIVS = [1, 2, 3, 4, 6, 8, 12, 24]
cand = [("eis", dd) for dd in DIVS] + \
       [("eta", {DIVS[i]: e[i] for i in range(len(DIVS)) if e[i] != 0}) for e in sols]
print(f"candidates: 8 Eisenstein + {len(sols)} eta-quotients (|r|<={B}) = {len(cand)}")

def modrow(vals):
    out = []
    for v in vals:
        try:
            num, den = v.numerator, v.denominator
        except AttributeError:
            num, den = int(v), 1
        out.append((num % P) * pow(den % P, -1, P) % P)
    return out

rows = []
ek = D._Ek_fast(K, L)
for s in cand:
    if s[0] == "eis":
        delta = s[1]
        rows.append(modrow([ek[m // delta] if m % delta == 0 else 0 for m in range(L + 1)]))
    else:
        rows.append(modrow(D._etaq_fast(s[1], L)))
print(f"exact q-expansions to n<={L}, reduced mod {P}  [{time.time()-t0:.0f}s]")

# Gaussian elimination mod p over the (len(cand) x L+1) matrix
ncols = L + 1
rank, pivrow = 0, 0
work = rows  # eliminate in place, column by column
nrows = len(work)
for col in range(ncols):
    piv = next((i for i in range(pivrow, nrows) if work[i][col]), None)
    if piv is None:
        continue
    work[pivrow], work[piv] = work[piv], work[pivrow]
    inv = pow(work[pivrow][col], -1, P)
    work[pivrow] = [(v * inv) % P for v in work[pivrow]]
    prow = work[pivrow]
    for i in range(nrows):
        if i != pivrow and work[i][col]:
            f = work[i][col]
            wi = work[i]
            work[i] = [(wi[k] - f * prow[k]) % P for k in range(ncols)]
    pivrow += 1
    rank = pivrow
    if rank == 72:
        break
print(f"rank mod {P} of the truncated candidate matrix = {rank}  [{time.time()-t0:.0f}s]")
print(f"dim M_18(Gamma_0(24)) = 72 (dimension formula)")
ok = (rank == 72)
print(f"==> exact rank = 72, candidates SPAN M_18(Gamma_0(24)): {'CERTIFIED' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
