#!/usr/bin/env python3
r"""Exact zeta(17)-guarded per-class Eisenstein lower-bound constants c_E(gamma), both sides.
Paper eq. (guard): c_E(g) = sum_{d|g, e_d<0} K*e_d/d^17 + (1+2^-17+2^-16/16) * sum_{d|g, e_d>0} K*e_d/d^17,
K = kappa_18 = -28728/43867.  Uses m^17 <= sigma_17(m) < zeta(17) m^17 and zeta(17) < 1+2^-17+2^-16/16.
All arithmetic exact rational; prints certified decimals. Provenance: paper polish 2026-07-11."""
import os, pickle, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import eisen_projection as EP
DIVS=[1,2,3,4,6,8,12,24]
K=sp.Rational(-28728,43867)
Z17=sp.Integer(1)+sp.Rational(1,2**17)+sp.Rational(1,2**16*16)
CODE=os.path.dirname(os.path.abspath(__file__))
with open(f'{CODE}/d36_iter1_vertex.pkl','rb') as f: VC=pickle.load(f)
P,PIV=EP._proj()
def eis(coeffs):
    v=sp.Matrix(72,1,lambda i,_: sp.Rational(coeffs[PIV[i]])); e=P*v
    return {DIVS[i]: sp.Rational(e[i]) for i in range(8)}
for name,key,Cw in [('b-side (g~)','b',0.358807778339),('a-side (g)','a',0.306329333530)]:
    ed=eis([sp.Rational(p,q) for p,q in VC[key]])
    print(f"== {name} ==")
    mn=None
    for g in DIVS:
        pos=sum(K*ed[d]/sp.Integer(d)**17 for d in DIVS if g%d==0 and K*ed[d]>0)
        neg=sum(K*ed[d]/sp.Integer(d)**17 for d in DIVS if g%d==0 and K*ed[d]<0)
        cg=pos+Z17*neg
        print(f"  gamma={g:<2d}: c_E = {float(cg):+.7e}  {'POS' if cg>0 else 'NEG'}")
        if mn is None or cg<mn[1]: mn=(g,cg)
    n0=(2*Cw/float(mn[1]))**0.125
    print(f"  MIN gamma={mn[0]}: {float(mn[1]):.10e};  n0=(2*{Cw}/c_E)^(1/8)={n0:.4f} -> ceil {math.ceil(n0)}\n")
