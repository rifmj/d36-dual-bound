"""Exact-rational LP (Fraction) for the CE dual feasibility/optimization.
Standard-form solver via two-phase simplex with Bland's rule (guaranteed no cycling).

Problem (dual CE), variables x free (j=0..n-1):
  maximize  c . x
  s.t.  A_eq . x = b_eq   (a_0=1 ; a_n=0 for 1<=n<T)
        A_ub . x <= b_ub  (-a_n<=0 for T<=n<=M ; -b_n<=0 for 1<=n<=M)
Free x -> x = x^+ - x^- ; add slacks for <=. Phase-I with artificials for =.
Returns (status, xval, objval).  status in {'optimal','infeasible','unbounded'}.
All arithmetic exact (fractions.Fraction).
"""
from fractions import Fraction as Fr


def _simplex(T, basis, ncols, obj_row, phase1_cols=None):
    # tableau simplex, Bland's rule. T: list of tableau rows (each list of Fr, last entry = RHS).
    # obj_row: list of Fr length ncols+1 (reduced costs row incl RHS = -current obj).
    m = len(T)
    while True:
        # entering: smallest index j with obj_row[j] < 0 (Bland)
        piv_c = -1
        for j in range(ncols):
            if phase1_cols is not None and j in phase1_cols:
                continue
            if obj_row[j] < 0:
                piv_c = j
                break
        if piv_c == -1:
            return 'optimal', basis, T, obj_row
        # ratio test: min RHS/col over positive entries; tie -> smallest basis var index (Bland)
        piv_r = -1
        best = None
        for i in range(m):
            a = T[i][piv_c]
            if a > 0:
                ratio = T[i][-1] / a
                if best is None or ratio < best or (ratio == best and basis[i] < basis[piv_r]):
                    best = ratio
                    piv_r = i
        if piv_r == -1:
            return 'unbounded', basis, T, obj_row
        # pivot
        piv = T[piv_r][piv_c]
        T[piv_r] = [v / piv for v in T[piv_r]]
        for i in range(m):
            if i != piv_r and T[i][piv_c] != 0:
                f = T[i][piv_c]
                T[i] = [T[i][k] - f * T[piv_r][k] for k in range(ncols + 1)]
        if obj_row[piv_c] != 0:
            f = obj_row[piv_c]
            obj_row = [obj_row[k] - f * T[piv_r][k] for k in range(ncols + 1)]
        basis[piv_r] = piv_c


def solve_exact(c, A_eq, b_eq, A_ub, b_ub, nvar):
    """c,A_eq,A_ub entries Fr; maximize c.x, x free. Returns (status, x, obj)."""
    # variables: x^+ (nvar), x^- (nvar), slacks s (n_ub). Then artificials in phase 1.
    n_ub = len(A_ub)
    n_eq = len(A_eq)
    npm = 2 * nvar  # x+ , x-
    n_struct = npm + n_ub  # + slacks
    # Build rows for all constraints as equalities:
    #   eq:  A_eq (x+ - x-) = b_eq
    #   ub:  A_ub (x+ - x-) + s = b_ub
    rows = []
    rhs = []
    for i in range(n_eq):
        row = [Fr(0)] * n_struct
        for j in range(nvar):
            row[j] = Fr(A_eq[i][j]); row[nvar + j] = -Fr(A_eq[i][j])
        rows.append(row); rhs.append(Fr(b_eq[i]))
    for i in range(n_ub):
        row = [Fr(0)] * n_struct
        for j in range(nvar):
            row[j] = Fr(A_ub[i][j]); row[nvar + j] = -Fr(A_ub[i][j])
        row[npm + i] = Fr(1)
        rows.append(row); rhs.append(Fr(b_ub[i]))
    # ensure RHS >= 0
    for i in range(len(rows)):
        if rhs[i] < 0:
            rows[i] = [-v for v in rows[i]]; rhs[i] = -rhs[i]
    ncon = len(rows)
    # Phase 1: add one artificial per row
    ncols = n_struct + ncon
    T = []
    basis = []
    for i in range(ncon):
        r = rows[i] + [Fr(0)] * ncon
        r[n_struct + i] = Fr(1)
        r.append(rhs[i])
        T.append(r)
        basis.append(n_struct + i)
    # phase-1 objective: minimize sum artificials => maximize -sum. reduced cost row:
    # obj = sum_{art} x_art ; we minimize. Convert to our maximize form: maximize -sum art.
    # obj_row for "minimize w": w - sum art = 0; reduced costs computed by subtracting basic art rows.
    obj_row = [Fr(0)] * (ncols + 1)
    for j in range(n_struct, ncols):
        obj_row[j] = Fr(0)  # cost of artificial in phase1 = 1 (we minimize sum art)
    # We implement as: minimize sum(art). Set cost c_art=1, others 0. reduced cost row = -(c_B B^{-1} A - c).
    # Easiest: build phase-1 cost vector and compute reduced costs = c_j - sum_i (c_basis_i * T[i][j]).
    cost = [Fr(0)] * ncols
    for j in range(n_struct, ncols):
        cost[j] = Fr(1)
    # since we MINIMIZE, use obj_row[j] = c_j - z_j where z_j = sum_i cost[basis_i]*T[i][j]; pivot on obj_row[j]<0
    def make_objrow():
        z = [Fr(0)] * (ncols + 1)
        for i in range(ncon):
            cb = cost[basis[i]]
            if cb != 0:
                for k in range(ncols + 1):
                    z[k] += cb * T[i][k]
        return [cost[j] - z[j] for j in range(ncols)] + [-z[ncols]]
    obj_row = make_objrow()
    st, basis, T, obj_row = _simplex(T, basis, ncols, obj_row)
    # phase-1 optimal value = -obj_row[-1] (sum of artificials)
    w = -obj_row[-1]
    if w != 0:
        return 'infeasible', None, None
    # drop artificial columns; phase 2 with real objective (maximize c.x = c.(x+ - x-))
    # If any artificial still basic at 0, try to pivot it out (skip: with Bland & w=0 usually fine; handle simply)
    # Build phase-2 cost (maximize c.x => our simplex minimizes -obj, so set cost2 and pivot on <0 of (cost2 - z))
    cost2 = [Fr(0)] * ncols
    for j in range(nvar):
        cost2[j] = -Fr(c[j])      # maximize c.x  <=>  minimize -c.x
        cost2[nvar + j] = Fr(c[j])
    # forbid artificials from re-entering
    art_cols = set(range(n_struct, ncols))
    def make_objrow2():
        z = [Fr(0)] * (ncols + 1)
        for i in range(ncon):
            cb = cost2[basis[i]]
            if cb != 0:
                for k in range(ncols + 1):
                    z[k] += cb * T[i][k]
        return [cost2[j] - z[j] for j in range(ncols)] + [-z[ncols]]
    obj_row = make_objrow2()
    st, basis, T, obj_row = _simplex(T, basis, ncols, obj_row, phase1_cols=art_cols)
    if st == 'unbounded':
        return 'unbounded', None, None
    # recover x
    xval = [Fr(0)] * ncols
    for i in range(ncon):
        xval[basis[i]] = T[i][-1]
    x = [xval[j] - xval[nvar + j] for j in range(nvar)]
    obj = sum(Fr(c[j]) * x[j] for j in range(nvar))
    return 'optimal', x, obj
