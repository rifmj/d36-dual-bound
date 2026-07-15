#!/usr/bin/env python3
r"""verify_all.py — single top-level entry point for the d=36 dual certificate.

One command that runs the whole rigorous chain in order and exits NON-ZERO on any
discrepancy (referee R6, point 4). It does not merge the logic into one file; it is a
single launch point that sequences the existing scripts:

  1. checksums     — SHA-256 of every pinned data file matches SHA256SUMS.
  2. exact         — verify_certificate.py: exact-rational gates G1-G7  -> VERIFIED.
  3. interval C_w  — d36_cs_certificate.py: newform decomposition (exact reconstruction
                     dev.=0) + the outward-rounded interval certification of the two C_w
                     upper bounds that gate G7 consumes.
  4. Arb re-check  — d36_arb_check.py: independent Arb (python-flint) re-solve of the
                     33x33 system + |lambda|-aggregation from the exported dyadic matrix
                     enclosures. OPTIONAL: skipped (with a warning) if python-flint is
                     absent; the primary certificate (steps 2-3) does not depend on it.

Trust boundary: step 2 (the driver) takes the two C_w upper bounds as SEPARATELY CERTIFIED
inputs; step 3 is what certifies them, and step 4 independently re-certifies the solve.
This runner is what makes "run one command" literally true.

Layout-robust: works both in the ancillary bundle (ancillary/code + ancillary/SHA256SUMS)
and the flat repo layout (code/ + SHA256SUMS at the parent). Run from anywhere:
    python3 verify_all.py
Exit code 0 iff every non-skipped step passes. Pure-mathematics research; standard jargon.
"""
import hashlib, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or "python3"

def find_sha_file():
    for rel in ("../SHA256SUMS", "../../SHA256SUMS", "SHA256SUMS"):
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return None

def step_checksums():
    shafile = find_sha_file()
    if not shafile:
        print("  [WARN] SHA256SUMS not found — skipping checksum step")
        return "SKIP"
    base = os.path.dirname(shafile)
    bad, n = [], 0
    for line in open(shafile):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        want, rel = parts[0].lower(), parts[1].lstrip("*").strip()
        fp = os.path.join(base, rel)
        if not os.path.exists(fp):
            bad.append(f"{rel}: MISSING"); continue
        h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        n += 1
        if h != want:
            bad.append(f"{rel}: HASH MISMATCH")
    if bad:
        for b in bad:
            print(f"    {b}")
        return "FAIL"
    print(f"    {n} data files match SHA256SUMS")
    return "PASS"

def run_step(script, need_ok, forbid=("Traceback", "NOT VERIFIED"), optional_import=None):
    """Run `script`; PASS iff rc==0 and every marker in need_ok present and none in forbid.
    If optional_import is set and the script dies on a missing module, return SKIP."""
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print(f"    {script}: NOT FOUND"); return "FAIL"
    t0 = time.time()
    r = subprocess.run([PY, path], cwd=HERE, capture_output=True, text=True)
    out = r.stdout + r.stderr
    dt = time.time() - t0
    if optional_import and (f"No module named" in out and optional_import in out):
        print(f"    {script}: SKIPPED — optional dependency '{optional_import}' not installed [{dt:.0f}s]")
        return "SKIP"
    ok = (r.returncode == 0
          and all(m in out for m in need_ok)
          and not any(f in out for f in forbid))
    tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or [""]
    print(f"    {script}: {'PASS' if ok else 'FAIL'} (rc={r.returncode}) [{dt:.0f}s]  | {tail[0][:80]}")
    if not ok:
        sys.stdout.write("\n".join("      " + ln for ln in out.splitlines()[-25:]) + "\n")
    return "PASS" if ok else "FAIL"

def main():
    print("=" * 78)
    print("verify_all.py — full d=36 dual-certificate verification chain")
    print("=" * 78)
    results = {}
    print("[1/4] checksums")
    results["checksums"] = step_checksums()
    print("[2/4] exact-rational gates (verify_certificate.py) ~2-4 min")
    results["exact"] = run_step("verify_certificate.py", need_ok=("VERIFIED",))
    print("[3/4] interval C_w certification (d36_cs_certificate.py)")
    results["interval"] = run_step("d36_cs_certificate.py", need_ok=("FULLY PROVED",))
    print("[4/4] independent Arb re-verification (d36_arb_check.py) — optional")
    results["arb"] = run_step("d36_arb_check.py",
                              need_ok=("ARB INDEPENDENT RE-VERIFICATION: PASS",),
                              optional_import="flint")
    print("-" * 78)
    for k, v in results.items():
        print(f"  {k:12s} {v}")
    hard_fail = any(v == "FAIL" for v in results.values())
    if hard_fail:
        print("VERIFICATION FAILED")
        return 1
    if results["arb"] == "SKIP":
        print("VERIFIED (primary certificate); the optional Arb cross-check was skipped "
              "(install python-flint to run it).")
    else:
        print("VERIFIED — all steps passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
