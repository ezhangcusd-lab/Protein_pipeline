#!/usr/bin/env python
"""
tmalign_io.py - plumbing for M.9. Handles running TMalign and parsing its
output. The insertion-detection logic lives elsewhere (write your own).

    from tmalign_io import align, chain_path, BACTERIA, EUKARYOTES, PROTEINS

    sb, se = align(chain_path("E.coli", "uL4"), chain_path("Human", "uL4"))
    # sb, se are equal-length strings, '-' marks a gap.
    # A eukaryote-specific insertion is a position where sb[i]=='-' and se[i]!='-'.
"""
import subprocess, os, re

CHAINS     = os.path.expanduser("~/moonlight_project/chains")
BACTERIA   = ["E.coli", "B.subtilis", "M.tb", "Thermus"]
EUKARYOTES = ["Human", "Mouse", "Rabbit", "Rat", "Drosophila", "Yeast"]
PROTEINS   = ["uL4", "uS3", "uL2"]


def chain_path(organism, protein):
    """Path to a chain PDB file. Raises if missing, so you fail loudly."""
    p = os.path.join(CHAINS, f"{organism}_{protein}.pdb")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def _run(pdb1, pdb2):
    r = subprocess.run(["TMalign", pdb1, pdb2], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"TMalign failed:\n{r.stderr[-300:]}")
    return r.stdout


def align(bact_pdb, euk_pdb):
    """Run TMalign(bacterium, eukaryote) -> (seq_bact, seq_euk).

    Both strings are the same length, gap-padded with '-'.
    Order matters: pass the BACTERIUM first, so its gaps mark the
    eukaryote's insertions.
    """
    lines = _run(bact_pdb, euk_pdb).splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('(":" denotes'):
            seq_b, seq_e = lines[i + 1], lines[i + 3]   # i+2 is the match line
            if len(seq_b) != len(seq_e):
                raise ValueError(f"alignment rows differ: {len(seq_b)} vs {len(seq_e)}")
            return seq_b, seq_e
    raise ValueError("no alignment block found in TMalign output")


def scores(bact_pdb, euk_pdb):
    """-> dict with rmsd, seq_id, aligned_len, tm_norm_1, tm_norm_2."""
    out = _run(bact_pdb, euk_pdb)
    d = {}
    m = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([\d.]+), Seq_ID=n_identical/n_aligned=\s*([\d.]+)", out)
    if m:
        d["aligned_len"], d["rmsd"], d["seq_id"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    tms = re.findall(r"TM-score=\s*([\d.]+)", out)
    if len(tms) >= 2:
        d["tm_norm_1"], d["tm_norm_2"] = float(tms[0]), float(tms[1])
    return d


if __name__ == "__main__":
    sb, se = align(chain_path("E.coli", "uL4"), chain_path("Human", "uL4"))
    print(f"alignment length: {len(sb)}")
    print(f"insertion positions (sb gap, se residue): "
          f"{sum(1 for b, e in zip(sb, se) if b == '-' and e != '-')}")
    print("scores:", scores(chain_path("E.coli", "uL4"), chain_path("Human", "uL4")))
