#!/usr/bin/env python
"""
Stage 1 — chain extraction.
For each ribosome structure, find the chains annotated as uL4 / uS3 / uS10
(by the mmCIF entity description) and write each as its own PDB file.
Output: ~/moonlight_project/chains/{organism}_{protein}.pdb   (10 orgs x 3 = up to 30)
"""
import os, re, gzip, csv
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

BASE = os.path.expanduser("~/moonlight_project")
STRUCT_DIR = f"{BASE}/structures"
OUT_DIR = f"{BASE}/chains"
os.makedirs(OUT_DIR, exist_ok=True)

# how to recognise each protein in the mmCIF entity description
#   universal name (modern) OR "ribosomal protein <oldname>"
TARGETS = {
    "uL4":  ("uL4",  "l4"),
    "uS3":  ("uS3",  "s3"),
    "uS10": ("uS10", "s10"),
}
def matches(desc, universal, old):
    d = desc.lower()
    if universal.lower() in d:
        return True
    # e.g. "50S ribosomal protein L4", "40S ribosomal protein S3"
    return bool(re.search(r"protein " + old + r"\b(?!\d)", d))

rows = list(csv.DictReader(open(f"{BASE}/data/structure_list.csv")))
print(f"{'organism':12} {'pdb':6} {'uL4':>10} {'uS3':>10} {'uS10':>10}")
summary = []
for r in rows:
    org, pdbid = r["organism"], r["pdb"]
    path = f"{STRUCT_DIR}/{pdbid}.cif.gz"
    if not os.path.exists(path):
        print(f"{org:12} {pdbid:6}  (no file)"); continue
    with gzip.open(path, "rt") as fh:
        cif = pdbx.CIFFile.read(fh)
    block = cif.block
    ent = block["entity"]
    ent_desc = {i: d for i, d in zip(ent["id"].as_array(str), ent["pdbx_description"].as_array(str))}
    # entity_id -> auth chain letters
    ep = block["entity_poly"]
    ent_chains = {}
    for eid, strands in zip(ep["entity_id"].as_array(str), ep["pdbx_strand_id"].as_array(str)):
        ent_chains[eid] = strands.split(",")
    atoms = pdbx.get_structure(cif, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    got = {}
    for prot, (uni, old) in TARGETS.items():
        # find entity whose description matches this protein
        hit = next((eid for eid, d in ent_desc.items() if matches(d, uni, old)), None)
        if hit is None or hit not in ent_chains:
            got[prot] = None; continue
        chain = ent_chains[hit][0]  # first copy
        sub = atoms[atoms.chain_id == chain]
        if sub.array_length() == 0:
            got[prot] = None; continue
        sub.chain_id = np.full(sub.array_length(), "A")  # PDB allows only 1-char chain IDs
        outp = f"{OUT_DIR}/{org}_{prot}.pdb"
        pf = pdb.PDBFile(); pf.set_structure(sub); pf.write(outp)
        nres = len(np.unique(sub.res_id))
        got[prot] = (chain, nres)
    def cell(x): return f"{x[0]}:{x[1]}res" if x else "MISSING"
    print(f"{org:12} {pdbid:6} {cell(got['uL4']):>10} {cell(got['uS3']):>10} {cell(got['uS10']):>10}")
    summary.append((org, pdbid, got))

nfound = sum(1 for _,_,g in summary for v in g.values() if v)
print(f"\nExtracted {nfound} chains total -> {OUT_DIR}")
miss = [(o,p,prot) for o,_,g in summary for prot,v in g.items() if not v for p in [o]]
if miss:
    print("MISSING (need fallback):", [(o,prot) for o,_,g in summary for prot,v in g.items() if not v])
