#!/usr/bin/env python
"""
Precise per-kingdom chain extraction for the 13 universal ribosomal proteins.

Matching strategy (per protein, per structure):
  1. Try the UNIVERSAL name (uLx / uSx) as a standalone token  -> safest, unambiguous.
  2. Else try the kingdom-appropriate OLD name(s) (bacterial vs eukaryotic synonyms).
Token boundaries use (?<![a-z0-9]) / (?![a-z0-9]) so that e.g.
  "S2" never matches "S20"/"S23"/"SA"; "L23" never matches "L23a"; "uL1" never matches "uL18".
Skips mitochondrial entities. Records EVERY entity that matches (to catch ambiguity).

Writes chains/{org}_{protein}.pdb (chain relabeled 'A', amino acids only) and a manifest.
Leaves uL4 and uS3 chains UNTOUCHED (already correct + embedded).
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

BACTERIA = {"E.coli", "Thermus", "B.subtilis", "M.tb"}
# organism -> pdb  (from data/structure_list.csv, S.aureus has no structure)
ORG_PDB = {
    "E.coli": "4U26", "Thermus": "1VY5", "B.subtilis": "8QCQ", "M.tb": "7SFR",
    "Human": "8GLP", "Yeast": "4U4R", "Rabbit": "7O7Y", "Drosophila": "6XU8",
    "Mouse": "9H4N", "Rat": "7QGG",
}

# protein -> (universal, [bacterial old names], [eukaryotic old-name synonyms])
# eukaryotic synonyms ordered by preference; yeast/drosophila-specific names included.
TARGETS = {
    "uL4":  ("uL4",  ["L4"],  ["L4"]),
    "uS2":  ("uS2",  ["S2"],  ["SA", "S0"]),        # euk RPSA; yeast RPS0
    "uS5":  ("uS5",  ["S5"],  ["S2"]),              # euk RPS2
    "uS4":  ("uS4",  ["S4"],  ["S9"]),              # euk RPS9
    "uL5":  ("uL5",  ["L5"],  ["L11"]),             # euk RPL11
    "uL1":  ("uL1",  ["L1"],  ["L10a", "L10A"]),    # euk RPL10A
    "uL3":  ("uL3",  ["L3"],  ["L3"]),
    "uL18": ("uL18", ["L18"], ["L5"]),              # euk RPL5
    "uS3":  ("uS3",  ["S3"],  ["S3"]),
    "uS11": ("uS11", ["S11"], ["S14", "S14a"]),     # euk RPS14; drosophila S14a
    "uL24": ("uL24", ["L24"], ["L26"]),             # euk RPL26
    "uL23": ("uL23", ["L23"], ["L23a", "L25"]),     # euk RPL23A (mammal); yeast RPL25
    "uS10": ("uS10", ["S10"], ["S20"]),             # euk RPS20  <<< THE FIX (was grabbing eS10)
}

# Do NOT re-extract these (already correct + embedded); leave chains as-is.
SKIP_PROTEINS = {"uL4", "uS3"}

def tok(name):
    """standalone-token regex: not preceded/followed by a letter or digit."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])")

def is_ribosomal(desc):
    d = desc.lower()
    if "ribosom" in d:
        return True
    # bare universal/euk tokens like "uS14", "eL8", "eS10"
    return bool(re.search(r"(?<![a-z])[eu][ls]\d", d))

def find_entities(ent_desc, names):
    """Return list of (eid, desc) whose description matches any token in `names`."""
    hits = []
    for eid, desc in ent_desc.items():
        d = desc.lower()
        if "mitochondrial" in d:
            continue
        if not is_ribosomal(desc):
            continue
        for nm in names:
            if tok(nm).search(d):
                hits.append((eid, desc, nm))
                break
    return hits

manifest = []
for org, pdbid in ORG_PDB.items():
    kingdom = "bact" if org in BACTERIA else "euk"
    path = f"{STRUCT_DIR}/{pdbid}.cif.gz"
    if not os.path.exists(path):
        print(f"{org}: NO FILE"); continue
    with gzip.open(path, "rt") as fh:
        cif = pdbx.CIFFile.read(fh)
    block = cif.block
    ent = block["entity"]
    ent_desc = {i: d for i, d in zip(ent["id"].as_array(str), ent["pdbx_description"].as_array(str))}
    ep = block["entity_poly"]
    ent_chains = {eid: strands.split(",")
                  for eid, strands in zip(ep["entity_id"].as_array(str), ep["pdbx_strand_id"].as_array(str))}
    atoms = pdbx.get_structure(cif, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]

    for prot, (uni, bact_names, euk_names) in TARGETS.items():
        if prot in SKIP_PROTEINS:
            continue
        old_names = bact_names if kingdom == "bact" else euk_names
        # universal first, then kingdom old names
        names_ordered = [uni] + old_names
        # match universal separately so we can prefer it
        uni_hits = find_entities(ent_desc, [uni])
        old_hits = find_entities(ent_desc, old_names)
        chosen = None; via = None; ambig = ""
        if uni_hits:
            chosen = uni_hits[0]; via = "universal"
            if len(uni_hits) > 1: ambig = f"AMBIG-uni:{[h[0] for h in uni_hits]}"
        elif old_hits:
            chosen = old_hits[0]; via = "oldname"
            if len({h[0] for h in old_hits}) > 1: ambig = f"AMBIG-old:{[(h[0],h[2]) for h in old_hits]}"
        if chosen is None:
            manifest.append(dict(org=org, pdb=pdbid, kingdom=kingdom, protein=prot,
                                 status="MISSING", via="", matched_desc="", chain="", n_res=0, note=""))
            continue
        eid, desc, matched_name = chosen
        if eid not in ent_chains:
            manifest.append(dict(org=org, pdb=pdbid, kingdom=kingdom, protein=prot,
                                 status="NO_POLY", via=via, matched_desc=desc, chain="", n_res=0, note=ambig))
            continue
        chain = ent_chains[eid][0]
        sub = atoms[atoms.chain_id == chain]
        if sub.array_length() == 0:
            manifest.append(dict(org=org, pdb=pdbid, kingdom=kingdom, protein=prot,
                                 status="EMPTY_CHAIN", via=via, matched_desc=desc, chain=chain, n_res=0, note=ambig))
            continue
        sub = sub.copy()
        sub.chain_id = np.full(sub.array_length(), "A")
        outp = f"{OUT_DIR}/{org}_{prot}.pdb"
        pf = pdb.PDBFile(); pf.set_structure(sub); pf.write(outp)
        nres = len(np.unique(sub.res_id))
        manifest.append(dict(org=org, pdb=pdbid, kingdom=kingdom, protein=prot,
                             status="OK", via=via, matched_desc=desc, chain=chain, n_res=nres,
                             note=(ambig + f" matched='{matched_name}'")))

# write manifest
mf = f"{BASE}/data/extract_manifest.csv"
cols = ["org","pdb","kingdom","protein","status","via","chain","n_res","matched_desc","note"]
with open(mf, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in manifest: w.writerow({k: r.get(k,"") for k in cols})

# pretty print grouped by protein
print(f"{'protein':6} {'org':11} {'status':11} {'via':9} {'chain':6} {'nres':>4}  desc")
for prot in TARGETS:
    if prot in SKIP_PROTEINS: continue
    for r in [x for x in manifest if x["protein"]==prot]:
        print(f"{r['protein']:6} {r['org']:11} {r['status']:11} {r['via']:9} {str(r['chain']):6} {r['n_res']:>4}  {r['matched_desc']}  {r['note']}")
    print()
print("manifest ->", mf)
