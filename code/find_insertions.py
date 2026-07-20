#!/usr/bin/env python
"""
M.9 - Localization of divergence.
Runs TMalign(bacterium, eukaryote), parses the structural alignment, and reports
eukaryote-specific insertions = runs of >=N consecutive positions where the
BACTERIAL sequence has a gap and the EUKARYOTE has residues.
Also classifies each run as N-terminal / internal / C-terminal, and reports
sensitivity to the minimum-run threshold.
"""
import subprocess, os, sys, json
from statistics import mean

CH = os.path.expanduser("~/moonlight_project/chains")
BACT = ["E.coli", "B.subtilis", "M.tb", "Thermus"]
EUK  = ["Human", "Mouse", "Rabbit", "Rat", "Drosophila", "Yeast"]
PROTS = ["uL4", "uS3", "uL2"]
MINRUN = 3

def tmalign_alignment(p1, p2):
    r = subprocess.run(["TMalign", p1, p2], capture_output=True, text=True)
    if r.returncode != 0: return None
    lines = r.stdout.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('(":" denotes'):
            return lines[i+1], lines[i+3]      # seq1 (bact), seq2 (euk)
    return None

def insertions(seq_b, seq_e, minrun):
    """runs where bacterial == '-' and eukaryote has a residue"""
    runs, cur, start = [], 0, None
    for i, (b, e) in enumerate(zip(seq_b, seq_e)):
        if b == "-" and e != "-":
            if cur == 0: start = i
            cur += 1
        else:
            if cur >= minrun: runs.append((start, cur))
            cur = 0
    if cur >= minrun: runs.append((start, cur))
    return runs

def classify(start, length, alnlen):
    if start <= 5: return "N-term"
    if start + length >= alnlen - 5: return "C-term"
    return "internal"

results = {}
for prot in PROTS:
    print(f"\n{'='*76}\n{prot}\n{'='*76}")
    print(f"{'eukaryote':12s} {'vs bact':11s} {'total_ins':>9s} {'n_runs':>7s} {'largest':>8s} {'where':>9s}")
    per_euk = {}
    for euk in EUK:
        pe = f"{CH}/{euk}_{prot}.pdb"
        if not os.path.exists(pe): continue
        tot_list, big_list, where_list = [], [], []
        for b in BACT:
            pb = f"{CH}/{b}_{prot}.pdb"
            if not os.path.exists(pb): continue
            aln = tmalign_alignment(pb, pe)
            if not aln: continue
            sb, se = aln
            runs = insertions(sb, se, MINRUN)
            tot = sum(l for _, l in runs)
            big = max((l for _, l in runs), default=0)
            bigrun = max(runs, key=lambda x: x[1], default=(0,0))
            where = classify(bigrun[0], bigrun[1], len(sb)) if runs else "-"
            print(f"{euk:12s} {b:11s} {tot:>9d} {len(runs):>7d} {big:>8d} {where:>9s}")
            tot_list.append(tot); big_list.append(big); where_list.append(where)
        if tot_list:
            per_euk[euk] = {"total_mean": mean(tot_list), "total_min": min(tot_list),
                            "total_max": max(tot_list), "largest_mean": mean(big_list),
                            "where": max(set(where_list), key=where_list.count)}
    results[prot] = per_euk
    print(f"\n  --- {prot} SUMMARY (mean over the 4 bacterial references) ---")
    for e, v in per_euk.items():
        print(f"   {e:12s} total insertions {v['total_mean']:6.1f} aa "
              f"(range {v['total_min']}-{v['total_max']})   largest {v['largest_mean']:5.1f} aa  [{v['where']}]")
    if per_euk:
        allm = [v["total_mean"] for v in per_euk.values()]
        print(f"   >> across eukaryotes: mean {mean(allm):.1f} aa, range {min(allm):.0f}-{max(allm):.0f} aa")

print(f"\n\n{'='*76}\nTHRESHOLD SENSITIVITY (uL4, vs E.coli) - 'why >=3?'\n{'='*76}")
for prot in PROTS:
    pb = f"{CH}/E.coli_{prot}.pdb"
    if not os.path.exists(pb): continue
    print(f"\n{prot}:")
    print(f"   {'euk':12s}" + "".join(f"{'>='+str(m):>9s}" for m in [1,2,3,5,10]))
    for euk in EUK:
        pe = f"{CH}/{euk}_{prot}.pdb"
        if not os.path.exists(pe): continue
        aln = tmalign_alignment(pb, pe)
        if not aln: continue
        sb, se = aln
        row = "".join(f"{sum(l for _,l in insertions(sb,se,m)):>9d}" for m in [1,2,3,5,10])
        print(f"   {euk:12s}{row}")
json.dump(results, open(os.path.expanduser("~/moonlight_project/results/insertions.json"),"w"), indent=1)
print("\nwrote results/insertions.json")
