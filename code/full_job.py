#!/usr/bin/env python
"""Detached full ribosomal-protein pipeline for the 12 VALID proteins.
Runs entirely on a compute node, no external interaction. Idempotent:
- skips existing embeddings (.npy)
- overwrites divergence/seqdist/cosine/etc CSVs each run (safe to re-run)

Order is chosen so the #1 offline risk (HF model loading) is exercised FIRST,
so a log tail within ~1 min confirms the job started cleanly.

Steps:
  0. Load SaProt + ESM-2 models OFFLINE (early crash if cache unreadable)
  1. gen 3di.json for ALL chains (resilient: skips a bad chain, does not abort)
  2. embed every chain with both models (LONG POLE ~1 min/chain), skip existing
  3. pipeline.compute_divergence() for the 12 proteins
  4. per-protein cosine matrices (+ uL4 kingdom-separation sanity table)
  5. pipeline.run_protein(p) for each of 12 (seqdist, euclid, foldseek, tree,
     PCA/UMAP plots, mantel, cophenetic) -- each sub-step safe-wrapped
  6. results/orthology_report.tsv (euk chain vs E.coli TM-score by shorter)
"""
import os, sys, json, glob, re, time, subprocess, tempfile, traceback
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/moonlight_project")
CH   = os.path.join(ROOT, "chains")
EMB  = os.path.join(ROOT, "embeddings")
RES  = os.path.join(ROOT, "results")
os.makedirs(EMB, exist_ok=True)
os.makedirs(RES, exist_ok=True)

VALID = ["uL4", "uS3", "uS2", "uS5", "uS4", "uL5", "uL3",
         "uL18", "uS11", "uL24", "uL23", "uS10"]           # uL1 DROPPED; uS4 flagged divergent
EUK_ORGS = ["Human", "Yeast", "Rabbit", "Rat", "Mouse", "Drosophila"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

t0 = time.time()
log("==================== full_job START ====================")
log(f"VALID proteins ({len(VALID)}): {', '.join(VALID)}")

# ==================== Step 0: load models OFFLINE ====================
import torch
from transformers import EsmModel, EsmTokenizer, AutoTokenizer
torch.set_grad_enabled(False)

log("Step 0: loading SaProt + ESM-2 models OFFLINE (HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s)"
    % (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")))
sap_tok   = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
sap_model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2", add_pooling_layer=False).eval()
log("  SaProt loaded OK")
esm_tok   = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", add_pooling_layer=False).eval()
log("  ESM-2 loaded OK")
log("Step 0 DONE: both models loaded offline")

# ==================== Step 1: 3di.json (resilient) ====================
OUTJSON = os.path.join(EMB, "3di.json")
log("Step 1: generating 3di.json for all chains")
pdbs = sorted(glob.glob(os.path.join(CH, "*.pdb")))
data = {}
mismatches = []
failed = []
for pdb in pdbs:
    name = os.path.basename(pdb)[:-4]
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf:
        out = tf.name
    r = subprocess.run(["foldseek", "structureto3didescriptor", pdb, out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  FOLDSEEK FAIL {name}: {r.stderr[-200:]}  (skipping this chain)")
        failed.append(name)
        try: os.remove(out)
        except OSError: pass
        continue
    with open(out) as fh:
        line = fh.readline().rstrip("\n")
    os.remove(out)
    parts = line.split("\t")
    aa, tdi = parts[1], parts[2]
    if len(aa) != len(tdi):
        mismatches.append((name, len(aa), len(tdi)))
    sa = "".join(a.upper() + d.lower() for a, d in zip(aa, tdi))
    data[name] = {"aa": aa, "tdi": tdi.lower(), "sa_seq": sa, "n_res": len(aa)}
with open(OUTJSON, "w") as fh:
    json.dump(data, fh, indent=1)
log(f"Step 1 DONE: wrote {OUTJSON} ({len(data)} chains); "
    f"failed={failed or 'none'}; mismatches={mismatches or 'none'}")

# ==================== Step 2: embeddings (long pole) ====================
chains = sorted(data.keys())

def embed_all(model, tok, seq_key, suffix):
    vecs = {}
    n_new = 0
    for name in chains:
        npy = os.path.join(EMB, f"{name}_{suffix}.npy")
        if os.path.exists(npy):
            vecs[name] = np.load(npy)
            continue
        seq = data[name][seq_key]
        enc = tok(seq, return_tensors="pt")
        out = model(**enc).last_hidden_state[0]
        nres = data[name]["n_res"]
        pooled = out[1:1 + nres].mean(dim=0).numpy().astype(np.float32)
        assert pooled.shape[0] == 1280, f"{name} dim {pooled.shape}"
        np.save(npy, pooled)
        vecs[name] = pooled
        n_new += 1
        log(f"  embed {suffix} {name:26s} nres={nres} tok_len={enc['input_ids'].shape[1]}")
    log(f"  {suffix}: {n_new} new, {len(vecs)-n_new} cached, {len(vecs)} total")
    return vecs

log("Step 2: SaProt embeddings")
sap_vecs = embed_all(sap_model, sap_tok, "sa_seq", "saprot")
log("Step 2: ESM-2 embeddings")
esm_vecs = embed_all(esm_model, esm_tok, "aa", "esm2")
log("Step 2 DONE: all embeddings present")

# ==================== Step 3: divergence CSVs ====================
import pipeline
pipeline.PROTEINS = list(VALID)

# safe-wrap the per-protein pipeline sub-steps so one failure never aborts a
# protein's remaining steps or the whole job
_SAFE_FNS = ["compute_sequence_divergence", "compute_euclidean", "plot_embedding",
             "compute_foldseek", "draw_tree", "run_mantel", "clustering_quality"]
def _make_safe(orig, fn):
    def wrapped(*a, **k):
        try:
            return orig(*a, **k)
        except Exception as e:
            log(f"[WARN] pipeline.{fn}{a} failed: {e!r}")
            traceback.print_exc()
    return wrapped
for _fn in _SAFE_FNS:
    setattr(pipeline, _fn, _make_safe(getattr(pipeline, _fn), _fn))

log("Step 3: compute_divergence for 12 proteins")
try:
    pipeline.compute_divergence()
except Exception as e:
    log(f"[WARN] compute_divergence raised: {e!r}")
    traceback.print_exc()
log("Step 3 DONE")

# ==================== Step 4: cosine matrices ====================
def cos_dist(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return 1.0 - sim

log("Step 4: cosine matrices")
matrices = {}
for prot in VALID:
    dcsv = os.path.join(RES, f"{prot}_divergence.csv")
    if not os.path.exists(dcsv):
        log(f"  [WARN] {prot}: no divergence.csv, skipping cosine")
        continue
    order = list(pd.read_csv(dcsv, index_col=0).index)
    for tag, vecs in [("saprot", sap_vecs), ("esm2", esm_vecs)]:
        try:
            n = len(order)
            M = np.zeros((n, n))
            for i, oi in enumerate(order):
                for k, ok in enumerate(order):
                    if i != k:
                        M[i, k] = cos_dist(vecs[f"{oi}_{prot}"], vecs[f"{ok}_{prot}"])
            df = pd.DataFrame(M, index=order, columns=order)
            df.to_csv(os.path.join(RES, f"{prot}_{tag}_cosine.csv"))
            matrices[(prot, tag)] = df
            log(f"  wrote {prot}_{tag}_cosine.csv (n={n} max={M.max():.4f})")
        except Exception as e:
            log(f"  [WARN] {prot}_{tag} cosine failed: {e!r}")

# uL4 kingdom-separation sanity table
BACT = ["E.coli", "B.subtilis", "M.tb", "Thermus"]
EUK = ["Human", "Yeast", "Rabbit", "Rat", "Mouse", "Drosophila"]
def group_means(df):
    def mp(A, B, within):
        vals = []
        for a in A:
            for b in B:
                if a in df.index and b in df.columns:
                    if within and a >= b:
                        continue
                    vals.append(df.loc[a, b])
        return float(np.mean(vals)) if vals else float("nan")
    wb = mp(BACT, BACT, True); we = mp(EUK, EUK, True); bt = mp(BACT, EUK, False)
    within = np.nanmean([wb, we])
    return wb, we, bt, bt / within
if ("uL4", "saprot") in matrices:
    log("Step 4 uL4 kingdom separation (cosine): model within_bact within_euk between ratio")
    for tag in ["saprot", "esm2"]:
        if ("uL4", tag) in matrices:
            wb, we, bt, r = group_means(matrices[("uL4", tag)])
            log(f"   {tag:8s} {wb:.4f} {we:.4f} {bt:.4f} ratio={r:.3f}")
log("Step 4 DONE")

# ==================== Step 5: per-protein pipeline ====================
log("Step 5: run_protein for each of 12 proteins")
for p in VALID:
    log(f"  --- run_protein({p}) ---")
    try:
        pipeline.run_protein(p)
    except Exception as e:
        log(f"  [WARN] run_protein({p}) aborted: {e!r}")
        traceback.print_exc()
log("Step 5 DONE")

# ==================== Step 6: orthology report ====================
log("Step 6: orthology_report.tsv")
def tmalign_short(ref, query):
    out = subprocess.run(["TMalign", ref, query], capture_output=True, text=True).stdout
    tms = re.findall(r"TM-score=\s*([\d.]+)", out)
    lens = re.findall(r"Length of Chain_\d+:\s*(\d+)", out)
    if len(tms) < 2 or len(lens) < 2:
        return None
    L1, L2 = int(lens[0]), int(lens[1])         # Chain_1=ref, Chain_2=query
    tm1, tm2 = float(tms[0]), float(tms[1])
    return tm1 if L1 <= L2 else tm2             # normalized by SHORTER chain
rows = []
for p in VALID:
    ref = os.path.join(CH, f"E.coli_{p}.pdb")
    if not os.path.exists(ref):
        log(f"  [WARN] {p}: no E.coli reference chain, skipping")
        continue
    for org in EUK_ORGS:
        q = os.path.join(CH, f"{org}_{p}.pdb")
        if not os.path.exists(q):
            continue
        tm = tmalign_short(ref, q)
        rows.append((p, org, f"{tm:.4f}" if tm is not None else "NA", "yes"))
        log(f"  {p:5s} {org:11s} TM(short)={tm}")
orp = os.path.join(RES, "orthology_report.tsv")
with open(orp, "w") as fh:
    fh.write("protein\torganism\ttm_score_by_shorter\tkept\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")
log(f"Step 6 DONE: wrote {orp} ({len(rows)} rows)")

log(f"==================== full_job DONE in {(time.time()-t0)/60:.1f} min ====================")
