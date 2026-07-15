#!/usr/bin/env python
"""Scoped, NON-DESTRUCTIVE uL2-only pipeline. Writes ONLY *_uL2* files +
results/ul2_orthology.tsv. Does not touch other proteins or orthology_report.tsv."""
import os, sys, glob, re, time, subprocess, tempfile, gzip, traceback
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np, pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as biopdb

ROOT=os.path.expanduser("~/moonlight_project")
CH=f"{ROOT}/chains"; EMB=f"{ROOT}/embeddings"; RES=f"{ROOT}/results"; STR=f"{ROOT}/structures"
sys.path.insert(0, f"{ROOT}/scripts")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
t0=time.time(); log("==== ul2_run START ====")

# ---- Step A: extract uL2 chains from the 10 original structures ----
S={"E.coli":"4U26","Thermus":"1VY5","B.subtilis":"8QCQ","M.tb":"7SFR",
   "Human":"8GLP","Yeast":"4U4R","Rabbit":"7O7Y","Drosophila":"6XU8","Mouse":"9H4N","Rat":"7QGG"}
pat=re.compile(r"(protein\s+L2\b(?!\d)|protein\s+L8\b(?!a)|\buL2\b(?!\d)|RPL8\b)", re.I)
log("Step A: extract uL2 chains")
for org,pdb in S.items():
    outp=f"{CH}/{org}_uL2.pdb"
    if os.path.exists(outp): log(f"  {org} exists, skip"); continue
    with gzip.open(f"{STR}/{pdb}.cif.gz","rt") as fh: cif=pdbx.CIFFile.read(fh)
    b=cif.block; ent=b["entity"]
    ent_desc={i:d for i,d in zip(ent["id"].as_array(str), ent["pdbx_description"].as_array(str))}
    ep=b["entity_poly"]; ent_chains={}
    for eid,strands in zip(ep["entity_id"].as_array(str), ep["pdbx_strand_id"].as_array(str)):
        ent_chains[eid]=strands.split(",")
    atoms=pdbx.get_structure(cif, model=1); atoms=atoms[struc.filter_amino_acids(atoms)]
    hit=next((eid for eid,d in ent_desc.items() if pat.search(d)), None)
    if hit is None or hit not in ent_chains: log(f"  {org}: NO uL2 MATCH"); continue
    chain=ent_chains[hit][0]; sub=atoms[atoms.chain_id==chain]
    if sub.array_length()==0: log(f"  {org}: empty chain {chain}"); continue
    sub.chain_id=np.full(sub.array_length(),"A")
    pf=biopdb.PDBFile(); pf.set_structure(sub); pf.write(outp)
    log(f"  {org}: '{ent_desc[hit]}' chain {chain} -> {len(np.unique(sub.res_id))} res")
ul2=sorted(glob.glob(f"{CH}/*_uL2.pdb")); log(f"Step A DONE: {len(ul2)} uL2 chains")

# ---- Step B: 3di + embeddings (uL2 chains only) ----
import torch
from transformers import EsmModel, EsmTokenizer, AutoTokenizer
torch.set_grad_enabled(False)
log("Step B: loading SaProt + ESM-2 offline")
sap_tok=EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
sap_model=EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2", add_pooling_layer=False).eval()
esm_tok=AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model=EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", add_pooling_layer=False).eval()
log("  models loaded")
def sa_of(pdb):
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf: out=tf.name
    subprocess.run(["foldseek","structureto3didescriptor",pdb,out],capture_output=True,text=True,check=True)
    line=open(out).readline().rstrip("\n"); os.remove(out)
    p=line.split("\t"); aa,tdi=p[1],p[2]
    return aa, "".join(a.upper()+d.lower() for a,d in zip(aa,tdi)), len(aa)
for pdb in ul2:
    name=os.path.basename(pdb)[:-4]; aa,sa,nres=sa_of(pdb)
    for tok,model,seq,suf in [(sap_tok,sap_model,sa,"saprot"),(esm_tok,esm_model,aa,"esm2")]:
        npy=f"{EMB}/{name}_{suf}.npy"
        if os.path.exists(npy): continue
        enc=tok(seq,return_tensors="pt")
        pooled=model(**enc).last_hidden_state[0][1:1+nres].mean(dim=0).numpy().astype(np.float32)
        assert pooled.shape[0]==1280, f"{name} dim {pooled.shape}"
        np.save(npy,pooled); log(f"  embed {suf} {name} nres={nres}")
log("Step B DONE")

# ---- Step C: pipeline (uL2 only) ----
import pipeline
pipeline.PROTEINS=["uL2"]
log("Step C: compute_divergence(uL2)")
pipeline.compute_divergence()
def cos_dist(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    return 1.0-float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
order=list(pd.read_csv(f"{RES}/uL2_divergence.csv",index_col=0).index)
for tag in ["saprot","esm2"]:
    vecs={o:np.load(f"{EMB}/{o}_uL2_{tag}.npy") for o in order}
    n=len(order); M=np.zeros((n,n))
    for i,oi in enumerate(order):
        for k,ok in enumerate(order):
            if i!=k: M[i,k]=cos_dist(vecs[oi],vecs[ok])
    pd.DataFrame(M,index=order,columns=order).to_csv(f"{RES}/uL2_{tag}_cosine.csv")
    log(f"  wrote uL2_{tag}_cosine.csv")
try:
    pipeline.run_protein("uL2")
except Exception as e:
    log(f"[WARN] run_protein(uL2): {e!r}"); traceback.print_exc()
log("Step C DONE")

# ---- Step D: orthology (SEPARATE file) ----
log("Step D: orthology")
def tm_short(ref,q):
    out=subprocess.run(["TMalign",ref,q],capture_output=True,text=True).stdout
    tms=re.findall(r"TM-score=\s*([\d.]+)",out); lens=re.findall(r"Length of Chain_\d+:\s*(\d+)",out)
    if len(tms)<2 or len(lens)<2: return None
    L1,L2=int(lens[0]),int(lens[1]); return float(tms[0]) if L1<=L2 else float(tms[1])
ref=f"{CH}/E.coli_uL2.pdb"
with open(f"{RES}/ul2_orthology.tsv","w") as fh:
    fh.write("protein\torganism\ttm_by_shorter\n")
    for org in ["Human","Yeast","Rabbit","Rat","Mouse","Drosophila"]:
        q=f"{CH}/{org}_uL2.pdb"
        if not os.path.exists(q): continue
        tm=tm_short(ref,q); fh.write(f"uL2\t{org}\t{tm}\n"); log(f"  uL2 {org} TM(short)={tm}")
log(f"==== ul2_run DONE in {(time.time()-t0)/60:.1f} min ====")
