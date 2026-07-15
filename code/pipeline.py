import glob, os, re, subprocess, shutil
from itertools import combinations
import numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH  = os.path.expanduser("~/moonlight_project/chains")
RES = os.path.expanduser("~/moonlight_project/results")
EMB = os.path.expanduser("~/moonlight_project/embeddings")
PROTEINS = ["uL4", "uS3", "uS10"]
EUK = {"Human", "Yeast", "Rabbit", "Drosophila", "Mouse", "Rat"}
MATS = ["divergence","seqdist","saprot_cosine","esm2_cosine","saprot_euclid","esm2_euclid","foldseek"]

# ==================== Zone 1: helpers ====================
def align_lines(f1, f2):
    out = subprocess.run(["TMalign", f1, f2], capture_output=True, text=True).stdout.splitlines()
    i = next(k for k, l in enumerate(out) if l.startswith('(":"'))
    return out[i+1], out[i+3]

def find_blocks(bact, euk, min_len=3):
    output=[]; missing=0; length=0
    for i in range(len(euk)):
        if bact[i] == "-":
            length += 1; missing = 1; pos = i
        if bact[i] != "-" and missing == 1:
            if length >= min_len: output.append((pos, length))
            length = 0; missing = 0
    if missing == 1 and length >= min_len: output.append((pos, length))
    return output

def mantel(D1, D2, n_perm=9999):
    D2 = D2.reindex(index=D1.index, columns=D1.columns)
    A = D1.values.astype(float); B = D2.values.astype(float)
    iu = np.triu_indices(A.shape[0], k=1)
    rx = rankdata(A[iu]); obs = np.corrcoef(rx, rankdata(B[iu]))[0, 1]
    n = A.shape[0]; count = 0
    for _ in range(n_perm):
        p = np.random.permutation(n)
        if abs(np.corrcoef(rx, rankdata(B[p][:, p][iu]))[0, 1]) >= abs(obs): count += 1
    return obs, (count + 1) / (n_perm + 1)

def load_vectors(protein, model):
    return {os.path.basename(f).split("_")[0]: np.load(f)
            for f in glob.glob(f"{EMB}/*_{protein}_{model}.npy")}

def reduce_2d(protein, model, method="pca"):
    vecs = load_vectors(protein, model); orgs = sorted(vecs)
    X = np.vstack([vecs[o] for o in orgs])
    if method == "pca":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(X)
    else:
        import umap
        coords = umap.UMAP(n_components=2, n_neighbors=5, random_state=0).fit_transform(X)
    return orgs, coords

def foldseek_scores(protein):
    chains = sorted(glob.glob(f"{CH}/*_{protein}.pdb"))
    qdir = f"/tmp/fs_{protein}"
    if os.path.exists(qdir): shutil.rmtree(qdir)
    os.makedirs(qdir)
    for c in chains: shutil.copy(c, qdir)
    out = f"/tmp/fs_{protein}.tsv"
    subprocess.run(["foldseek","easy-search",qdir,qdir,out,f"/tmp/fstmp_{protein}",
        "--format-output","query,target,alntmscore","--alignment-type","1",
        "-e","10","--exhaustive-search","1"], capture_output=True, text=True)
    scores = {}
    for line in open(out):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3: scores[(p[0].split("_")[0], p[1].split("_")[0])] = float(p[2])
    return scores

# ==================== Zone 2: stages ====================
def compute_divergence():
    for protein in PROTEINS:
        files = sorted(glob.glob(f"{CH}/*_{protein}.pdb"))
        orgs  = [os.path.basename(f).split("_")[0] for f in files]
        div = pd.DataFrame(index=orgs, columns=orgs, dtype=float)
        for a, b in combinations(files, 2):
            oa = os.path.basename(a).split("_")[0]; ob = os.path.basename(b).split("_")[0]
            out = subprocess.run(["TMalign", a, b], capture_output=True, text=True).stdout
            tms = re.findall(r"TM-score=\s*([\d.]+)", out)
            div.loc[oa,ob] = div.loc[ob,oa] = 1.0 - (float(tms[0]) + float(tms[1]))/2.0
        for o in orgs: div.loc[o, o] = 0.0
        div.to_csv(f"{RES}/{protein}_divergence.csv"); print(protein, "done")

def compute_sequence_divergence(protein):
    import biotite.structure.io.pdb as biopdb
    import biotite.structure as struc
    from biotite.sequence import ProteinSequence
    def chain_seq(path):
        a = biopdb.PDBFile.read(path).get_structure(model=1)
        a = a[struc.filter_amino_acids(a)]
        out = []
        for rid in sorted(set(a.res_id)):
            nm = a[a.res_id == rid].res_name[0]
            try: out.append(ProteinSequence.convert_letter_3to1(nm))
            except Exception: out.append("X")
        return "".join(out)
    files = sorted(glob.glob(f"{CH}/*_{protein}.pdb"))
    orgs = [os.path.basename(f).split("_")[0] for f in files]
    fasta = f"/tmp/{protein}.fasta"
    with open(fasta, "w") as fh:
        for o, f in zip(orgs, files): fh.write(f">{o}\n{chain_seq(f)}\n")
    aln_txt = subprocess.run(["mafft","--auto","--quiet",fasta], capture_output=True, text=True).stdout
    aln = {}; name = None
    for line in aln_txt.splitlines():
        if line.startswith(">"): name = line[1:].strip(); aln[name] = ""
        elif name: aln[name] += line.strip()
    def pid(s1, s2):
        same = comp = 0
        for c1, c2 in zip(s1, s2):
            if c1 != "-" and c2 != "-":
                comp += 1; same += (c1 == c2)
        return same / comp if comp else 0.0
    dist = pd.DataFrame(index=orgs, columns=orgs, dtype=float)
    for a, b in combinations(orgs, 2):
        dist.loc[a, b] = dist.loc[b, a] = 1.0 - pid(aln[a], aln[b])
    for o in orgs: dist.loc[o, o] = 0.0
    dist.to_csv(f"{RES}/{protein}_seqdist.csv"); print(f"wrote {protein}_seqdist.csv")

def compute_euclidean(protein="uL4", model="saprot"):
    vecs = load_vectors(protein, model); orgs = sorted(vecs)
    df = pd.DataFrame(index=orgs, columns=orgs, dtype=float)
    for a, b in combinations(orgs, 2):
        df.loc[a, b] = df.loc[b, a] = np.linalg.norm(vecs[a] - vecs[b])
    for o in orgs: df.loc[o, o] = 0.0
    df.to_csv(f"{RES}/{protein}_{model}_euclid.csv"); print(f"wrote {protein}_{model}_euclid.csv")

def compute_foldseek(protein="uL4"):
    scores = foldseek_scores(protein); orgs = sorted({a for a, _ in scores})
    df = pd.DataFrame(index=orgs, columns=orgs, dtype=float)
    for a in orgs:
        for b in orgs:
            df.loc[a, b] = 0.0 if a == b else max(0.0, 1 - (scores[(a, b)] + scores[(b, a)]) / 2)
    df.to_csv(f"{RES}/{protein}_foldseek.csv"); print(f"wrote {protein}_foldseek.csv")

def region_divergence(bact, euk, start, end):
    i_div=i_tot=o_div=o_tot=0; pos=0
    for i in range(len(euk)):
        if euk[i] == "-": continue
        pos += 1
        diverged = (bact[i] == "-") or (bact[i] != euk[i])
        if start <= pos <= end: i_tot += 1; i_div += diverged
        else:                   o_tot += 1; o_div += diverged
    inside  = i_div / i_tot if i_tot else float("nan")
    outside = o_div / o_tot if o_tot else float("nan")
    return inside, outside

def draw_tree(protein="uL4", matrix="divergence"):
    d = pd.read_csv(f"{RES}/{protein}_{matrix}.csv", index_col=0)
    Z = linkage(squareform(d.values), method="average")
    plt.figure(figsize=(9, 5))
    dendrogram(Z, labels=list(d.index), leaf_rotation=45, color_threshold=0.20)
    plt.ylabel(f"{matrix} distance"); plt.title(f"{protein}: hierarchical clustering ({matrix})")
    plt.tight_layout(); out = f"{RES}/{protein}_tree_{matrix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)

def plot_embedding(protein="uL4", model="saprot", method="pca"):
    orgs, coords = reduce_2d(protein, model, method)
    plt.figure(figsize=(7, 6))
    for i, org in enumerate(orgs):
        plt.scatter(coords[i,0], coords[i,1], color=("tab:orange" if org in EUK else "tab:green"), s=80)
        plt.annotate(org, (coords[i,0], coords[i,1]), xytext=(5,5), textcoords="offset points", fontsize=9)
    from matplotlib.lines import Line2D
    plt.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='tab:green', label='Bacteria',markersize=9),
                        Line2D([0],[0],marker='o',color='w',markerfacecolor='tab:orange',label='Eukaryotes',markersize=9)])
    plt.xlabel(f"{method.upper()} 1"); plt.ylabel(f"{method.upper()} 2")
    plt.title(f"{protein} {model} embeddings ({method.upper()})")
    plt.tight_layout(); out = f"{RES}/{protein}_{model}_{method}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)

def run_mantel(protein="uL4"):
    np.random.seed(0)
    mats = [m for m in MATS if os.path.exists(f"{RES}/{protein}_{m}.csv")]
    loaded = {m: pd.read_csv(f"{RES}/{protein}_{m}.csv", index_col=0) for m in mats}
    print(f"\n=== Mantel ({protein}) ===")
    for m1, m2 in combinations(mats, 2):
        r, p = mantel(loaded[m1], loaded[m2]); print(f"  {m1:14} vs {m2:14}  r={r:+.3f}  p={p:.4f}")

def clustering_quality(protein="uL4"):
    print(f"\n=== Cophenetic ({protein}) ===")
    for m in [m for m in MATS if os.path.exists(f"{RES}/{protein}_{m}.csv")]:
        cond = squareform(pd.read_csv(f"{RES}/{protein}_{m}.csv", index_col=0).values)
        c, _ = cophenet(linkage(cond, method="average"), cond); print(f"  {m:14} = {c:.3f}")

# ==================== orchestrate ====================
def run_protein(protein):
    print(f"\n########## {protein} ##########")
    compute_sequence_divergence(protein)
    for model in ["saprot", "esm2"]:
        compute_euclidean(protein, model)
        plot_embedding(protein, model, "pca"); plot_embedding(protein, model, "umap")
    compute_foldseek(protein)
    draw_tree(protein, "divergence")
    run_mantel(protein)
    clustering_quality(protein)
    bpath = f"{CH}/E.coli_{protein}.pdb"; epath = f"{CH}/Human_{protein}.pdb"
    if os.path.exists(bpath) and os.path.exists(epath):
        bact, euk = align_lines(bpath, epath)
        print(f"{protein} euk-only blocks (E.coli vs Human):", find_blocks(bact, euk))

def main():
    # compute_divergence()                        # slow — divergence CSVs already exist
    for protein in ["uS3", "uS10"]:               # uL4 already complete
        run_protein(protein)

if __name__ == "__main__":
    main()
