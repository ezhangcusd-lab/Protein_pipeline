# Protein Pipeline

Structural, sequence, and embedding-based divergence analysis of **universal ribosomal proteins** (uL4, uS3, uL2, and others) across **4 bacteria and 6 eukaryotes**. Built to study how proteins that share the same ribosomal role diverge between domains — relevant to their extraribosomal ("moonlighting") functions.

## What it does
For each protein, the pipeline measures divergence between every pair of organisms through three independent lenses, then clusters and validates:

1. **Structure** — TMalign (3-D superposition) and Foldseek (3Di structural tokens)
2. **Sequence** — MAFFT multiple alignment → percent identity
3. **Embeddings** — SaProt (structure-aware) and ESM-2 (sequence-only) protein language models, mean-pooled per chain; cosine and Euclidean distance
4. **Clustering & viz** — average-linkage (UPGMA) dendrograms, PCA, UMAP, heatmaps
5. **Validation** — Mantel test (agreement between methods) and cophenetic correlation (tree faithfulness)
6. **Localization** — eukaryote-specific insertions from the structural alignment
7. **Orthology check** — each eukaryotic chain structurally aligned (TMalign) to a bacterial reference to confirm identity (guards against paralog contamination)

## Code (`code/`)
| File | Purpose |
|---|---|
| `pipeline.py` | Core functions: divergence/seqdist/foldseek/euclidean matrices, clustering, PCA/UMAP, Mantel, cophenetic, region analysis |
| `full_job.py` | Detached full run over all proteins (embeddings + all stages), SLURM-friendly |
| `extract_chains.py` | Extract single-protein chains from whole-ribosome mmCIF structures |
| `ul2_run.py` | Scoped single-protein driver (example: uL2) *(added when available)* |
| `extract_targets.py` | Hardened multi-protein extractor with paralog-safe name matching *(added when available)* |

## Results (`results/`)
Per protein: `*_divergence.csv` (structural), `*_seqdist.csv` (sequence), `*_foldseek.csv`, `*_saprot_cosine/euclid.csv`, `*_esm2_cosine/euclid.csv` (embeddings), plus dendrograms, PCA/UMAP plots, readable heatmaps, and `orthology_report.tsv`.

## Data source & environment
Structures from the RCSB Protein Data Bank ([rcsb.org](https://www.rcsb.org)); sequences read from the structure chains. Run on the UCSB Pod HPC cluster under Python 3.11. Key tools: TMalign 20240303, Foldseek 10, MAFFT v7.526, SaProt_650M_AF2, ESM-2 (esm2_t33_650M_UR50D), scipy/scikit-learn/umap-learn, biotite. (Full versions in the paper's Table 2.)
