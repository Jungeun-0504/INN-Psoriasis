# INN-Psoriasis

Code and data for the **Reactome-constrained interpretable neural network (INN)** analysis in the study *“Pathway-informed integration of genetic causal inference and disease transcriptomics identifies candidate genes associated with psoriasis pathogenesis.”*

The INN uses gene expression data and Reactome-informed connections to distinguish lesional from non-lesional psoriatic skin. SHAP analysis is used to identify influential biological pathways. This repository covers the **INN component** of the study, rather than the full Mendelian randomization, differential-expression, WGCNA, or experimental workflows.

## Repository structure

- `Code/` — data preparation notebook, training/SHAP pipeline script, plotting notebook, and Conda environment file.
- `data/` — integrated expression matrix and sample phenotype labels derived from public datasets.
- `pathway/` — Reactome pathway annotations and network connectivity files.
- `src/` — model training, SHAP calculation, and result-processing scripts.
- `psoriasis/` — model input and analysis output directories, including provided SHAP results.

## Data

The study integrates five public GEO datasets: **GSE67785, GSE74697, GSE83645, GSE107871, and GSE121212**. Large data files are tracked with **Git LFS**.

## Setup and execution

A Linux environment with Conda and CUDA-capable GPUs is expected. From the repository root:

```bash
git lfs pull
conda env create -f Code/INN_environment.yaml
conda activate INN_env
```

Prepare model inputs using `Code/input_data_prep_Psor_PP_PN.ipynb`, then run the training and SHAP workflow:

```bash
bash Code/run_INN.sh
```

**Before running:** Set the project root path in the preprocessing notebook and ensure `Code/run_INN.sh` uses the correct repository path. The training pipeline expects prepared files under `psoriasis/input/` and writes results under `psoriasis/result/`.

## Citation

Kim J., et al. *Pathway-informed integration of genetic causal inference and disease transcriptomics identifies candidate genes associated with psoriasis pathogenesis.* Manuscript.
