# AI chatbot health-advice adoption intention

This repository provides scored analysis data and code for a longitudinal study
of AI chatbot health-advice adoption intention. Machine learning compares the
predictive importance of the study factors, and cross-lagged panel network
analysis estimates their prospective relationships across two survey waves.

## Data and files

The dataset contains 2,149 participants and 77 analysis fields comprising scored
measures and covariates. Each row combines a participant's records across the
two waves. Release identifiers link the analysis data to the original training,
validation, and test assignments.

- [Analysis data](data/analysis_scores.csv) contain the participant-level model inputs.
- [Data dictionary](data/data_dictionary.csv) documents field labels, waves,
  domains, model roles, and missingness.
- [Data partitions](data/ml_partitions.csv) preserve split membership and row
  order for model fitting and evaluation.
- [Network nodes](data/network_nodes.csv) pair the wave-specific columns used
  in the longitudinal network.
- [Analysis configuration](config/analysis.json) records model parameters,
  predictor order, and preprocessing settings.
- [Reference results](benchmarks/) provide model-performance metrics, SHAP
  importance, network coefficients, centrality estimates, and stability summaries.

## Run the analyses

Install the Python dependencies in [requirements.txt](requirements.txt) and the
R packages listed in [r-packages.txt](r-packages.txt). Run both scripts from the
repository directory.

```sh
python scripts/reproduce_ml.py
Rscript scripts/reproduce_clpn.R
```

### Machine learning

The Python script refits seven models using the retained parameters and computes
their training, validation, and test performance. Training and validation
metrics use models fitted on the training data. Test metrics use models refitted
on the combined training and validation data. The script also calculates grouped
interventional TreeSHAP values and feature importance for CatBoost.

Results are saved in `results/ml/`, including model-performance comparisons,
test predictions, participant-level SHAP values, feature importance, and
comparisons with the reference results.

### Longitudinal network

The R script estimates a 30-node directed network using node-wise 10-fold
cross-validated lasso. It applies the recorded preprocessing and random-number
settings, calculates cross-lagged coefficients, and derives out-, in-, and
one-step bridge expected influence.

Results are saved in `results/clpn/`, including standardized model inputs,
coefficient matrices, centrality estimates, and comparisons with the reference
results.

### Bootstrap analyses

The following command runs 1,000 nonparametric bootstrap replicates and 1,000
case-dropping replicates, saving edge summaries and correlation-stability
coefficients in a separate output directory.

```sh
Rscript scripts/reproduce_clpn.R --bootstraps 1000 --output results/bootstrap_full
```

The replicate count is controlled by `--bootstraps`, and the output location is
controlled by `--output`.

## Verified results

Runs from a separate working directory reproduced the seven models' performance
metrics, SHAP importance values, and main network coefficients within
floating-point precision. Out-, in-, and one-step bridge expected influence
also matched the reference results. Detailed checks are recorded in
[VALIDATION.md](VALIDATION.md), with numerical comparisons in
[validation_summary.json](validation_summary.json).
