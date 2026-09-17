# Analysis data and code

Analysis-only data and code for the study of AI chatbot health-advice adoption
intention. This repository is maintained by [dianshili](https://github.com/dianshili).

## Data boundary

The data begin after questionnaire scoring. They contain only the participant-
level scores and covariates required for the reported machine-learning and
cross-lagged panel network analyses. They do not contain questionnaire item-
response matrices, source participant identifiers, unused measures, or programs
that construct or score measurements. New release identifiers preserve the
within-participant pairing of the two waves.

- `data/analysis_scores.csv` contains 2,149 records, a release identifier, and 77
  analysis fields. Empty cells retain the original model-input missingness.
- `data/ml_partitions.csv` records the original training, validation, and test
  assignment. Row order within each partition is part of the reproducible input.
- `data/data_dictionary.csv` describes field labels, wave, domain, model role,
  and missingness. Construct definitions and response scales are documented in
  the manuscript's supplementary measurement table.
- `data/network_nodes.csv` links the two wave-specific columns for each node.
- `config/analysis.json` contains the retained analysis parameters and field
  order. Do not change these when reproducing the benchmarks.
- `benchmarks/` contains aggregate reference results, not fitted objects.

## Reproduction

Install the Python packages in `requirements.txt` in a dedicated environment.
The tested R packages and versions are recorded in `r-packages.txt`.

From this directory, run:

```sh
python scripts/reproduce_ml.py
Rscript scripts/reproduce_clpn.R
```

The Python script refits seven models at their retained parameters, calculates
training/validation/test metrics, and computes grouped interventional TreeSHAP
for the retained CatBoost model. Training/validation metrics come from a model
fitted on training data; test metrics come from refitting on training plus
validation data. It does not rerun the historical hyperparameter searches.

The R script performs node-wise 10-fold cross-validated lasso, reconstructs the
directed coefficient matrix, and calculates expected-influence summaries. It
uses the same numeric, categorical, missing-value, and standardization rules as
the original model, starting from already-scored inputs. It also specifies the
original L'Ecuyer-CMRG random-number generator for cross-validation and bootstrap
sampling. TreeSHAP contribution values are reproduced unchanged; the diagnostic
prediction baseline includes CatBoost's global bias.

Bootstrap execution is optional and is not part of the default quick run:

```sh
Rscript scripts/reproduce_clpn.R --bootstraps 5 --output results/bootstrap_smoke
Rscript scripts/reproduce_clpn.R --bootstraps 1000 --output results/bootstrap_full
```

The first command tests execution only. It must not be used to report confidence
intervals or stability coefficients. The second command requests the original
1,000 nonparametric and 1,000 case-dropping replicates and can take substantial
time. Only completed checks may be described as reproduced. See `VALIDATION.md`
for the current test status.

## Scope of reproducibility

The package supports analysis from final scores, not independent reconstruction
of scores or item-level reliability. The default checks compare newly computed
model metrics, SHAP importance, and network coefficients with the aggregate
benchmarks. Source-data collection, measurement scoring, the full history of
model selection, and pixel-identical manuscript figure rendering are outside
this package.

The current scripts do not regenerate descriptive tables, LOWESS curves and
their zero crossings, the 500-resample test-metric distributions, two-step bridge
expected influence, node predictability excluding autoregression, squared
outgoing influence, or all bootstrap difference-test and stability-curve
summaries. The presence of an aggregate benchmark does not mean that every
reported statistic is recalculated by these scripts.

The package does not access the network, the original research folder, or any
questionnaire database. Outputs are written only under the selected output
directory. No repository upload or data-publication command is provided.

## Release scope

This release contains scored analysis inputs and downstream code. Original
questionnaire records, measurement-scoring programs, and the source-identifier
mapping are not distributed. Removal of source identifiers and item responses
is not a certification of anonymity.
