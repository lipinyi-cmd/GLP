# Groundwater Level Time Series Modeling

MATLAB implementations of LSTM, multi-head self-attention LSTM (MHSA-LSTM), and support vector regression (SVM). All three models use Bayesian hyperparameter optimization, with the mean RMSE over four expanding-window temporal validation rounds as the objective.

## Project Files

| File | Purpose |
| --- | --- |
| `LSTM.m` | Entry point for LSTM training and testing |
| `MHSA_LSTM.m` | Entry point for MHSA-LSTM training and testing |
| `svm.m` | Entry point for support vector regression training and testing |
| `expand_T1_trainAndValidate_trainnet_w_K_Fold_times.m` | Temporal validation objective function for LSTM |
| `expand_T1_att_trainAndValidate_trainnet_w_K_Fold_times.m` | Temporal validation objective function for MHSA-LSTM |
| `expand_kfoldLossSVM_time.m` | Temporal validation objective function for SVM |
| `figure6.m` | Metrics and plots in the original groundwater level units for LSTM/MHSA-LSTM; run the corresponding model first |
| `fig3.m` | Metrics and plots in the original groundwater level units for SVM; run `svm.m` first |
| `data/` | Directory for local input data |

## Requirements

- MATLAB: must support `trainnet` and `dlarray`; the attention model also uses `selfAttentionLayer` and `indexing1dLayer`.
- Deep Learning Toolbox: required for both LSTM models.
- Statistics and Machine Learning Toolbox: required for `bayesopt`, `optimizableVariable`, and `fitrsvm`.
- Econometrics Toolbox: SVM uses `lagmatrix` to construct lagged features.
- Parallel Computing Toolbox: optional. The entry scripts set `useParallel` to `false` by default; this toolbox is required when parallel optimization is enabled.

## Data Preparation

Place the workbook at `data/X_y_2.xlsx`. Each script locates the file relative to its own directory. Alternatively, edit `dataFile` at the top of the entry script to point to your data.

| Worksheet | Format |
| --- | --- |
| First worksheet / `X` | Exogenous features, with numeric values in the first six columns. Both LSTM models read the first worksheet, while SVM reads `X` by name, so place `X` first. |
| `y_30860019` | Default target for LSTM; the second column contains groundwater levels. |
| `y_30860016` | Default target for MHSA-LSTM; the second column contains groundwater levels. |
| `y_30860017` | Default target for SVM; the second column contains groundwater levels. |

All worksheets must have matching rows in the same chronological order, with headers in the first row. Feature columns 1–5 represent radiation, air temperature, relative humidity, precipitation, and ET, respectively. Define the sixth column according to your dataset. Both LSTM models use all six columns and append the groundwater level from the previous time step. SVM uses precipitation, ET, and selected lagged groundwater level, radiation, and humidity features.

Provide enough numeric data rows and handle missing values and constant columns before running the scripts to avoid invalid values during normalization. The original observation workbook is not distributed with this project.

## Usage

In MATLAB, set the current folder to this directory and run a model. To obtain metrics in the original groundwater level units and additional plots, then run the corresponding plotting script:

```matlab
LSTM
figure6
```

```matlab
MHSA_LSTM
figure6
```

```matlab
svm
fig3
```

The plotting scripts read variables left in the workspace by the preceding model, so run them in the same workspace.

## Default Experiment Settings

| Setting | LSTM | MHSA-LSTM | SVM |
| --- | --- | --- | --- |
| Target worksheet | `y_30860019` | `y_30860016` | `y_30860017` |
| Number of data rows read | 730 | 2191 | 2191 |
| Time window | 7 | 30 | Explicit lags of up to 7 time steps |
| Initial training split | 80% | 80% | 80% |
| Training data used for the final model | All training windows | All training windows | Last 730 samples of the training partition |
| Bayesian optimization evaluations | 30 | 30 | 30 |
| Epochs per neural network training run | 100 | 100 | Not applicable |

Change the station, number of data rows, and parallel processing option at the top of each entry script. Modify the window length and other model parameters in the corresponding script. The original choices of different stations and sample lengths are retained, so the default results cannot be directly compared as models evaluated under identical data conditions.

## Uploading to GitHub

Use this directory as the repository root. Commit the `.m` source files, README files, `data/README.md`, and Git configuration files. The `.gitignore` file excludes observation data, MATLAB workspaces and models, figure files, output directories, and temporary files; data remain on your local machine.

To initialize a repository for the first time, run the following commands in this directory:

```sh
git init
git add .
git diff --cached --stat
git commit -m "Prepare groundwater forecasting code for GitHub"
git branch -M main
```

Then connect your GitHub repository and push the changes. No open-source license has been specified for this project. Before publicly sharing the code or permitting reuse, the code rights holder should select an appropriate license.
