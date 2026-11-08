# DDGANI

This repository provides the implementation of **DDGANI**, a table data filling algorithm for missing value imputation. It includes several UCI dataset examples, configurable experiment settings, and baseline methods for comparison.

## Setup Environment

Create and activate the conda environment:

```bash
conda create -n DDGANI python=3.10
conda activate DDGANI
pip install -r requirements.txt
```

## Configurations

Algorithm parameters can be modified in:

```bash
param/param.json
```

You can also add configuration items for your own datasets in this file. The main parameters are listed below:

| Parameter          | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `name`             | Dataset name                                              |
| `T`                | Number of diffusion steps                                 |
| `file_path`        | Dataset file path                                         |
| `categorical_cols` | Index list of categorical columns                         |
| `top_k`            | Top-k samples used for attention-based filling            |
| `model_name`       | Preprocessing method for numerical data, such as `minmax` |
| `epochs`            | Number of diffusion training epochs                       |
| `lr`                | Learning rate                                             |
| `batch_size`        | Training batch size                                       |
| `steps_per_epoch`   | Number of batches used for each training update           |
| `loss_weight`      | Weight of the loss function                               |
| `use_FD`            | Enable the functional-dependency (FD) method              |
| `use_CFD`           | Enable the conditional-functional-dependency (CFD) method  |

The values of `use_FD` and `use_CFD` must be strings with exact capitalization:

```json
"use_FD": "True",
"use_CFD": "False"
```

The selected dataset entry in `param/param.json` is the source of truth for dependency-method selection. `main.py` loads that entry and overwrites the command-line values of `UseFD` and `UseCFD`. Therefore, change `use_FD` and `use_CFD` in the corresponding dataset block before running an experiment.

## FD and CFD Methods

DDGANI currently supports two data-dependency methods. Both methods mine dependency rules from the observed part of the incomplete table and use the resulting neural rule models as an additional loss during diffusion training.

### FD: Functional Dependency

FD mines ordinary functional dependencies of the form:

```text
X -> Y
```

For example, the values of `X` determine the value of `Y`. The mined FD models contribute to the generator objective through `loss_weight["FD_weight"]`.

Enable FD in the dataset configuration:

```json
{
  "name": "tax",
  "use_FD": "True",
  "use_CFD": "False"
}
```

The program should print:

```text
UseFD
```

### CFD: Conditional Functional Dependency

CFD mines conditional dependencies. A CFD applies a dependency only under a specific condition or pattern, rather than requiring the dependency to hold over the entire table. Candidate rules are filtered using observed support and missing-value support bounds. The CFD loss is weighted by `loss_weight["CFD_weight"]`.

Enable CFD in the dataset configuration:

```json
{
  "name": "tax",
  "use_FD": "False",
  "use_CFD": "True"
}
```

The program should print:

```text
UseCFD
```

Use exactly one of the two methods for an experiment. The current implementation checks FD first:

```text
use_FD = "True"  -> FD is selected
use_FD = "False" and use_CFD = "True" -> CFD is selected
```

If both flags are set to `"True"`, FD takes precedence. If both are `"False"`, no dependency-rule model is mined.

## Run an Experiment

Run the experiment from the repository root:

```bash
python test_main/main.py --Data tax --MissType MCAR --MissRate 0.2 \
    --UseAttention True --UseLearner True
```

Alternatively:

```bash
cd test_main
python main.py --Data tax --MissType MCAR --MissRate 0.2 \
    --UseAttention True --UseLearner True
```

The `Data` parameter must be consistent with the `name` field in `param/param.json`. Select FD or CFD by editing the matching dataset block as described above; adding `--UseFD` or `--UseCFD` to the command does not override that block in the current implementation.

The `MissType` parameter specifies the missing data mechanism. Supported options include:

```text
MCAR, MNAR, MAR, Region
```

The `MissRate` parameter controls the missing rate.

You can enable different modules by setting the following parameters:

| Parameter      | Description                               |
| -------------- | ----------------------------------------- |
| `UseAttention` | Enable the attention-based filling module |
| `UseLearner`   | Enable the downstream learner module      |
| `use_FD`       | Select the FD dependency method           |
| `use_CFD`      | Select the CFD dependency method          |

The command-line `UseAttention` and `UseLearner` options control their respective modules. FD/CFD selection is configured per dataset in `param/param.json`.

## Adding a New Tabular Dataset

To add a new tabular dataset, place the dataset file under one of the following folders:

```bash
dataset/mix datasets
dataset/numerical datasets
```

Then add the corresponding dataset configuration in:

```bash
param/param.json
```

Make sure that the dataset name in `param.json` matches the value passed to the `--Data` parameter when running the experiment.

## Baseline Methods

Several publicly available or reproduced data filling methods are provided in the `BaseLine` folder.

Most baseline methods can be executed by modifying the corresponding code in:

```bash
/test_main/main.py
```

For the remaining baseline methods, please refer to their official GitHub repositories for detailed running instructions.
