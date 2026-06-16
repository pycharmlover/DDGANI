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
/param/param.json
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
| `loss_weight`      | Weight of the loss function                               |

## Run an Experiment

Enter the experiment directory:

```bash
cd test_main
```

Run DDGANI on the Adult dataset:

```bash
python main.py --Data adult --MissType MCAR --MissRate 0.2 --UseAttention True --UseLearner True --UseCFD True
```

The `Data` parameter must be consistent with the `name` field in `/param/param.json`.

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
| `UseCFD`       | Enable the data dependency plugin         |

## Adding a New Tabular Dataset

To add a new tabular dataset, place the dataset file under one of the following folders:

```bash
dataset/mix datasets
dataset/numerical datasets
```

Then add the corresponding dataset configuration in:

```bash
/param/param.json
```

Make sure that the dataset name in `param.json` matches the value passed to the `--Data` parameter when running the experiment.

## Baseline Methods

Several publicly available or reproduced data filling methods are provided in the `BaseLine` folder.

Most baseline methods can be executed by modifying the corresponding code in:

```bash
/test_main/main.py
```

For the remaining baseline methods, please refer to their official GitHub repositories for detailed running instructions.
