# Scripts

For both calculation of metrics and model training, two scripts exist that generate bash scripts to be called individually. These bash scripts will land in a `ROOT/bash_scripts/` folder.

## Calculate Metrics

Call:

```bash
python scripts/generate_metrics_scripts.py --experiment_group "drop-only"
```

to generate bash scripts in `ROOT/bash_scripts/`. Note additional optional arguments include `--dataset_config_path` and `--metrics_config_path` (see below).

Once this is done, you can call each of the individual scripts individually, e.g.

```bash
bash_scripts/drop-only_0_0_metrics.sh
```

to compute metrics for that particular DMPair. Note that the results file from calling this will land in a `ROOT/results/` folder which will be created if it does not already exist.

## Train Models

Call:

```bash
python scripts/train_script.py
```

to generate bash scripts in `ROOT/bash_scripts/`. Note additional optional arguments include `--dataset_config_path` and `--trainer_config_path` (see below).

Once this is done, you can call each of the individual scripts individually, e.g.

```bash
bash_scripts/drop-only_0_0_train.sh
```

to train models for that particular DMPair. Note that logging is performed with wadnb and so you will need this set up on your machine in order to proceed.

## Config Files

Three config files need to be setup for the scripts to work. A dataset config file, a metrics config file, and a trainer config file.

### Dataset Config File

This should be YAML file containing the following elements:

- seeds: a list of seed numbers to use in generating datasets. For each seed, every DMPair combination will be generated
- val_split: a single number, the size of the validation split for model training
- experiment_group: groups of experiments to run separately. Within each group should be several dataset specifications, covering drop percentages and dataset transformations.

You can see an example in the repository config folder as [datasets.yaml](/configs/datasets.yaml).

### Metrics Config File

This should be a YAML file containing the elements with the following structure

- an initial key denoting the name you wish to assign to the metric
- nested within this, a key:value pair giving a function name (see our function dictionary at [this location](src/modsim2/similarity/constants.py))
- if relevant, another key for arguments giving the key:value pairs of argument names and argument values to be passed to the metric call

You can see an example in the repository config folder as [metrics.yaml](/configs/metrics.yaml).

### Trainer Config File

This should be a YAML file containing the following elements:

- trainer_kwargs: provides keyword arguments to be passed directly to the trainer (e.g. max number of epochs)
- model: arguments to pass directly to the model
- wandb: arguments to pass directly to wandb

You can see an example in the repository config folder as [trainer.yaml](/configs/trainer.yaml).
