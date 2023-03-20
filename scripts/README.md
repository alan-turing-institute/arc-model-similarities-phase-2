# Scripts

For calculation of metrics, model training, and transfer attacks, scripts exist that generate bash/slurm scripts to be called individually.

For training and attack prerequisites, see [requirements](#requirements)

## Calculate Metrics

Call:

```bash
python scripts/generate_metrics_scripts.py --experiment_group "drop-only"
```

to generate bash scripts in `ROOT/metrics_scripts/`. Note additional optional arguments include `--dataset_config_path` and `--metrics_config_path` (see below).

Once this is done, you can call each of the individual scripts individually, e.g.

```bash
metrics_scripts/drop-only_0_0_metrics.sh
```

to compute metrics for that particular DMPair. Note that the results file from calling this will land in a `ROOT/results/` folder which will be created if it does not already exist.

## Train Models

Call:

```bash
python scripts/generate_train_scripts.py --experiment_group "drop-only"
```

to generate slurm scripts in `ROOT/train_scripts/`. Note additional optional arguments include `--dataset_config_path`, `--metrics_config_path`, `--account_name`, and `--conda_env_path` (see below).

Once this is done, you can run all of the generated scripts for a given experiment group by calling `scripts/train_all.sh` followed by the name of the experiment group, e.g.

```bash
scripts/train_all.sh drop-only
```

Alternatively, you can call each of the individual scripts individually, e.g.

```bash
sbatch trainer_scripts/drop-only_0_0_trainer.sh
```

to train models for that particular DMPair. Note that logging is performed with wandb and so you will need this set up on your machine in order to proceed.

## Transfer Attacks

Call:

```bash
python scripts/generate_attack_scripts.py --experiment_group "drop-only"
```

to generate bash scripts in `ROOT/attack_scripts/`. Note additional optional arguments include `--dataset_config_path`, `--trainer_config_path`, and `--attack_config_path` (see below).

Once this is done, you can call each of the individual scripts individually, e.g.

```bash
attack_scripts/drop-only_0_0_attack.sh
```

to perform a transfer attack and compute success metrics for that particular DMPair. Note that results will be logged back to wandb.

## Requirements

### Training Requirements

The slurm script generated for model training contains some arguments specific to our HPC, and the template will likely need some modifications to be appropriate to your own usage.

Broadly, the requirements are:

- slurm
- an existing conda environment (passed to `--conda_env_path`)
- wandb

Please note in particular that the account name (passed to the generation script via `--account_name`) argument is specific to our HPC and may not be required for your usage.

### Transfer Attack Requirements

For the transfer attack scripts to work, you will need models with the correct names and IDs on your wandb account.

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
- nested within this, a key:value pair giving a function name (see our function dictionary at [this location](/src/modsim2/similarity/constants.py))
- if relevant, another key for arguments giving the key:value pairs of argument names and argument values to be passed to the metric call

You can see an example in the repository config folder as [metrics.yaml](/configs/metrics.yaml).

### Trainer Config File

This should be a YAML file containing the following elements:

- trainer_kwargs: provides keyword arguments to be passed directly to the trainer (e.g. max number of epochs)
- model: arguments to pass directly to the model
- wandb: arguments to pass directly to wandb

You can see an example in the repository config folder as [trainer.yaml](/configs/trainer.yaml).

### Attack Config File

This should be a YAML file containing the following elements:

- num_attack_images: the number of attack images to generate for each attack
- attack names: a list of attack names. Note only `L2FastGradientAttack` and `BoundaryAttack` are implemented
- epsilons: a list of values of epsilon to compute an attack for
- model_version: wandb model version for downloading a model from wandb

You can see an example in the repository config folder as [attack.yaml](/configs/attack.yaml).
