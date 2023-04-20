# Scripts

This README describes which scripts to call to replicate the results of this project. There are separate scripts for computing dataset similarity metrics, model training, and attack transferring.

Note that while measuring dataset similarity and model training can be performed in any order, performing transfer attacks must be done after training the models.

For each of these three tasks, a generation script exists which will generate bash/slurm scripts that can be called individually for a given dataset pair.

**Requirements:** For broad requirements beyond installing the project source code, see [requirements](#requirements) below.

**Important:** for model training and transfer attacks, we use wandb to store our results (see more in [requirements](#requirements) below). Our training script will produce an exception if the run name already exists on wandb, and the attack script will produce an exception if the run name is not unique on wandb. To perform the same run again, it will be necessary to either rename the old run, rename the new run, or delete the old run.

## Calculate Metrics

The metrics calculation script `scripts/calculate_metrics.py` will take a dataset config and metrics configs as input along with additional parameters to determine which dataset pair to compute metrics for. For a single dataset pair in the dataset config, it will compute every metric in the metrics config. The results will then be stored in a JSON file.

You can generate bash scripts for every dataset pair in a given experiment group in the dataset config by calling

```bash
python scripts/generate_metrics_scripts.py --experiment_group "drop-only"
```

where `"drop-only"` is an example of an exerperiment group in our dataset config (see below for more information on configs). It will generate bash scripts in `ROOT/metrics_scripts/`. Note additional optional arguments that can be passed to the generation script include `--dataset_config_path` and `--metrics_config_path`.

Once this is done, you can call each of the individual scripts individually, e.g.

```bash
metrics_scripts/drop-only_0_0_metrics.sh
```

to compute metrics for that particular DMPair, where `drop-only_0_0_metrics.sh` is one of the bash scripts generated for the `drop-only` experiment group, with dataset pair index `0` and seed index `0`. Note that the JSON results file from calling this will land in a `ROOT/results/` folder which will be created if it does not already exist.

## Train Models

The model training script `scripts/train_models.py` will take a dataset config and trainer configs as input along with additional parameters to determine which dataset pair to train models for. For a single dataset pair in the dataset config, it will train a model for each dataset in that pair and log the results to wandb.

You can generate slurm scripts for every dataset pair in a given experiment group in the dataset config by calling

```bash
python scripts/generate_train_scripts.py --experiment_group "drop-only"
```

where `"drop-only"` is an example of an exerperiment group in our dataset config (see below for more information on configs). It will generate bash scripts in `ROOT/train_scripts/`. Note additional optional arguments to the generation script include `--dataset_config_path`, `--metrics_config_path`, `--account_name`, and `--conda_env_path` (see below).

Once this is done, you can run all of the generated scripts for a given experiment group by calling `scripts/train_all.sh` followed by the name of the experiment group, e.g.

```bash
scripts/train_all.sh drop-only
```

where once again `drop-only` is an example of an exerperiment group in our dataset config.

Alternatively, you can call each of the scripts individually, e.g.

```bash
sbatch trainer_scripts/drop-only_0_0_trainer.sh
```

to train models for that particular dataset pair, where `drop-only_0_0_trainer` is one of the bash scripts generated for the `drop-only` experiment group, with dataset pair index `0` and seed index `0`. Note that logging is performed with wandb and so you will need this set up on your machine in order to proceed. Note also that the output directory specified in the slurm template (see [requirements](#requirements) below.) must exist prior to running the scripts.

## Transfer Attacks

The transfer attack script `scripts/attack.py` will take a dataset config, trainer config, and transfer attack config as input along with additional parameters to determine which dataset pair to perform transfer attacks for. For a single dataset pair, it will generate two sets of adversarial images for each model in corresponding to that pair. For each pair, it will generate one adversarial attack using that model's test dataset, and one adversarial attack using the other model's test dataset (so e.g. for model_A, it will produce adversial images for test_A and test_B).

It will do this for every attack specified in the attack config (note however we have only implemented two attack types from foolbox for the package code). It will also perform an attack for every value of epsilon supplied in the config, and perform best image selection by choosing the first successful image with the lowest value of epsilon. If no generated image is successful in the attack generation step, it will select the adversarial image generated by the highest value of epsilon.

You can generate slurm scripts for every dataset pair in a given experiment group in the dataset config by calling

```bash
python scripts/generate_attack_scripts.py --experiment_group "drop-only"
```

where `"drop-only"` is an example of an exerperiment group in our dataset config (see below for more information on configs). It will generate bash scripts in `ROOT/attack_scripts/`. Note additional optional arguments include `--dataset_config_path`, `--trainer_config_path`, and `--attack_config_path` (see below).

Once this is done, you can run all of the generated scripts for a given experiment group by calling `scripts/attack_all.sh` followed by the name of the experiment group, e.g.

```bash
scripts/attack_all.sh drop-only
```

where once again `drop-only` is an example of an exerperiment group in our dataset config.

Alternatively, you can call each of the scripts individually, e.g.

```bash
sbatch attack_scripts/drop-only_0_0_attack.sh
```

to perform transfer attacks for that particular dataset pair, where `drop-only_0_0_trainer` is one of the bash scripts generated for the `drop-only` experiment group, with dataset pair index `0` and seed index `0`.

As stated above, note that a unique run must exist on wandb for the transfer attack to be performed. Results for model A as the surrogate model (i.e. with B as the target model) will be logged to model A on wandb, and vice versa for model B.  Note also that the output directory specified in the slurm template (see [requirements](#requirements) below.) must exist prior to running the scripts.

## Requirements

### Training Requirements

The slurm scripts generated for model training contain some arguments specific to our HPC, and the templates will likely need some modifications to be appropriate to your own usage. See our [train template](/scripts/templates/slurm-train-template.sh) for specific arguments.

Broadly, the requirements are:

- slurm
- an existing conda environment (passed to `--conda_env_path`)
- wandb

Please note in particular that the account name (passed to the generation script via `--account_name`) argument is specific to our HPC and may not be required for your usage.

### Transfer Attack Requirements

The slurm scripts generated for the transfer attacks contain some arguments specific to our HPC, and the templates will likely need some modifications to be appropriate to your own usage. See our [transfer attack template](/scripts/templates/slurm-attack-template.sh) for specific arguments.

For the transfer attack scripts to work, you will need models with the correct names and IDs on your wandb account.

Like those for the training scripts, the slurm scripts generated for model training contain some arguments specific to our HPC, and the templates will likely need some modifications to be appropriate to your own usage. This will like the training script require an existing conda environment (passed to `--conda_env_path`).

## Configs

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
