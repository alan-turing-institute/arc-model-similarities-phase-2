#!/bin/sh
#SBATCH --account {{account_name}}
#SBATCH --qos turing
#SBATCH --time 0-0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name ms2-{{experiment_name}}
#SBATCH --output ./slurm_train_logs/{{experiment_name}}-train-%j.out

module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your Conda environment (modify as appropriate)
CONDA_ENV_PATH={{conda_env_path}}

conda activate ${CONDA_ENV_PATH}
{{python_call}}
