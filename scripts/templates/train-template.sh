#!/bin/sh
#SBATCH --account vjgo8416-mod-sim-2  # Only required if you are a member of more than one Baskerville project
#SBATCH --qos turing  # upon signing-up to Baskerville you will be assigned a qos
#SBATCH --time 0-0:30:0  # Time assigned for the simulation
#SBATCH --nodes 1  # Normally set to 1 unless your job requires multi-node, multi-GPU
#SBATCH --gpus 1  # Resource allocation on Baskerville is primarily based on GPU requirement
#SBATCH --cpus-per-gpu 36  # This number should normally be fixed as "36" to ensure that the system resources are used effectively
#SBATCH --job-name ms2-{{experiment_name}}  # Title for the job
#SBATCH --output=/output/%j.out

module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your Conda environment (modify as appropriate)
# N.B. this path will be created by the subsequent commands if it doesn't already exist
CONDA_ENV_PATH="/bask/projects/v/vjgo8416-mod-sim-2/ms2env"

conda activate ${CONDA_ENV_PATH}
{{python_call}}
