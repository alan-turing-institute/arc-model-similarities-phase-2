# Usage: scripts/slurm_metrics_all.sh mmd drop-only
for FILE in find ./metrics_scripts/${1}/${2}*
do
    sbatch $FILE
done
