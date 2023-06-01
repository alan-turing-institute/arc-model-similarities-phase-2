# Usage: scripts/slurm_attack_all.sh mmd drop-only
for FILE in find ./attack_scripts/${1}/${2}*
do
    sbatch $FILE
done
