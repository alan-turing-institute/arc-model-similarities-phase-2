# Usage: scripts/slurm_attack_all.sh drop-only
for FILE in find ./attack_scripts/${1}*
do
    sbatch $FILE
done
