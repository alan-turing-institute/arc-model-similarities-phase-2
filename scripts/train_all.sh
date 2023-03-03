# Useage: scripts/train_all.sh drop-only
for FILE in find ./train_scripts/${1}*
do
    sbatch $FILE
done
