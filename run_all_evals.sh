
for file in task_lists/green.txt; do
    while read -r line; do
        sed -e "s/taskname/${line}/g" configs/0shot_eval.jsonnet > 0shot_eval_tmp.jsonnet;
        mixture="$(basename -s .txt $file)"
        sed -i -e "s/mixturename/${mixture}/g" 0shot_eval_tmp.jsonnet;
        CKPT=null tango run -d output 0shot_eval_tmp.jsonnet;
        rm 0shot_eval_tmp.jsonnet;
    done < $file
done
