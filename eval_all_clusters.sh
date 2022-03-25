# 'clusters' should contain the subsample files.
for file in /net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_*; do
    echo "Evaluating cluster ${file}"
    for file2 in task_lists/green.txt; do
        while read -r line; do
            clusterfile=$(echo $file | sed 's_/_\\/_g')
            sed -e "s/subsampleindicesfile/${clusterfile}/g" configs/0shot_eval_subset.jsonnet > 0shot_eval_subset_tmp.jsonnet;
            echo "evaluating task ${line}"
            sed -i -e "s/taskname/${line}/g" 0shot_eval_subset_tmp.jsonnet;
            mixture="$(basename -s .txt $file2)"
            sed -i -e "s/mixturename/${mixture}/g" 0shot_eval_subset_tmp.jsonnet;
            CKPT=null tango run -d output 0shot_eval_subset_tmp.jsonnet;
            rm 0shot_eval_subset_tmp.jsonnet;
        done < $file2
    done
done
