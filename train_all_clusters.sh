# 'clusters' should contain the subsample files.
for file in /home/hamishi/t0_stuff/cluster_indices/train_cluster*; do
    clusterfile=$(echo $file | sed 's_/_\\/_g')
    echo "stuff: ${clusterfile}"
    sed -e "s/subsampleindicesfile/${clusterfile}/g" configs/multi_task_subset.jsonnet > multi_task_subset_tmp.jsonnet;
    CKPT=null tango run -d output multi_task_subset_tmp.jsonnet;
    rm multi_task_subset_tmp.jsonnet;
    break;
done
