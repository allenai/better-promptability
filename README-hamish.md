# Notes on running experiments with this codebase

I've added three scripts:
- `train_all_clusters.sh` for training based on each cluster.
- `eval_all_clusters.sh` for evaluating on each cluster. change the checkpoint used in `configs/0shot_eval_subset.jsonnet` to change the model evaluated. The output from this can be fed into `summarise_log.py` to produce the results per-task from each cluster space separated on each line (this can be copy-pasted to google sheets easily). (you can save the output doing something like `eval_all_clusters.sh &> tmp.txt`)
- `run_all_evals.sh` just runs evals without subsets, used for generating the hidden states. Note that you'll have to uncomment some code in `/meta_learn_prompt/models/prefix_transformer.py` to save hidden states.

See the interior of each script to see how to run stuff individually - most of the time the script just substitutes in the particular task/cluster file name into a config file with everything else setup.