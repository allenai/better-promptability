'''
Small script to summarise the really big logs I get when evaluating across all clusters.
'''
import re
import sys
from statistics import mean

with open(sys.argv[1], 'r') as f:
    loglines = '\n'.join(f.readlines())

# split into the clusters
# Evaluating cluster /home/hamishi/t0_stuff/cluster_indices/test_cluster_0_indices.pkl
# /net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_tfidf/test_cluster_0_indices.pkl
loglines = loglines.split('Evaluating cluster /net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_')

print(len(loglines))

print([line[0:50] for line in loglines])

def score_logline(logline):
     # grab the tasks in the order we want
    task_pattern = re.compile(r'evaluating task ([\d\w\.]+)')
    tasks = [x.group(1) for x in re.finditer(task_pattern, logline)]
    # for each cluster, grab all datasets we didn't see
    missing_key_pattern = re.compile(r'KeyError: \'([\w\d\.]+)\'')
    missing_datasets = [x.group(1) for x in re.finditer(missing_key_pattern, logline)]
    # sometimes the url stuff breaks down. grab the immediate next dataset and add the previous one to missing ds
    missing_key_pattern = re.compile(r'url\: [\d\w\.\/_:]+\n\nevaluating task ([\d\w\.]+)')
    next_datasets = [x.group(1) for x in re.finditer(missing_key_pattern, logline)]
    for ds in next_datasets:
        missing_datasets.append(tasks[tasks.index(ds) - 1])
    # aaand sometimes I get a fileexists error for some reason.
    # FileExistsError: [Errno 17] File exists: 'output/runs/one-adder' -> 'output/latest'
    missing_key_pattern = re.compile(r'FileExistsError: \[Errno 17\] File exists: \'[\w\d\s>\-\/\']+\'\n\nevaluating task ([\d\w\.]+)')
    next_datasets = [x.group(1) for x in re.finditer(missing_key_pattern, logline)]
    for ds in next_datasets:
        missing_datasets.append(tasks[tasks.index(ds) - 1])
    # ive seen this error type once, too 
    # IndexError: Invalid value 10047 in indices iterable. All values must be within range [-10042, 10041].
    # evaluating task hellaswag_complete_first_then_score_eval
    missing_key_pattern = re.compile(r'IndexError: Invalid value [\d]+ in indices iterable\. All values must be within range \[[0-9\s,\-]+\]\.\n\nevaluating task ([\d\w\.]+)')
    next_datasets = [x.group(1) for x in re.finditer(missing_key_pattern, logline)]
    for ds in next_datasets:
        missing_datasets.append(tasks[tasks.index(ds) - 1])
    # FileNotFoundError: [Errno 2] No such file or directory: '/net/nfs.cirrascale/allennlp/hamishi/meta-learn-prompt/0shot_eval_subset_tmp_2.jsonnet'
    # evaluating task winogrande_winogrande_xl_stand_for_score_eval
    missing_key_pattern = re.compile(r'FileNotFoundError: \[Errno 2\] No such file or directory: \'[\/\-\.\w\d_]+\'\n\nevaluating task ([\d\w\.]+)')
    next_datasets = [x.group(1) for x in re.finditer(missing_key_pattern, logline)]
    for ds in next_datasets:
        missing_datasets.append(tasks[tasks.index(ds) - 1])
    # grab scores
    score_pattern = re.compile(r'\{\'categorical_accuracy\': ([\.\d]+)\}')
    scores = [float(x.group(1)) for x in re.finditer(score_pattern, logline)]
   
    filtered_tasks = [t for t in tasks if t not in missing_datasets]
    # print(len(filtered_tasks),len(scores))
    # print(filtered_tasks)
    # print(missing_datasets)
    # print(logline[0:100])
    
    assert(len(filtered_tasks) == len(scores))
    # map scores to tasks 
    task_scores = {task: score for task, score in zip(filtered_tasks, scores)}
    for t in missing_datasets:
        task_scores[t] = '0'
    return mean(scores), task_scores

tasks = [t.strip() for t in open('task_lists/green.txt', 'r').readlines()]
#tasks = [t for t in score_logline(loglines[1])[1]]

for task in tasks:
    print(task, end=' ')
print()
for line in loglines[1:]:
    res = score_logline(line)[1]
    for task in tasks:
        print(res.get(task, 0.0), end=' ')
    print()

