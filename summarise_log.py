'''
Small script to summarise the really big logs I get when evaluating across all clusters.
'''
import re
import sys
from statistics import mean

with open(sys.argv[1], 'r') as f:
    loglines = '\n'.join(f.readlines())

# split into the clusters
loglines = loglines.split('Evaluating cluster /home/hamishi/t0_stuff/cluster_indices/test_cluster_')

print(len(loglines))

def score_logline(logline):
    # for each cluster, grab all datasets we didn't see
    missing_key_pattern = re.compile(r'KeyError: \'([\w\d\.]+)\'')
    missing_datasets = [x.group(1) for x in re.finditer(missing_key_pattern, logline)]
    # grab scores
    score_pattern = re.compile(r'\{\'categorical_accuracy\': ([\.\d]+)\}')
    scores = [float(x.group(1)) for x in re.finditer(score_pattern, logline)]
    # grab the tasks in the order we want
    task_pattern = re.compile(r'evaluating task ([\d\w\.]+)')
    tasks = [x.group(1) for x in re.finditer(task_pattern, logline)]
    filtered_tasks = [t for t in tasks if t not in missing_datasets]
    assert(len(filtered_tasks) == len(scores))
    # map scores to tasks 
    task_scores = {task: score for task, score in zip(filtered_tasks, scores)}
    for t in missing_datasets:
        task_scores[t] = 0
    return mean(scores), task_scores

tasks = [t for t in score_logline(loglines[1])[1]]

for task in tasks:
    print(task, end=' ')
print()
for line in loglines[1:]:
    res = score_logline(line)[1]
    for task in tasks:
        print(res[task], end=' ')
    print()

