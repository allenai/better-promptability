'''
A small script to get the number of samples for each task in each cluster.
Assumes pickle file follows the index file used by the rest of repo.
'''
import pickle
import sys

filename = f'../t0_clusters/{sys.argv[1]}/'

tasks = [t.strip() for t in open('task_lists/green.txt', 'r').readlines()]

def counter(idx):
    x = pickle.load(open(filename + f'test_cluster_{idx}_indices.pkl', 'rb'))
    for t in tasks:
        print(len(x.get(t, [])), end=' ')
    print()

for t in tasks:
    print(t, end=' ')
print()

for idx in [0, 10, 11, 12, 13, 14]:
    counter(idx)

for idx in range(1, 10):
    counter(idx)
