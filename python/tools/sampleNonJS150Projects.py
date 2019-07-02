import ast
import os
import sys

from random import sample

SAMPLE_SIZE = 100

if __name__ == "__main__":
    project_repos = set()

    with open('reposJS150', 'r') as f:
        for line in f:
            repo = line.rstrip()
            project_repos.add(repo)
    
    with open('/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/code/DeepSStuBs/sampledJSTestProjectNames', 'r') as  f:
        projects = ast.literal_eval(f.read())
        for repo in projects:
            project_repos.add(repo)

    path = '/media/mpatsis/SeagateExternal/PhD/rafaelository/data/GHCorpora/MAST/JavaScript'
    repos = next(os.walk(path))[1]
    for_sample = []
    for repo in repos:
        # print(repo, repo in repos)
        if not repo in project_repos:
            for_sample.append(repo)
    
    for sampled_project in sample(for_sample, SAMPLE_SIZE):
        print(sampled_project)
