if __name__ == "__main__":
    project_repos = set()

    with open('data/js/programs_clean_training.txt', 'r') as f:
        for line in f:
            repo = line.split('/')[3] + '.' + line.split('/')[4]
            project_repos.add(repo)

    with open('data/js/programs_clean_eval.txt', 'r') as f:
        for line in f:
            repo = line.split('/')[3] + '.' + line.split('/')[4]
            project_repos.add(repo)

    for project_repo in project_repos:
        print(project_repo)
