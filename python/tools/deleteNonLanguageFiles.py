import os
import sys

VERBOSE = True
ENDL = '\n'

language = 'JavaScript'
lang_to_suffix = {
    'JavaScript': '.js',
    'Java': '.java',
    'python': '.py'
}


def delete_files_without_suffix(projects_home, suffix):
    repos = next(os.walk(projects_home))[1]
    
    for repo in repos:
        if VERBOSE: print('Deleting non-matching files from repo:', repo)
        repo_contents = os.walk(os.path.join(projects_home, repo))
        for entry in repo_contents:
            deletion_files = [os.path.join(entry[0], file) \
                for file in entry[2] if not file.endswith(suffix)]
            for deletion_file in deletion_files:
                os.remove(deletion_file)
            

if __name__ == "__main__":
    projects_home = sys.argv[1]
    language = sys.argv[2]
    assert(language in lang_to_suffix)

    suffix = lang_to_suffix[language]
    delete_files_without_suffix(projects_home, suffix)
