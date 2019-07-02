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

# path_remove_str = '/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/codeCorpora/MAST/JavaScript'
path_remove_str = '/media/mpatsis/SeagateExternal/PhD/rafaelository/data/GHCorpora/MAST/JavaScript'

def get_files_with_suffix(projects_home, suffix):
    repos = next(os.walk(projects_home))[1]
    
    for repo in repos:
        repo_filtered_files = []
        if VERBOSE: print('Collecting files from repo:', repo)
        repo_contents = os.walk(os.path.join(projects_home, repo))
        for entry in repo_contents:
            filtered_files = [os.path.join(entry[0], file) \
                for file in entry[2] if file.endswith(suffix)]
            filtered_files = ['data/js' + file[len(path_remove_str):] for file in filtered_files]
            repo_filtered_files.extend(filtered_files)
        yield repo_filtered_files


if __name__ == "__main__":
    projects_home = sys.argv[1]
    language = sys.argv[2]
    assert(language in lang_to_suffix)

    suffix = lang_to_suffix[language]
    with open('programs_JS_dataset.txt', 'w') as f:
        for lang_files in get_files_with_suffix(projects_home, suffix):
            for lang_file in lang_files:
                f.write(lang_file)
                f.write(ENDL)
