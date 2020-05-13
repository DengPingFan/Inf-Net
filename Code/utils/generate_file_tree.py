# -*- coding: utf-8 -*-
import sys
from pathlib import Path


class DirectionTree(object):
    """generate file tree like `tree E:/xxx/xxx` command in windows
    pathname: target directory
    filename: save directory, default = 'tree.txt'
    """

    def __init__(self, pathname='.', filename='tree.txt'):
        super(DirectionTree, self).__init__()
        self.pathname = Path(pathname)
        self.filename = filename
        self.tree = ''

    def set_path(self, pathname):
        self.pathname = Path(pathname)

    def set_filename(self, filename):
        self.filename = filename

    def generate_tree(self, n=0):
        if self.pathname.is_file():
            self.tree += '    |' * n + '-' * 4 + self.pathname.name + '\n'
        elif self.pathname.is_dir():
            self.tree += '    |' * n + '-' * 4 + \
                str(self.pathname.relative_to(self.pathname.parent)) + '\\' + '\n'

            for cp in self.pathname.iterdir():
                self.pathname = Path(cp)
                self.generate_tree(n + 1)

    def save_file(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.tree)


if __name__ == '__main__':
    """Usage:
    When params_number = 1, print the dir_tree of current repository, like:
        `python generate_file_tree.py`
    elif params_number = 2, print the dir_tree of assigned repository, like:
        `python generate_file_tree.py E:/xxx/xxx`
    elif params_number = 3, print the dir_tree of assigned repository and save as '*.txt' file, like:
        `python generate_file_tree.py D:/Github/Inf-Net dirtree.txt` 
    """
    dirtree = DirectionTree()
    if len(sys.argv) == 1:
        dirtree.set_path(Path.cwd())
        dirtree.generate_tree()
        print(dirtree.tree)
    elif len(sys.argv) == 2 and Path(sys.argv[1]).exists():
        dirtree.set_path(sys.argv[1])
        dirtree.generate_tree()
        print(dirtree.tree)
    elif len(sys.argv) == 3 and Path(sys.argv[1]).exists():
        dirtree.set_path(sys.argv[1])
        dirtree.generate_tree()
        dirtree.set_filename(sys.argv[2])
        dirtree.save_file()
    else:
        raise ValueError('Invalid parser! Double-check your configuration.')