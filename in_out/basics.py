"""Very basic in-out utilities.
The MIT License (MIT)
Originally created in 2016 for Python 2.x. (Updated in 2019 for Python 3.x.)
@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
"""

import os
import os.path as osp
import numpy as np
import warnings
import re
from six.moves import cPickle
from six.moves import range


def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """ Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def create_dir(dir_path):
    """ Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def delete_files_in_directory(dir_path):
    """ Deletes all files that are directly under the dir_path.
    """
    file_list = [f for f in os.listdir(dir_path) if osp.isfile(osp.join(dir_path, f))]
    for f in file_list:
        os.remove(osp.join(dir_path, f))


def copy_folder_structure(top_dir, out_dir):
    if top_dir[-1] != os.sep:
        top_dir += os.sep

    all_dirs = (dir_name for dir_name, _, _ in os.walk(top_dir))
    next(all_dirs)     # Exhaust first name which is identical to the top_dir.
    for d in all_dirs:
        create_dir(osp.join(out_dir, d.replace(top_dir, '')))


def shuffle_lines_of_file(in_file, out_file, seed=None):
    if seed is not None:
        np.random.seed(seed)
    with open(in_file, 'r') as f_in:
        all_lines = f_in.readlines()
        np.random.shuffle(all_lines)

    with open(out_file, 'w') as f_out:
        f_out.writelines(all_lines)


def boot_strap_lines_of_file(file_in, lines_total, file_out, skip_rows=0, seed=None):
    """ Copies or removes at random lines of the input file, so that its total number
    of lines is as requested.
    """
    # TODO: check again if last-line to be newline
    with open(file_in, 'r') as f_in:
        original_lines = f_in.readlines()

    if skip_rows > 0:
        first_lines = original_lines[:skip_rows]
        original_lines = original_lines[skip_rows:]

    new_line = False
    if original_lines[-1].endswith('\n'):
        new_line = True
    else:
        original_lines[-1] = original_lines[-1] + '\n'

    diff = len(original_lines) - lines_total
    orig_l = np.arange(len(original_lines))

    if seed is not None:
        np.random.seed(seed)

    if diff > 0:
        drop_index = np.random.choice(orig_l, diff, replace=False)
        keep_index = np.setdiff1d(orig_l, drop_index)
    elif diff < 0:
        boot_strap_index = np.random.choice(orig_l, -diff, replace=True)
        keep_index = np.hstack((orig_l, boot_strap_index))
    else:
        keep_index = orig_l

    keep_index = sorted(keep_index)
    with open(file_out, "w") as f_out:
        if skip_rows > 0:
            f_out.writelines(first_lines)
        for l in keep_index[:-1]:
            f_out.write(original_lines[l])
        if not new_line:
            f_out.write(original_lines[-1][:-1])
        else:
            f_out.write(original_lines[-1])


def read_header_of_np_saved_txt(in_file):
    header = ''
    with open(in_file, 'r') as f_in:
        for line in f_in:
            if line.startswith('#'):
                header = header + line
            else:
                break

    if header == '':
        warnings.warn('No header exists in this file.')
        header = None

    return header


def files_in_subdirs(top_dir, search_pattern, followlinks=False):
    """ Return a generator of all files in subdirectories that match the given regular expression (pattern).
    """
    join = os.path.join
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir, followlinks=followlinks):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                yield full_name


def immediate_subdirectories(top_dir, full_path=True):
    dir_names = [name for name in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, name))]
    if full_path:
        dir_names = [osp.join(top_dir, name) for name in dir_names]
    return dir_names


def splitall(path):
    """
    Examples:
        splitall('a/b/c') -> ['a', 'b', 'c']         
        splitall('/a/b/c/')  -> ['/', 'a', 'b', 'c', '']
        
    NOTE: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = osp.split(path)
        if parts[0] == path:   # Sentinel for absolute paths.
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # Sentinel for relative paths.
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def str2bool(v):
    """ boolean values for argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
