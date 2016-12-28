'''
Created on December 27, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import os
import os.path as osp

def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
        
                
def copy_folder_structure(top_dir, out_dir):    
    if top_dir[-1] != os.sep:
        top_dir += os.sep
    
    all_dirs  = (dir_name for dir_name, _, _ in os.walk(top_dir))
    all_dirs.next() # Exhaust first name which is identical to the top_dir.    
    for d in all_dirs:                
        create_dir(osp.join(out_dir, d.replace(top_dir, '')))
        

def shuffle_lines_of_file(in_file, out_file, seed=0):
    np.random.seed(seed) 
    with open(in_file, 'r') as f_in:
        all_lines = f_in.readlines()
        np.random.shuffle(all_lines)
        
    with open(out_file, 'w') as f_out:
        f_out.writelines(all_lines)