# Author : Bryce Xu
# Time : 2020/2/29
# Function: 

import os, shutil

image_dir = '/mnt/datadev_2/std/SUN/images/'

sub_dirs = os.listdir(image_dir)

def copy_file(origin_dir, target_dir):
    dir_files = os.listdir(origin_dir)
    for file in dir_files:
        file_path = os.path.join(origin_dir, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)

for sub_dir in sub_dirs:
     sub_current_dir = os.path.join(image_dir, sub_dir)
     subsub_dirs = os.listdir(sub_current_dir)
     for subsub_dir in subsub_dirs:
         subsub_current_dir = os.path.join(sub_current_dir, subsub_dir)
         if os.path.isdir(subsub_current_dir):
             create_dir = os.path.join(image_dir, sub_dir + '_' + subsub_dir)
             os.mkdir(create_dir)
             target_dir = create_dir
             origin_dir = subsub_current_dir
             copy_file(origin_dir, target_dir)