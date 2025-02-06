import os
import shutil

def copy_images(source_list, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    file = open(source_list, 'r')
    for filename in file.readlines():
        filename = filename.replace('\n', '')
        if filename == '':
            continue
        target_file = os.path.join(target_folder, os.path.basename(filename))
        shutil.copy2(filename, target_file)

copy_images(r'D:\label\AI\cfg\cfg_trans_new2\valid.txt', 'val')
