import os
import shutil

def copy_images(source_list, target_folder):
    if not os.path.exists('labels\\' + target_folder):
        os.makedirs('labels\\' + target_folder)

    target_folder = 'labels\\' + target_folder

    file = open(source_list, 'r')
    for filename in file.readlines():
        filename = filename.replace('.jpg\n', '.txt')
        if filename == '':
            continue
        target_file = os.path.join(target_folder, os.path.basename(filename))
        source_file = os.path.join('yolov8_labels', os.path.basename(filename))
        shutil.copy2(source_file, target_file)

copy_images(r'D:\label\AI\cfg\cfg_trans_new2\valid.txt', 'val')
