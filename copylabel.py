import os
import shutil

def copy_images(source_list, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    file = open(source_list, 'r')
    for filename in file.readlines():
        if filename[-5:] == '.jpg\n':
            filename = filename.replace('.jpg\n', '.txt')
        elif filename[-5:] == 'jpeg\n':
            filename = filename.replace('.jpeg\n', '.txt')
        elif filename[-5:] == '.png\n':
            filename = filename.replace('.png\n', '.txt')
        elif filename[-5:] == '.bmp\n':
            filename = filename.replace('.bmp\n', '.txt')
        if filename == '':
            continue
        target_file = os.path.join(target_folder, os.path.basename(filename))
        # source_file = os.path.join('yolov8_labels', os.path.basename(filename))
        # shutil.copy2(source_file, target_file)
        print(filename)
        shutil.copy2(filename, target_file)

copy_images(r'D:\label\AI\1TTF\font_data\train.txt', r'labels\font_data_V6\train')
copy_images(r'D:\label\AI\1TTF\font_data\valid.txt', r'labels\font_data_V6\val')
# copy_images(r'D:\label\AI\glass-bottle\cfg\cfg_cam45\valid.txt', r'labels\cam45\val')
