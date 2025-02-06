import os

def convert_yolov3_to_yolov8(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(source_folder, filename), 'r') as file:
                lines = file.readlines()

            with open(os.path.join(target_folder, filename), 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    bbox = parts[1:5]  # (x_center, y_center, width, height)
                    # YOLOv8 標註不需要改動物件檢測部分
                    file.write(f"{class_id} {' '.join(bbox)}\n")

convert_yolov3_to_yolov8('D:\label\AI\Yolos_trans_new2', 'yolov8_labels')
