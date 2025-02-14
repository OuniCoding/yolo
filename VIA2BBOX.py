import json
import os

# 設定類別對應表（VIA 類別名稱 -> YOLO 類別 ID）
class_mapping = {
    "color": 0,
    "safe": 1,
    "broken": 2
}

# 讀取 VIA JSON
with open(r"via\via_project_14Feb2025_11h28m.json", "r", encoding="utf-8") as f:
    via_data = json.load(f)

# 轉換標註
via_img_metadata = via_data['_via_img_metadata']
for img_key, img_data in via_img_metadata.items():
    img_filename = img_data["filename"]
    img_width = 416 #1920  # 可改成動態獲取
    img_height = 416    #1080

    txt_filename = 'via\\' + os.path.splitext(img_filename)[0] + ".txt"

    with open(txt_filename, "w") as txt_file:
        for region in img_data["regions"]:
            shape = region["shape_attributes"]
            label = region["region_attributes"]["class"]

            if label not in class_mapping:
                continue

            class_id = class_mapping[label]

            # 檢查是否為多邊形
            if shape["name"] == "polygon":
                x_points = shape["all_points_x"]
                y_points = shape["all_points_y"]

                # 計算最小外接矩形 (Bounding Box)
                x_min, x_max = min(x_points), max(x_points)
                y_min, y_max = min(y_points), max(y_points)
                w = x_max - x_min
                h = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # 轉換為 YOLO 格式（歸一化）
                x_center /= img_width
                y_center /= img_height
                norm_w = w / img_width
                norm_h = h / img_height

                # 寫入 YOLOv8 格式 (Bounding Box)
                txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

print("轉換完成！")