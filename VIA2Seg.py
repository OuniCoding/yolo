import json
import os
from pathlib import Path
from PIL import Image  # 用於讀取影像尺寸

# 設定類別對應表（VIA 類別名稱 -> YOLO 類別 ID）
class_mapping = {
    "color": 0,
    "safe": 1,
    "broken": 2
}
# 讀取 VIA JSON
with open(r"F:\project\python\shape\via\via_project_14Feb2025_11h28m.json", "r", encoding="utf-8") as f:
    via_data = json.load(f)

# 輸出目錄
output_dir = r"F:\project\python\shape\via\labels-seg"
os.makedirs(output_dir, exist_ok=True)

# 轉換標註
via_img_metadata = via_data['_via_img_metadata']
for img_key, img_data in via_img_metadata.items():
    img_filename = img_data["filename"]
    # img_path = Path("images") / img_filename  # 假設影像在 images 資料夾
    img_path = Path(r"F:\project\python\shape\via") / img_filename  # 假設影像在 images 資料夾
    txt_filename = os.path.splitext(img_filename)[0] + ".txt"
    txt_path = Path(output_dir) / txt_filename

    # 讀取影像尺寸
    if img_path.exists():
        img = Image.open(img_path)
        img_width, img_height = img.size
    else:
        print(f"⚠️ 找不到影像: {img_filename}，請手動提供尺寸！")
        img_width, img_height = 1920, 1080  # 預設尺寸（如有需要，請修改）

    with open(txt_path, "w") as txt_file:
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

                # 轉換為 YOLO Segmentation 格式（歸一化）
                norm_points = []
                for x, y in zip(x_points, y_points):
                    norm_x = x / img_width
                    norm_y = y / img_height
                    norm_points.append(f"{norm_x:.6f} {norm_y:.6f}")

                # 輸出 YOLOv8 Segmentation 格式
                txt_file.write(f"{class_id} " + " ".join(norm_points) + "\n")

print("✅ VIA JSON 轉 YOLOv8 Segmentation 完成！")