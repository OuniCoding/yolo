### **VIA 多邊形（Polygon）標註轉換為 YOLOv8 格式**
在 VIA 標註 JSON 中，多邊形標註的 `shape_attributes` 通常包含 `all_points_x` 和 `all_points_y`，這些座標需要轉換為 YOLOv8 的格式。YOLOv8 支援**多邊形標註（Segmentation 模式）**和**邊界框標註（Bounding Box 模式）**：

- **Bounding Box（BBOX）格式**：計算多邊形的最小外接矩形（Bounding Box）。
- **Segmentation（Seg）格式**：直接使用多邊形的頂點（YOLOv8 支援 segmentation）。

---

## **1. 讀取 VIA JSON 並解析多邊形標註**
VIA JSON 多邊形標註範例如下：
```json
{
    "image1.jpg123456": {
        "filename": "image1.jpg",
        "size": 123456,
        "regions": [
            {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [100, 200, 250, 150],
                    "all_points_y": [150, 100, 200, 250]
                },
                "region_attributes": {
                    "class": "dog"
                }
            }
        ]
    }
}
```

---

## **2. 轉換為 YOLOv8 格式**
### **A. 轉換為 Bounding Box 格式**
這種方式計算多邊形的最小外接矩形（Bounding Box），並輸出標註為：
```
<class_id> <x_center> <y_center> <width> <height>
```

**Python 轉換程式（BBOX 格式）：**
```python
import json
import os

# 設定類別對應表（VIA 類別名稱 -> YOLO 類別 ID）
class_mapping = {
    "dog": 0,
    "cat": 1,
    "person": 2
}

# 讀取 VIA JSON
with open("via_annotations.json", "r", encoding="utf-8") as f:
    via_data = json.load(f)

# 轉換標註
for img_key, img_data in via_data.items():
    img_filename = img_data["filename"]
    img_width = 1920  # 可改成動態獲取
    img_height = 1080

    txt_filename = os.path.splitext(img_filename)[0] + ".txt"

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
```

### **輸出範例 (`image1.txt`)**
```text 
0 0.182292 0.162037 0.078125 0.231481
```
這表示：
- **類別 ID 為 `0`**（對應 `dog`）。
- **標註框的中心點 (`x_center, y_center`) 和寬高 (`width, height`) 均已歸一化**。

---

## **B. 轉換為 YOLOv8 Segmentation 格式**
如果要保留多邊形標註，YOLOv8 支援 Segmentation 標註格式：
```text
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```
其中 `(x1, y1) (x2, y2) ... (xn, yn)` 為歸一化的多邊形頂點。

**Python 轉換程式（Segmentation 格式）：**
```python
import json
import os

# 設定類別對應表
class_mapping = {
    "dog": 0,
    "cat": 1,
    "person": 2
}

# 讀取 VIA JSON
with open("via_annotations.json", "r", encoding="utf-8") as f:
    via_data = json.load(f)

# 轉換標註
for img_key, img_data in via_data.items():
    img_filename = img_data["filename"]
    img_width = 1920  # 需根據實際影像尺寸調整
    img_height = 1080

    txt_filename = os.path.splitext(img_filename)[0] + ".txt"

    with open(txt_filename, "w") as txt_file:
        for region in img_data["regions"]:
            shape = region["shape_attributes"]
            label = region["region_attributes"]["class"]

            if label not in class_mapping:
                continue

            class_id = class_mapping[label]

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

print("Segmentation 轉換完成！")
```

### **Segmentation 輸出範例 (`image1.txt`)**
```text
0 0.052083 0.138889 0.104167 0.092593 0.130208 0.185185 0.078125 0.231481
```
- `0` 是類別 ID。
- 後面的數字是 `(x1, y1) (x2, y2) ...`，表示歸一化的多邊形座標。

---

## **3. 選擇哪種格式？**
| 格式 | 適用情境 | 優勢 | 缺點 |
|------|----------|------|------|
| **Bounding Box (BBOX)** | 適用於物體較為矩形的標註，如目標檢測 | YOLOv8 兼容性好，適用於 `detect` 模型 | 可能不適用於不規則形狀 |
| **Segmentation (Seg)** | 物體形狀較不規則時，如人體、車輛輪廓 | 精確度更高，適用於 `segment` 模型 | 訓練需求較高 |

如果是訓練物件偵測（Object Detection），請選擇 **Bounding Box**。  
如果是語意分割（Instance Segmentation），請選擇 **Segmentation**。

---

### **VIA 多邊形（Polygon）標註轉換為 YOLOv8 TXT 格式**
VIA（VGG Image Annotator）標註的多邊形格式需要轉換為 YOLOv8 的 Segmentation 標註格式。YOLOv8 支援**多邊形（Segmentation）** 和 **矩形（Bounding Box）** 兩種格式，這裡主要處理 **Segmentation 格式**。

---

## **1. VIA JSON 多邊形格式範例**
VIA JSON 內的 **polygon（多邊形）** 標註通常如下：
```json
{
    "image1.jpg123456": {
        "filename": "image1.jpg",
        "size": 123456,
        "regions": [
            {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [100, 200, 250, 150],
                    "all_points_y": [150, 100, 200, 250]
                },
                "region_attributes": {
                    "class": "dog"
                }
            }
        ]
    }
}
```
---

## **2. YOLOv8 Segmentation 格式**
YOLOv8 的 **Segmentation 格式**：
```text
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```
- `class_id`：類別編號（0, 1, 2, ...）
- `(x1, y1), (x2, y2), ... (xn, yn)`：多邊形頂點的座標，**歸一化（除以影像寬高）**

---

## **3. Python 轉換程式**
### **🔹 轉換多邊形到 YOLOv8 TXT**
```python
import json
import os
from pathlib import Path
from PIL import Image  # 用於讀取影像尺寸

# 設定類別對應表（VIA 類別名稱 -> YOLO 類別 ID）
class_mapping = {
    "dog": 0,
    "cat": 1,
    "person": 2
}

# 讀取 VIA JSON
with open("via_annotations.json", "r", encoding="utf-8") as f:
    via_data = json.load(f)

# 輸出目錄
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

# 轉換標註
for img_key, img_data in via_data.items():
    img_filename = img_data["filename"]
    img_path = Path("images") / img_filename  # 假設影像在 images 資料夾
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
```

---

## **4. 產生的 YOLOv8 TXT 檔案**
對應的 `image1.txt` 內容如下：
```
0 0.052083 0.138889 0.104167 0.092593 0.130208 0.185185 0.078125 0.231481
```
- **`0`**：類別 ID（對應 `dog`）。
- **多邊形點座標**：`x1 y1 x2 y2 ...`，所有座標已**歸一化**（相對於影像尺寸的比例）。

---

## **5. 說明**
✅ **完整支援 VIA 多邊形格式**  
✅ **自動讀取影像尺寸**（若找不到影像，則使用預設大小 `1920x1080`）  
✅ **產出標註存放於 `labels` 目錄**（YOLOv8 標準格式）  
✅ **適用於 YOLOv8 `segment` 模型**（`yolov8n-seg.pt`）  

---

## **6. 轉換 Bounding Box**
如果你的 YOLOv8 模型是 **`detect`（物件偵測）**，你需要將 **Polygon 轉換為 Bounding Box**。請使用這段程式：
```python
if shape["name"] == "polygon":
    x_points = shape["all_points_x"]
    y_points = shape["all_points_y"]

    # 計算 Bounding Box
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
```
- **這會將多邊形轉換為最小外接矩形（Bounding Box）。**
- **適用於 YOLOv8 `detect` 模型（物件偵測）。**

---

## **7. 總結**
| 格式 | 適用模型 | 格式說明 |
|------|---------|---------|
| **Segmentation（多邊形）** | `yolov8-seg.pt` | 直接使用多邊形座標 |
| **Bounding Box（外接矩形）** | `yolov8n.pt` | 轉換為矩形框 |

### **你該選哪個？**
- 目標檢測（Object Detection） → **Bounding Box 格式**
- 目標分割（Instance Segmentation） → **Segmentation 格式**

---
