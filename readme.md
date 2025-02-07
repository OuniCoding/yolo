# 🚀 YOLOv3 轉 YOLOv8 的完整流程
## 步驟 1：環境準備
1. 安裝 YOLOv8（Ultralytics）

```commandline
 pip install ultralytics
```
2. 確認依賴

    +  Python ≥ 3.8

    + PyTorch ≥ 1.8

    + OpenCV、NumPy 等常見庫

***
## 步驟 2：資料集格式轉換

YOLOv3 和 YOLOv8 都使用 YOLO 格式，但 YOLOv8 的格式更簡潔，並且支援分割（Segmentation）等任務。

### ✅ YOLOv3 格式範例：
```text
    0 0.5 0.5 0.4 0.3
```
+ 0 → 類別 ID

+ 0.5 0.5 → 邊界框中心座標（歸一化）

+ 0.4 0.3 → 邊界框寬度和高度（歸一化）

### 🔄 YOLOv8 格式變化：
+ 基本的物件檢測與 YOLOv3 類似。

+ 若使用分割任務，則會包含多邊形點座標：
    
```text
    0 0.5 0.5 0.4 0.3 0.1 0.2 0.3 0.4 ...
```

### 轉換工具：

使用 Python 轉換標註格式：
```python
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

convert_yolov3_to_yolov8('yolov3_labels', 'yolov8_labels')
```
***

## 步驟 3：建立 YOLOv8 的資料集配置檔
建立一個 data.yaml 文件：
```yaml
train: /path/to/train/images
val: /path/to/val/images

nc: 3  # 類別數量
names: ['class1', 'class2', 'class3']  # 類別名稱
```
***

## 步驟 4：選擇 YOLOv8 模型架構
YOLOv8 提供不同大小的模型：

+ YOLOv8n（Nano） - 最輕量化
+ YOLOv8s（Small） - 適合嵌入式設備
+ YOLOv8m（Medium） - 性能與速度平衡
+ YOLOv8l（Large） - 高精度模型
+ YOLOv8x（Extra Large） - 最高性能

### 🚀 YOLOv8 官方模型下載連結

|模型版本|下載連結|大小| 適用場景 |
|------|-------|----|-----|
|YOLOv8n (Nano)|[下載 YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)|~6 MB|超輕量級，適合嵌入式裝置|
YOLOv8s (Small)|[下載 YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)|~20 MB|快速推論，適合中等設備|
YOLOv8m (Medium)|[下載 YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)|~50 MB|性能與速度平衡|
YOLOv8l (Large)|[下載 YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)|~80 MB|高精度檢測|
YOLOv8x (Xtreme)|[下載 YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)|~130 MB|最佳精度，適合高端硬體|

### ⚡ 使用 Python 自動下載模型

+ 自動下載模型：
```python
from ultralytics import YOLO

# 載入不同的模型
odel_n = YOLO('./model/yolov8n.pt')   # Nano
model_s = YOLO('./model/yolov8s.pt')   # Small
model_m = YOLO('./model/yolov8m.pt')   # Medium
model_l = YOLO('./model/yolov8l.pt')   # Large
model_x = YOLO('./model/yolov8x.pt')   # Xtreme
```
執行後，模型會自動下載至 ~/model/ 目錄。

### 🎯 其他模型格式支援
你也可以將 YOLOv8 模型轉換成其他格式，如 ONNX、TensorRT：
```commandline
yolo export model=yolov8s.pt format=onnx
yolo export model=yolov8s.pt format=engine  # TensorRT
```

***

## 步驟 5：重新訓練模型
使用 YOLOv8 進行訓練：
```commandline
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16
```
+ task=detect → 目標檢測任務
+ model=yolov8s.pt → 選擇模型
+ data=data.yaml → 指定資料集
+ epochs=100 → 訓練輪數
+ imgsz=640 → 圖像尺寸
+ batch=16 → 批次大小

案例
```yaml
path: f:\project\python\YoloV8
train: images\train
val: images\val

nc: 4  # 類別數量
names: ['cap_in', 'safe', 'broken', 'flaw']  # 類別名稱
```
```commandline
yolo task=detect mode=train model=model\yolov8n.pt data=data\cap.yaml epochs=100 imgsz=416 batch=4 device=0
```
```commandline
yolo task=detect mode=train model=model\yolov8n.pt data=data\cap.yaml epochs=100 imgsz=416 batch=4 device=0 lr0=0.001 momentum=0.937 weight_decay=0.0005 patience=0
```
***

## 步驟 6：模型評估與測試
測試模型效果：

```commandline
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```
進行推論：

```commandline
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path_to_images
```

***

## 步驟 7：模型匯出（可選）
將模型導出為 ONNX、TensorRT 等格式：

```commandline
yolo export model=runs/detect/train/weights/best.pt format=onnx
```
***

## ⚡ 優化建議
1. 數據增強：

    + YOLOv8 支援 Mosaic、MixUp、HSV 變化等先進數據增強技術。

2. 超參數調整：

    + 嘗試不同的學習率和批次大小，以獲得更好的結果。

3. FP16 精度訓練：

    + 如果使用 RTX 4070 Ti 或 4060 Ti，可以加速訓練：
```commandline
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16 device=0 --half
```

***

## 🎯 常見問題
+ 標註轉換錯誤？  

  檢查標註座標是否仍保持歸一化（0~1 之間）。

+ 模型表現下降？

  嘗試進行超參數微調，並使用更大或更適合的模型架構（如 YOLOv8m 或 YOLOv8l）。

+ 推論速度慢？

  可以使用 TensorRT 或 ONNX 進行模型加速。

***

## 進階超參數配置的各參數影響說明

 在 YOLOv8 中，進階超參數的配置對模型的訓練效果、準確度、收斂速度等有重要影響。這些超參數主要可分為 資料增強、模型訓練、優化器調整 等幾個類別。

 將從以下幾個面向進行詳細說明：

+ 1️⃣ 訓練相關超參數
+ 2️⃣ 資料增強超參數
+ 3️⃣ 優化器與正則化超參數
+ 4️⃣ 進階技巧與最佳化策略
+ 5️⃣ 範例：完整超參數 YAML 設定

***

### ✅ 1️⃣ 訓練相關超參數

|參數|預設值|說明|
|---|-----|---|
|epochs|100|訓練的總迭代次數。數值越大，模型學習越充分，但訓練時間也更長。|
|batch|16|每個訓練批次的圖片數量。增加此值可加速訓練，但需更大顯示記憶體。|
imgsz|640|輸入圖片的大小。較大尺寸可提高準確率，但會增加計算負擔。|
|device|0|指定使用哪個 GPU，cpu 表示使用 CPU。|
|workers|8|數據加載的工作執行緒數量。增加此值可加速數據預處理。|
|patience|50|控制早停（early stopping）的耐心值。若驗證集沒有改善，則提前終止訓練。|
|resume|False|是否從中斷的地方繼續訓練。|

影響：

+ epochs 和 batch 會直接影響訓練的速度和效果。
+ imgsz 影響模型解析度，推薦根據任務需求進行調整。

***

### ✅ 2️⃣ 資料增強超參數
|參數|預設值|說明|
|---|---|---|
|hsv_h|0.015|調整色相（Hue）的範圍，增加色彩多樣性。|
|hsv_s|0.7|調整飽和度（Saturation）的範圍。|
|hsv_v|0.4|調整亮度（Value）的範圍，模擬不同光照條件。|
|flipud|0.0|上下翻轉圖片的概率，適用於對稱物件。|
|fliplr|0.5|左右翻轉圖片的概率，常用於物件偵測任務。|
|mosaic|1.0|Mosaic 增強技術的概率，將多張圖片拼接，提升模型泛化能力。|
|mixup|0.0|MixUp 增強技術，將兩張圖片融合，增加樣本多樣性。|
|degrees|0.0|隨機旋轉圖片的角度範圍。|
|scale|0.5|隨機縮放的比例範圍。|
|shear|0.0|隨機剪切圖片的範圍。|
|translate|0.1|圖片隨機平移的範圍。|

影響：

+ 資料增強 有助於提高模型的泛化能力，特別是在訓練資料有限時。
+ mosaic 和 mixup 對提升物件偵測任務特別有效。

***

### ✅ 3️⃣ 優化器與正則化超參數

|參數| 預設值    |說明|
|---|--------|---|
|optimizer| SGD    |選擇優化器，常見為 SGD 或 Adam。|
|lr0| 0.01   |初始學習率，影響模型的收斂速度。|
|lrf| 0.01   |最終學習率（learning rate factor），控制學習率衰減。|
|momentum| 0.937  |動量參數，用於加速梯度下降。|
|weight_decay| 0.0005 |權重衰減（正則化），防止過擬合。|
|warmup_epochs| 3.0    |學習率預熱（warm-up）的 epoch 數，避免訓練初期震盪。|
|warmup_momentum| 0.8    |預熱期間使用的動量值。|
|warmup_bias_lr| 0.1    |預熱期間 bias 的學習率。|

影響：

+ lr0 是最重要的超參數之一，學習率過高會導致收斂不穩定，過低則收斂太慢。
+ momentum 有助於加速收斂並減少震盪。

***

### ✅ 4️⃣ 進階技巧與最佳化策略

|參數|說明|
|---|---|
|cache|是否將資料快取到記憶體或硬碟，加速訓練速度。|
|sync_bn|同步 BatchNorm，適用於多 GPU 訓練。|
|opout|丟棄神經元的概率，防止過擬合。|
|amp|啟用自動混合精度（Automatic Mixed Precision），提升訓練效率。|
|rect|是否使用矩形訓練模式，適合偵測不規則尺寸物件。|
|cos_lr|使用餘弦學習率衰減策略，提升收斂效果。|
|label_smoothing|標籤平滑化技術，防止模型過度自信，提升泛化能力。|

***

### ✅ 5️⃣ YAML 超參數設定範例

```yaml
# train.yaml
epochs: 150
batch: 32
imgsz: 640
device: 0
workers: 8
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
patience: 50
amp: True

# Data Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
degrees: 0.0
scale: 0.5
shear: 0.0
translate: 0.1

# Regularization
label_smoothing: 0.0
dropout: 0.0
```

***

### ✅ 6️⃣ 調整超參數的建議

1. 學習率 (lr0) 調整：

   + 初期可設定較高學習率，加速收斂；訓練後期逐漸降低學習率。
   + 搭配 cos_lr 使用餘弦衰減，效果更穩定。

2. 增強泛化能力：

    + 增加 mosaic、mixup 等增強技術。
    + 調整 weight_decay 和 dropout 來防止過擬合。

3. 早停機制（Early Stopping）：

    + 設定 patience，避免模型過度訓練導致性能下降。
   
4. 加速訓練：

    + 使用 amp=True 進行混合精度訓練，減少記憶體佔用，提升速度。
    + 設定 cache=True，加速數據讀取。

***

### ✅ 7️⃣ 常見問題與解決方案

|問題|解決方案|
|--|--|
|模型訓練緩慢|調整 batch、workers，使用 amp=True。|
|過擬合問題|增加 weight_decay，啟用 dropout 或 label_smoothing。|
|驗證集準確率無法提升|調整 lr0、使用 cos_lr，或增加資料增強。|
|記憶體不足|減少 batch，降低 imgsz，或啟用混合精度訓練。|

***

### 🔑 8️⃣ 總結

+ 學習率 是最關鍵的超參數，影響模型收斂速度與穩定性。
+ 資料增強 有助於提升模型的泛化能力，尤其適用於資料不均衡或小樣本場景。
+ 適當的 正則化（如 weight_decay、dropout）可有效防止過擬合。
+ 進行 超參數搜尋（Hyperparameter Tuning）可進一步提升模型性能。

***

## threshold設置

在 YOLOv8 中，閾值（threshold） 主要指 信心閾值（confidence threshold） 
和 非極大值抑制閾值（NMS threshold），這些參數控制模型的預測結果，確保輸出較可靠的偵測結果。

### ✅ 1️⃣ YOLOv8 閾值相關參數

|參數|預設值|作用|
|---|---|---|
|conf|0.25|信心閾值（Confidence Threshold）：低於此值的預測框將被忽略。|
|iou|0.7|非極大值抑制（NMS IoU 閾值）：用於移除重疊的偵測框，確保每個物件只有一個偵測框。|
|agnostic_nms|False|是否啟用類別無關的 NMS（適用於重疊物件較多的場景）。|
|max_det|300|最大偵測數量，控制每張圖片最多輸出多少個目標框。|

***

### ✅ 2️⃣ 設定 conf 與 iou 閾值的方式

#### (1) 直接在推理時設定

當執行推理時，你可以直接設定 conf 和 iou：
```python
from ultralytics import YOLO

# 載入模型
model = YOLO("best.pt")

# 設定信心閾值 0.5，NMS IoU 閾值 0.4
results = model("image.jpg", conf=0.5, iou=0.4)

# 顯示結果
results.show()
```

***

#### （2）設定到 .predict() 方法
```python
results = model.predict("image.jpg", conf=0.6, iou=0.5)
```
+ conf=0.6：只顯示信心值 高於 60% 的預測結果。

+ iou=0.5：NMS IoU 閾值設定為 0.5，減少重疊的框。

***

#### （3）應用於影片或多張圖片
```python
model.train(data="data.yaml", epochs=100, conf=0.3, iou=0.5)
```

+ 這樣在訓練時，模型會根據 conf=0.3 來篩選標註框。

***

#### ✅ 4️⃣ 設定 conf 與 iou 的最佳值

|場景|建議 conf 值|建議 iou 值|
|---|---|---|
|一般物件偵測|0.25 - 0.5|0.5 - 0.7|
|偵測小物件|0.2 - 0.4|0.4 - 0.6|
|過濾假陽性（更嚴格）|0.5 - 0.7|0.3 - 0.5|

+ 如果你發現預測結果過多或誤判嚴重，可以嘗試 提高 conf 或 降低 iou。

***

#### ✅ 5️⃣ 如何測試最佳 threshold？

如果你不確定哪個 conf 值效果最好，可以使用 多次測試不同的 threshold：
```python
for conf_thresh in [0.25, 0.5, 0.75]:
    results = model("image.jpg", conf=conf_thresh, iou=0.5)
    print(f"Conf = {conf_thresh}, Detections: {len(results[0].boxes)}")
```

這樣可以找到最適合你數據集的 conf 閥值。

***

#### ✅ 6️⃣ 總結
conf 決定是否保留偵測框，較低值會產生更多預測，較高值可過濾掉低置信度的結果。
iou 控制非極大值抑制（NMS），較低值減少重疊的框，較高值允許更多重疊物件。
建議根據實際應用場景 測試不同的 conf 和 iou 值，找到最佳設定。

***

## 在 **YOLOv8** 中，`model.export()` 可以將模型匯出為多種格式，以便部署到不同的推理框架或設備上。

---

## ✅ **1️⃣ 可匯出的格式**
以下是 `model.export()` 支援的格式：

| **格式**      | **副檔名**    | **用途** |
|--------------|-------------|---------|
| **PyTorch**   | `.pt`        | YOLOv8 的原生格式，適用於 Python 環境 |
| **TorchScript** | `.torchscript` | 適用於無 PyTorch 環境的 C++/Python 部署 |
| **ONNX**      | `.onnx`      | 適用於 TensorRT、OpenVINO、ONNX Runtime |
| **OpenVINO**  | `.xml` + `.bin` | Intel 硬體加速推理 |
| **TensorRT**  | `.engine`    | NVIDIA GPU 硬體加速（Jetson、RTX 顯卡） |
| **CoreML**    | `.mlmodel`   | Apple 設備（iOS/macOS） |
| **TF SavedModel** | `.pb`    | TensorFlow 格式，適用於 TensorFlow 伺服器 |
| **TF Lite**   | `.tflite`    | 適用於行動設備（Android、Raspberry Pi） |
| **TF Edge TPU** | `.tflite` | Google Coral Edge TPU |
| **FastSAM**   | `.onnx`      | 專門用於 segmentation（YOLOv8-SAM）|

---

## ✅ **2️⃣ 匯出方式**
### **（1）基本匯出**
```python
from ultralytics import YOLO

# 載入訓練好的 YOLOv8 模型
model = YOLO("best.pt")

# 匯出為 ONNX 格式
model.export(format="onnx")
```
這會在 **當前目錄** 產生 `best.onnx`。

---

### **（2）指定匯出路徑**
```python
model.export(format="onnx", path="models/yolov8.onnx")
```
這會將 ONNX 模型存到 `models/yolov8.onnx`。

---

### **（3）匯出為 TensorRT**
```python
model.export(format="engine")  # 需要安裝 TensorRT
```

---

### **（4）匯出為 TensorFlow Lite**
```python
model.export(format="tflite")  # 適用於行動裝置
```

---

### **（5）啟用 FP16（半精度）加速**
某些格式（如 TensorRT、ONNX）可以啟用 **FP16 半精度** 加速：
```python
model.export(format="onnx", half=True)  # 啟用 FP16
```

---

## ✅ **3️⃣ 匯出格式比較**
| **格式**      | **適用場景** | **優勢** |
|--------------|------------|--------|
| **PyTorch (`.pt`)**  | Python 環境 | 訓練與推理 |
| **ONNX (`.onnx`)**   | TensorRT、OpenVINO | 跨平台、推理快 |
| **TensorRT (`.engine`)** | NVIDIA GPU | 推理極快 |
| **TF Lite (`.tflite`)** | 行動裝置 | 輕量化 |
| **OpenVINO (`.xml`)** | Intel 硬體 | 最佳化推理 |
| **CoreML (`.mlmodel`)** | Apple 設備 | iOS/macOS |
| **TorchScript (`.torchscript`)** | 無 PyTorch 環境 | C++/Python |

如果你要 **在 PC 上高效推理**，建議使用 **ONNX 或 TensorRT**。  
如果你要 **在行動裝置運行**，建議使用 **TFLite 或 CoreML**。🚀