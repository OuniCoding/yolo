from ultralytics import YOLO
import cv2
import os

img_path = r'e:/logo/trans/2025-01-10/1/resultNG/'
out_path = r'outputs/'
name_path = r'test1/'

img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png') or _.endswith('.bmp'))]
if not os.path.exists(out_path + name_path):
    os.makedirs(out_path + name_path)

out_path = out_path + name_path

# 載入模型
model = YOLO('runs/detect/train/weights/best.pt')

# 類別名稱對應表
class_names = model.names
colors = [(255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247)]

for t, i in enumerate(img_files):
    img = cv2.imread(img_path + i)
    # 圖像推理
    # results = model(img, save=True, project=out_path, name=name_path)
    results = model(img, project=out_path, device=0, conf=0.3, iou=0.4)

    result = results[0]

    # 顯示結果
    # result.show()

    # 儲存檢測後的圖像
    # result.save()

    # 取得邊界框與標示
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]  # 取得類別名稱
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]

        print(f"Class: {cls_name}, Confidence: {conf:.2f}, BBox: {xyxy}")

        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), colors[cls_id], 2)
        cv2.putText(img, "{}-{:.2f}".format(cls_name, float(conf)),
                    (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    colors[cls_id], 2, cv2.LINE_AA)

    cv2.imshow('Inference', img)
    cv2.imwrite(out_path + i, img)

    cv2.waitKey(10)
cv2.destroyAllWindows()
