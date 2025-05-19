# -- coding: utf-8
'''
 yolo task=detect mode=train model=model/yolov8s.pt data=data/cap.yaml epochs=8000 imgsz=416 batch=64 device=0 lr0=0.001 momentum=0.937 weight_decay=0.0005 patience=0
 yolo task=detect mode=val model=runs/detect/train6/weights/best.pt data=data/cap.yaml
'''
from ultralytics import YOLO
import cv2
import os
import time

import rotation as ro

def release_model(model):   # release_model(model)
    import torch, gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

img_path = r'F:\\project\\味全貝納頌\\OCR_Code\\sample\\' # r'e:/logo/trans/2025-01-10/1/resultNG/'
out_path = r'outputs/'
name_path = r'test1/'

img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png') or _.endswith('.bmp'))]
if not os.path.exists(out_path + name_path):
    os.makedirs(out_path + name_path)

out_path = out_path + name_path

# 載入模型
# model = YOLO(r'F:\project\ouni_mutil_cam\cfg\cam2\train\weights\best.pt')  #('runs/detect/train/weights/best.pt')
# model = YOLO(r'F:\project\python\YoloV8\runs\detect\cam1_train\weights\best.pt')
model = YOLO(r'E:\AI\1TTF\cfg\train\weights\best.pt')
# 類別名稱對應表
class_names = model.names
colors = [(255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247),
          (255, 41, 5),
          (233, 219, 13),
          (7, 94, 255),
          (15, 255, 247)
          ]

for t, i in enumerate(img_files):
    print('filename:', i)
    thres = 50
    if i == '4.jpg' or i == '901jpg.jpg':
        thres = 103
    elif i == '5.jpg':
        thres = 106
    elif i == '6.jpg':
        thres = 122
    elif i == '1803.jpg' or i == '3.jpg':
        thres = 25
    elif i == '2702.jpg' or i == '901.jpg' or i == '2.jpg':
        thres = 45

    begin_time = time.time()
    img = cv2.imread(img_path + i)
    # 校正圖像
    corrected, angle = ro.correct_rotation_by_layout(img, thres)
    print(f"建議旋轉角度：{angle}°")
    # corrected = cv2.resize(corrected, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Corrected", corrected)
    img = corrected.copy()

    # 圖像推理
    # results = model(img, save=True, project=out_path, name=name_path)
    results = model(img, project=out_path, device='cuda:0', conf=0.3, iou=0.4)

    result = results[0]

    # 顯示結果
    # result.show()

    # 儲存檢測後的圖像
    # result.save()

    # 取得邊界框與標示
    l = len(result.boxes)
    a = []
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]  # 取得類別名稱
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]

        print(f"Class: {cls_name}, Confidence: {conf:.2f}, BBox: {xyxy}")

        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), colors[cls_id], 2)
        #cv2.putText(img, "{}-{:.2f}".format(cls_name, float(conf)),
        #            (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #            colors[cls_id], 2, cv2.LINE_AA)
        cv2.putText(img, "{}".format(cls_name),
                    (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    colors[cls_id], 2, cv2.LINE_AA)

        if l == 16:
            a.append([cls_name, xyxy])
    if l == 16:
        a = sorted(a, key=lambda x: x[1][0])
        # 找出全體 box 的極值
        x1_min = min([box[1][0] for box in a])
        y1_min = min([box[1][1] for box in a])
        x2_max = max([box[1][2] for box in a])
        y2_max = max([box[1][3] for box in a])

        print(f"最小 x1: {x1_min}")
        print(f"最小 y1: {y1_min}")
        print(f"最大 x2: {x2_max}")
        print(f"最大 y2: {y2_max}")
        # ----------- 分行參數設定 -------------
        line_threshold = 100  # 控制「上下差多少像素」就算同一行，可依影像實際高度微調

        # ----------- 依 y1 排序，初步分行 -------------
        data_sorted_by_y = sorted(a, key=lambda x: x[1][1])  # y1 排序
        lines = []  # 每一行是一個 list

        for item in data_sorted_by_y:
            char, box = item
            y1 = box[1]
            matched = False
            for line in lines:
                line_y = line[0][1][1]  # 該行第一個字的 y1
                if abs(y1 - line_y) < line_threshold:
                    line.append(item)
                    matched = True
                    break
            if not matched:
                lines.append([item])  # 新增新的一行

        # ----------- 每行內部以 x1 排序 -------------
        for line in lines:
            line.sort(key=lambda x: x[1][0])

        # ----------- 合併文字 -------------
        text_lines = []
        for line in lines:
            text = ''.join([ch[0] for ch in line])
            text_lines.append(text)

        # ----------- 輸出結果 -------------
        for j, line_text in enumerate(text_lines):
            print(f"第 {j + 1} 行：{line_text}")

    cv2.imshow('Inference', img)
    cv2.imwrite(out_path + 'ocr' + i, img)
    print(f'工作時間={time.time() - begin_time}s \n')

    cv2.waitKey(1)
cv2.destroyAllWindows()
release_model(model)
