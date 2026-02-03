# -- coding: utf-8
'''
 yolo task=detect mode=train model=model/yolov8s.pt data=data/cap.yaml epochs=8000 imgsz=416 batch=64 device=0 lr0=0.001 momentum=0.937 weight_decay=0.0005 patience=0
 yolo task=detect mode=val model=runs/detect/train6/weights/best.pt data=data/cap.yaml
'''
from ultralytics import YOLO
import cv2
import os
import time
import datetime
import torch
import rotation1024 as ro
import argparse
import numpy as np

# 處理參數與環境變數
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

DEBUG_MODE = args.debug
# 通知模組進入 debug 模式
ro.set_debug_mode(DEBUG_MODE)
def release_model(model):   # release_model(model)
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# img_path = r'F:\\project\\味全貝納頌\\OCR_Code\\sample\\' # thres =50 # r'e:/logo/trans/2025-01-10/1/resultNG/'
# img_path = r'D:\\label\\AI\\1TTF\\OCR DATA\\OCR DATA\\black\\' # thres =126, thres_m=38
img_path = r'D:\\logo\\la\\' # 166,65
# img_path = r'D:\\logo\\tla\\2025-07-15\\1\\' # 166,52
#img_path = r'D:\\logo\\ma\\1\\' # 148,38
img_path = r'D:\\logo\\cma\\2025-07-18\\1\\resultG\\' # 140,38
img_path = r"D:\\logo\\特濃黑咖啡\\2025-07-15\\1\\resultG\\" #126,35
#img_path  = r'D:\\logo\\經典曼特寧\\2025-07-24\\1\\resultNG\\' #126,35
#img_path = r'D:\\logo\\經典拿鐵\\2025-07-28\\1\\resultG\\' #180,96
# img_path = r'D:\\label\\AI\\1TTF\\OCR DATA\\OCR DATA\\pink\\' # thres =126, thres_m=38
# img_path = r'D:\\label\\AI\\1TTF\\OCR DATA\\OCR DATA\\blue\\' # thres =128, thres_m=38
# img_path = r'd:\\label\\AI\\1TTF\\OCR DATA\\OCR DATA\\gold\\' # thres =229 ?? thres_m = 120
# img_path = r'E:\\AI\\1TTF\\OCR DATA\\OCR DATA\\pink\\' # thres =230 ?? thres_m = 120
# img_path = r'E:\\AI\\1TTF\\OCR DATA\\OCR DATA\\purple\\' # thres =128, thres_m=40
# img_path = r'D:\\logo\\經典曼特寧\\2025-06-02\\1\\resultNG\\' #經典曼特寧
now = datetime.datetime.now()
new = datetime.datetime.now()+ datetime.timedelta(days=15)
now_str = now.strftime('%Y%m%d')
new_str = new.strftime('%Y%m%d')
print(f'今日日期:{now_str}, 目標日期:{new_str}')

# 載入模型
# model = YOLO(r'F:\project\ouni_mutil_cam\cfg\cam2\train\weights\best.pt')  #('runs/detect/train/weights/best.pt')
# model = YOLO(r'F:\project\python\YoloV8\runs\detect\cam1_train\weights\best.pt')
model = YOLO(r'.\runs\detect\ocr_V9m\train\weights\best.pt')
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
# 預先載入
frame = cv2.imread('black512.jpg')
predict = model(frame)
# 讀取檢測圖檔
out_path = r'outputs/'
name_path = r'test2/'

img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png') or _.endswith('.bmp'))]
if not os.path.exists(out_path + name_path):
    os.makedirs(out_path + name_path)

out_path = out_path + name_path
index = 0
print(f'Files = {len(img_files)}')
while True:
    name = img_files[index]
# for t, name in enumerate(img_files):
    print('filename:', img_path+name)
    thres = 134 #180 #163  133
    thres_m = 35 #96    #43  28
    checkRaduis = 55
    if name == '4.jpg' or name == '901jpg.jpg':
        thres = 103
    elif name == '5.jpg':
        thres = 106
    elif name == '6.jpg':
        thres = 122
    elif name == '1803.jpg' or name == '3.jpg':
        thres = 25
    elif name == '2702.jpg' or name == '901.jpg' or name == '2.jpg':
        thres = 45

    begin_time = time.time()
    img = cv2.imread(img_path + name)
    print(img.shape)
    # 校正圖像, 文字轉向水平排列
    # rotated = ro.rotate_image_to_horizontal_text(img, thres)
    r_box, rotated, r_bin = ro.rotate_image_to_horizontal_text(img, thres)
    # cv2.waitKey(0)

    # 遮罩檢測區
    maskimg = rotated.copy()    #cv2.resize(rotated, (512, 512), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(maskimg, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, thres_m, 255, cv2.THRESH_BINARY)
    cv2.imshow('start', bin_img)
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
    else:
        print('⚠️ 未發現物件 ⚠️')
        continue
    center = (int(x), int(y))
    radius = int(radius)
    print(f'半徑pixels={radius}')
    #black = cv2.imread('black512.jpg')
    black = cv2.imread('black.jpg')
    black_b = black.copy()  #cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    cv2.circle(black_b, center, radius-checkRaduis, (255, 255, 255), -1)
    ret, black_b = cv2.threshold(black_b, 1, 255, cv2.THRESH_BINARY)
    if DEBUG_MODE:
        cv2.imshow('Mask', black_b)
    # image = cv2.bitwise_and(maskimg, black_b)
    rotated_temp = rotated.copy()
    rotated = cv2.bitwise_and(maskimg, black_b)
    if DEBUG_MODE:
        # print(r_box)
        if r_box.size > 0:
            maskR_img = cv2.drawContours(rotated, [r_box], 0, (0, 255, 0), 2)  # 綠色框，線寬為2
            cv2.imshow('Mask_R', maskR_img)

    # 文字是否上下顛倒
    h, w = rotated.shape[:2]
    circle_center = center      #(w // 2, h // 2)
    circle_radius = radius-checkRaduis   #min(w, h) // 2 - 100  # 根據紅圈估個近似值

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
    if DEBUG_MODE:
        cv2.imshow('R', thresh)

    # --------------------------------------------------------------------------------------------------------------------
    # 高斯平滑 去噪
    #Gaussian = cv2.GaussianBlur(thresh, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    Gaussian = cv2.GaussianBlur(thresh, (9, 9), 0, 0, cv2.BORDER_DEFAULT)

    contours, _ = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    threshCnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 512 if w > 15 and h > 15:  # 過濾雜訊 if w > 5 and h > 5:  # 過濾雜訊
        if w > 30 and h > 30:  # 過濾雜訊 if w > 5 and h > 5:  # 過濾雜訊
            centers.append((x + w / 2, y + h / 2))
            threshCnts.append(cnt)

    groups = ro.group_by_y_axis(centers, threshold=50)
    for i, group in enumerate(groups):
        print(f"第 {i + 1} 組：{group}")

    if len(groups) > 1:
        if len(groups[1]) > len(groups[0]):
            # 執行旋轉
            (h, w) = rotated.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 180, 1.0)
            rotated = cv2.warpAffine(rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            r_bin = cv2.warpAffine(r_bin, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rotated_temp = cv2.warpAffine(rotated_temp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            if DEBUG_MODE:
                cv2.imshow('R1', rotated)
            circle_center = (512 - circle_center[0], 512 - circle_center[1])
    # --------------------------------------------------------------------------------------------------------------------

    cv2.imshow("Corrected", rotated)
    corrected = rotated
    # img = corrected.copy()

    # 圖像推理
    # results = model(img, save=True, project=out_path, name=name_path)
    results = model(corrected, project=out_path, device='cuda:0', conf=0.1, iou=0.4, agnostic_nms=True) # conf=0.6, iou=0.5, agnostic_nms=True
    #r_bin = cv2.cvtColor(r_bin, cv2.COLOR_GRAY2BGR)
    #results = model(r_bin, project=out_path, device='cuda:0', conf=0.05, iou=0.4,
    #                agnostic_nms=True)  # conf=0.6, iou=0.5, agnostic_nms=True

    result = results[0]

    # 顯示結果
    # result.show()

    # 儲存檢測後的圖像
    # result.save()

    # 取得邊界框與標示
    # l = len(result.boxes)
    a = []
    big_rect_pts = np.array(r_box)#[[r_box[0][0], r_box[0][1]], [r_box[1][0], r_box[1][1]], , [x3, y3]])
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]  # 取得類別名稱
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]

        print(f"Class: {cls_name}, Confidence: {conf:.2f}, BBox: {xyxy}")

        if r_box.size > 0:
            if abs(int(xyxy[0])-int(xyxy[2])) >= 10 and abs(int(xyxy[1])-int(xyxy[3])) >= 37:   # 過濾過小的辨識結果
                # 設置檢查範圍，篩選剃除舉行框外字元
                min_xyxy_x, max_xyxy_x = min(xyxy[0], xyxy[2]), max(xyxy[0], xyxy[2])
                min_xyxy_y, max_xyxy_y = min(xyxy[1], xyxy[3]), max(xyxy[1], xyxy[3])
                xyxy_rect_pts = [
                    (min_xyxy_x, min_xyxy_y),  # 左上
                    (max_xyxy_x, min_xyxy_y),  # 右上
                    (max_xyxy_x, max_xyxy_y),  # 右下
                    (min_xyxy_x, max_xyxy_y),  # 左下
                    ]
                in_range_flag = True
                for pt in xyxy_rect_pts:
                    in_range = cv2.pointPolygonTest(big_rect_pts.astype(np.float32), pt, measureDist=True)
                    if in_range <  -20: #-6 for 512 # -1e-6:    #0:   # 1e-6    #
                        in_range_flag = False
                # 設置檢查範圍，篩選剃除舉行框外字元
                if in_range_flag:
                    cv2.rectangle(rotated_temp, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), colors[cls_id], 2)
                    cv2.putText(rotated_temp, "{}".format(cls_name),
                            (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            colors[cls_id], 2, cv2.LINE_AA)
                    a.append([cls_name, xyxy])
        # if l == 16:
        #     a.append([cls_name, xyxy])
    l = len(a)
    if l == 16:
        a = sorted(a, key=lambda x: x[1][0])
        # 找出全體 box 的極值
        x1_min = min([box[1][0] for box in a])
        y1_min = min([box[1][1] for box in a])
        x2_max = max([box[1][2] for box in a])
        y2_max = max([box[1][3] for box in a])

        # print(f"最小 x1: {x1_min}")
        # print(f"最小 y1: {y1_min}")
        # print(f"最大 x2: {x2_max}")
        # print(f"最大 y2: {y2_max}")
        # ----------- 分行參數設定 -------------
        line_threshold = 25  # 控制「上下差多少像素」就算同一行，可依影像實際高度微調

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
        if new_str == text_lines[0][0:8]:
            print('✅ 日期正確')
        else:
            print('❌ 日期錯誤')
        print(f'✅ 日期={text_lines[0][0:8]}')
        print(f'✅ 批號={text_lines[1][3:7]}')

    else:
        print(f"❌ 字數有缺 {16-l}")

    cv2.imshow('Inference', rotated_temp)
    cv2.imwrite(out_path + 'ocr' + name, rotated_temp)
    print(f'第 {index + 1} 張圖片 工作時間={time.time() - begin_time}s \n')

    # key = cv2.waitKey(0) & 0xFF
    key = cv2.waitKeyEx(0)
    if key & 0xff == ord('q') or key & 0xff == ord('Q'):
        break
    elif key == 2490368 or key == 2424832:
        index -= 1
        if index < 0:
            index = len(img_files)-1
    elif key == 2621440 or key == 2555904:
        index += 1
        if index >= len(img_files):
            index = 0
cv2.destroyAllWindows()
release_model(model)
