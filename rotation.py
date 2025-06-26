import cv2
import numpy as np
import matplotlib.pyplot as plt

DEBUG_MODE = False

def set_debug_mode(flag: bool):
    global DEBUG_MODE
    DEBUG_MODE = flag
    if DEBUG_MODE:
        print("✅ [mymodule] DEBUG 模式已啟用")

def group_by_x_axis(points, threshold=10):
    """
    根據 x 值接近程度進行分類分組。
    threshold: 同組內 y 值最大差距（像素）
    回傳：List[List[point]]
    """
    points = sorted(points, key=lambda p: p[0])  # 依 x 值排序
    groups = []

    for pt in points:
        added = False
        for group in groups:
            if abs(group[0][0] - pt[0]) < threshold:
                group.append(pt)
                added = True
                break
        if not added:
            groups.append([pt])

    groups = [item for item in groups if len(item) != 1]

    return groups
def group_by_y_axis(points, threshold=10):
    """
    根據 y 值接近程度進行分類分組。
    threshold: 同組內 y 值最大差距（像素）
    回傳：List[List[point]]
    """
    points = sorted(points, key=lambda p: p[1])  # 依 y 值排序
    groups = []

    for pt in points:
        added = False
        for group in groups:
            if abs(group[0][1] - pt[1]) < threshold:
                group.append(pt)
                added = True
                break
        if not added:
            groups.append([pt])

    groups = [item for item in groups if len(item) != 1]

    return groups
def rotate_image_to_horizontal_text(image, thres):
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # , bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_img = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
    cv2.imshow('bin_img', bin_img)

    # 反轉使文字是白色，背景是黑色（Tesseract & contour 更好處理）
    # bin_img = 255 - bin_img
    # ---------------------------------------------------------------------------------------
    # 高斯平滑 去噪
    Gaussian = cv2.GaussianBlur(bin_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    # 中值濾波
    Median = cv2.medianBlur(Gaussian, 5)
    # cv2.imshow('Median', Median)

    # Sobel運算元 XY方向求梯度 cv2.CV_8U
    x = cv2.Sobel(Median, cv2.CV_32F, 1, 0, ksize=1)  # X方向
    y = cv2.Sobel(Median, cv2.CV_32F, 0, 1, ksize=3)  # Y方向
    # cv2.imshow('X', x)
    # cv2.imshow('Y', y)

    absX = cv2.convertScaleAbs(x)  # 轉回uint8
    absY = cv2.convertScaleAbs(y)  #
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 0.5 0.5; 5.jpg:0.3, 0.6; 1,2,3,6.jpg:0.4, 0.7
    # cv2.imshow('X1', absX)
    # cv2.imshow('Y1', absY)
    # cv2.imshow('Sobel', Sobel)
    # cv2.waitKey(0)

    # 二值化處理 周圍畫素影響
    blurred = cv2.GaussianBlur(Sobel, (3, 3), 0)  # 再進行一次高斯去噪
    # 注意170可以替換的
    ret, Binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    if DEBUG_MODE:
        cv2.imshow('Binary', Binary)

    # 找輪廓
    # contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     print("❌ 無法找到任何文字輪廓")
    #     return image

    contours, _ = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    threshCnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 15:  # 過濾雜訊 if w > 5 and h > 5:  # 過濾雜訊
            centers.append((x + w / 2, y + h / 2))
            threshCnts.append(cnt)
    cur_img = Gaussian.copy()  #Binary.copy()
    cv2.drawContours(cur_img, threshCnts, -1, (255, 255, 255), 1)
    if DEBUG_MODE:
        cv2.imshow('d',cur_img)

    if len(centers) < 2:
        print("❌ 無法找到任何文字輪廓")
        return image

    # 合併所有文字區域，取得最小外接矩形
    # all_points = np.vstack(contours)
    all_points = np.vstack(threshCnts)
    rect = cv2.minAreaRect(all_points)  # ((center_x, center_y), (w, h), angle)
    (center_x, center_y), (w, h), angle = rect  #rect[-1]
    box = cv2.boxPoints(rect)  # 轉換為4個頂點
    # box = np.int0(box)  # 將頂點轉換為整數座標 #np 1.24 以下使用
    box = np.intp(box)
    # 繪製最小包圍矩形
    img = image.copy()
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)  # 綠色框，線寬為2
    cv2.imshow('draw', img)

    # 根據 angle 調整，讓文字水平：
    # if rect[1][0] < rect[1][1]:
    if w < h:
        angle = angle + 90

    print(f"[INFO] 偵測到旋轉角度：{angle:.2f}°，開始旋轉校正")

    # 執行旋轉
    (h, w) = image.shape[:2]
    if angle != 0:
        angle1 = angle-180
    else:
        angle1 = angle
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle1, 1.0)   # angle-180
    # rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated
