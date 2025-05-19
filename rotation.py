import cv2
import numpy as np
import matplotlib.pyplot as plt

def group_by_x_axis(points, threshold=10):
    """
    根據 x 值接近程度進行分類分組。
    threshold: 同組內 y 值最大差距（像素）
    回傳：List[List[point]]
    """
    points = sorted(points, key=lambda p: p[0])  # 依 y 值排序
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

    return groups
def get_orientation_angle_from_boxes(centers):
    """
    使用主成分分析 (PCA) 推算中心點主要排列方向角度
    """
    data = np.array(centers)
    data = data - np.mean(data, axis=0)
    cov = np.cov(data.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(main_axis[1], main_axis[0]) * 180 / np.pi
    return angle

def determine_rotation_angle(angle):
    """
    根據主軸角度回推應該旋轉多少度使文字橫向排列
    """
    if not (-20 < angle < 20):
        angle = 180 + angle
    return angle

    angle = angle % 360
    if -45 <= angle <= 45 or 315 <= angle <= 360:
        return 0
    elif 45 < angle <= 135:
        return -90
    elif 135 < angle <= 225:
        return -180
    else:
        return -270

def correct_rotation_by_layout(image, thres):
    """
    偵測文字排列方向並自動旋轉
    """
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
    #_, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #binarized = 255 - binarized  # 反轉使文字為白
    binarized = binary  #gray

    # 找輪廓
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    threshCnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 4 and h > 4:  # 過濾雜訊 if w > 5 and h > 5:  # 過濾雜訊
            centers.append((x + w / 2, y + h / 2))
            threshCnts.append(cnt)
    cur_img = binarized.copy()
    cv2.drawContours(cur_img, threshCnts, -1, (255, 255, 255), 3)
    cv2.imshow('d',cur_img)

    if len(centers) < 2:
        print("未找到足夠的文字區塊")
        return image, 0

    # 推估主方向角度
    layout_angle = get_orientation_angle_from_boxes(centers)
    rotate_angle = determine_rotation_angle(layout_angle)

    if -20 < rotate_angle < 20:   #rotate_angle == 0:
        groups = group_by_y_axis(centers, threshold=50)
        for i, group in enumerate(groups):
            print(f"第 {i + 1} 組：{group}")

        if len(groups) > 1:
            if len(groups[1]) > len(groups[0]):
                rotate_angle = 180

    if 75 < rotate_angle < 105:
        groups = group_by_x_axis(centers, threshold=39)
        for i, group in enumerate(groups):
            print(f"第 {i + 1} 組：{group}")

        if len(groups) > 1:
            if len(groups[0]) > len(groups[1]):
                rotate_angle = 180 + rotate_angle

    # 旋轉圖像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))

    return rotated, rotate_angle

# 測試程式
if __name__ == "__main__":
    path = "1.jpg.jpg"  # 或換成 "1802jpg.jpg"
    thres = 50
    if path == '4.jpg' or path == '901jpg.jpg':
        thres = 103
    elif path == '5.jpg':
        thres = 106
    elif path == '6.jpg':
        thres = 122
    elif path == '1803.jpg' or path == '3.jpg':
        thres = 25
    elif path == '2702.jpg' or path == '901.jpg' or path == '2.jpg':
        thres = 45

    img = cv2.imread(path)
    corrected, angle = correct_rotation_by_layout(img, thres)

    print(f"建議旋轉角度：{angle}°")
    cv2.imshow("Corrected", corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
