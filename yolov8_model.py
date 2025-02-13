from ultralytics import YOLO

# 載入不同的模型
# Bounding Box 邊界
model_n = YOLO('./model/yolov8n.pt')   # Nano
model_s = YOLO('./model/yolov8s.pt')   # Small
model_m = YOLO('./model/yolov8m.pt')   # Medium
model_l = YOLO('./model/yolov8l.pt')   # Large
model_x = YOLO('./model/yolov8x.pt')   # Xtreme

# Instance Segmentation 多邊形
model_n = YOLO('./model/yolov8n-seg.pt')   # Nano
model_s = YOLO('./model/yolov8s-seg.pt')   # Small
model_m = YOLO('./model/yolov8m-seg.pt')   # Medium
model_l = YOLO('./model/yolov8l-seg.pt')   # Large
model_x = YOLO('./model/yolov8x-seg.pt')   # Xtreme