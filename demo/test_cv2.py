from typing import Tuple

import cv2
import numpy as np

# 加载视频
cap: cv2.VideoCapture = cv2.VideoCapture('test2.mp4')

# 定义矩形框的坐标和颜色
x1: int = 100
y1: int = 100
x2: int = 300
y2: int = 300
color: Tuple[int, int, int] = (0, 255, 0)

# 循环遍历视频中的每一帧
while cap.isOpened():
    # 读取一帧
    ret: bool
    frame: np.ndarray
    ret, frame = cap.read()
    if not ret:
        break

    # 在指定区域画矩形框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 显示处理后的帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
