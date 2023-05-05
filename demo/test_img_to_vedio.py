import os

import cv2

# 设置输出视频的参数
fps = 25
width = 640
height = 480

# 获取图片列表
image_folder = './'
images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')] * 200
images.sort()  # 将图片按顺序排序

# 创建输出视频对象
video_name = 'test2.mp4'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 逐一读取图片并写入输出视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

# 释放输出视频对象和所有窗口
video.release()
cv2.destroyAllWindows()
