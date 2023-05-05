"""
Source: https://github.com/AK391/yolov5/blob/master/utils/gradio/demo.py
"""

import gradio as gr
import torch
from PIL import Image

yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt',
                              # trust_repo=True,
                              # force_reload=True
                              )  # force_reload=True to update


def yolo(im: Image, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.LANCZOS)  # resize
    results = yolov5_model(im)  # inference
    results.render()  # updates results.ims with boxes and labels
    return Image.fromarray(results.ims[0])


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "YOLOv5"
description = "YOLOv5 demo for fire detection. Upload an image or click an example image to use."
article = "See https://github.com/robmarkcole/fire-detection-from-images"
examples = [['pan-fire.jpg'], ['fire-basket.jpg'], ['my-fire.jpg']]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True)


