import torch
from PIL import Image

image = Image.open('my-fire.jpg')

# Show basic information about the image
print(image.format)  # e.g. JPEG, PNG, etc.
print(image.size)  # e.g. (1920, 1080) for a 1920x1080 pixel image
print(image.mode)  # e.g. RGB, L, etc.

# Display the image
# image.show()


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


# image = yolo(image)
# image.show()


results0 = yolov5_model(image)
results0.render()
Image.fromarray(results0.ims[0]).show()
