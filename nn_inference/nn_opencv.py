import cv2
import time
import numpy as np

img = np.zeros((640, 640, 3), dtype=np.uint8)

# For conversion
# https://github.com/WongKinYiu/yolov7
# python export.py --weights yolov7.pt --grid --simplify --topk-all 300 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
model_weights = 'yolov7.onnx'
net = cv2.dnn.readNet(model_weights)
output_names = net.getUnconnectedOutLayersNames()

blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(640, 640))

net.setInput(blob)
output = net.forward(output_names)

start = time.time()

n_iter = 500
for _ in range(n_iter):
    net.setInput(blob)
    output = net.forward(output_names)
end_time = (time.time() - start) / n_iter * 1000

print(f"Average time: {end_time:.3f} ms")
with open("inference_opencv.txt", "w") as file:
    file.write(f"{end_time:.3f}")