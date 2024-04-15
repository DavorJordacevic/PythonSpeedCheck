import cv2
import time
import onnxruntime
import numpy as np

img = np.zeros((640, 640, 3), dtype=np.uint8)

# For conversion
# https://github.com/WongKinYiu/yolov7
# python export.py --weights yolov7.pt --grid --simplify --topk-all 300 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
model_weights = 'yolov7.onnx'
session = onnxruntime.InferenceSession(model_weights, providers=['CPUExecutionProvider'])
model_inputs = session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
model_outputs = session.get_outputs()
output_names = [model_outputs[i].name for i in range(len(model_outputs))]

blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(640, 640))
outputs = session.run(output_names, {input_names[0]: blob})

start = time.time()

n_iter = 500
for _ in range(n_iter):
    outputs = session.run(output_names, {input_names[0]: blob})
end_time = (time.time() - start) / n_iter * 1000

print(f"Average time: {end_time:.3f} ms")
with open("inference_onnxruntime.txt", "w") as file:
    file.write(f"{end_time:.3f}")