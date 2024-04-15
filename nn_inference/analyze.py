import os
import matplotlib.pyplot as plt

os.system("python nn_onnxruntime.py")
os.system("python nn_opencv.py")

with open("inference_opencv.txt") as file:
    opencv_time = float(file.read())

with open("inference_onnxruntime.txt") as file:
    onnxruntime_time = float(file.read())

plt.figure(figsize=(10, 5))
plt.bar(["OpenCV", "ONNXRuntime"], [opencv_time, onnxruntime_time])
for i, v in enumerate([opencv_time, onnxruntime_time]):
    plt.text(i, v, f"{v:.2f}", ha="center")
plt.ylabel("Time (ms)")
plt.ylabel("Method")
plt.title("Average inference speed comparison")
plt.savefig("avg_inference_speed_comparison.png")

os.remove("inference_opencv.txt")
os.remove("inference_onnxruntime.txt")

print('Results saved in "avg_inference_speed_comparison.png')