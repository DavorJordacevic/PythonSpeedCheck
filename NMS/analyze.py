import os
import matplotlib.pyplot as plt

os.system("python nms_numpy.py")
os.system("python nms_opencv.py")
os.system("python nms_torch.py")
os.system("python nms_numba.py")

with open("nms_numpy.txt") as file:
    numpy_time = float(file.read())

with open("nms_opencv.txt") as file:
    opencv_time = float(file.read())

with open("nms_torch.txt") as file:
    torch_time = float(file.read())

with open("nms_numba.txt") as file:
    numba_time = float(file.read())

plt.figure(figsize=(10, 5))
plt.bar(["Numpy", "OpenCV", "Torchvision", "Numpy+Numba"], [numpy_time, opencv_time, torch_time, numba_time])
for i, v in enumerate([numpy_time, opencv_time, torch_time, numba_time]):
    plt.text(i, v, f"{v:.2f}", ha="center")
plt.ylabel("Time (ms)")
plt.ylabel("Method")
plt.title("NMS speed comparison for 1000 iterations")
plt.savefig("nms_speed_comparison.png")

os.remove("nms_numpy.txt")
os.remove("nms_opencv.txt")
os.remove("nms_torch.txt")
os.remove("nms_numba.txt")

print('Results saved in "nms_speed_comparison.png')