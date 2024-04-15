import os
import matplotlib.pyplot as plt

os.system("python he_numpy.py")
os.system("python he_opencv.py")
os.system("python he_numba.py")

with open("he_numpy.txt") as file:
    numpy_time = float(file.read())

with open("he_opencv.txt") as file:
    opencv_time = float(file.read())

with open("he_numba.txt") as file:
    numba_time = float(file.read())

plt.figure(figsize=(10, 5))
plt.bar(["Numpy", "OpenCV", "Numpy+Numba"], [numpy_time, opencv_time, numba_time])
for i, v in enumerate([numpy_time, opencv_time, numba_time]):
    plt.text(i, v, f"{v:.2f}", ha="center")
plt.ylabel("Time (ms)")
plt.ylabel("Method")
plt.title("Histogram equalization speed comparison for 100 iterations")
plt.savefig("histogram_equalization_speed_comparison.png")

os.remove("he_numpy.txt")
os.remove("he_opencv.txt")
os.remove("he_numba.txt")

print('Results saved in "histogram_equalization_speed_comparison.png')