import cv2
import time
import numpy as np
from numba import njit

@njit(fastmath=True, cache=False)
def make_histogram(image, bins=256):
    histogram = np.zeros(bins)
    for pixel in image:
        histogram[pixel] += 1
    return histogram

@njit(fastmath=True, cache=False)
def cumsum(values):
    result = [values[0]]
    for i in values[1:]:
        result.append(result[-1] + i)
    return result

@njit(fastmath=True, cache=False)
def normalize(entries):
    entries_arr = np.array(entries)
    min_val = np.min(entries_arr)
    max_val = np.max(entries_arr)
    numerator = (entries_arr - min_val) * 255
    denominator = max_val - min_val
    result = numerator / denominator
    result = result.astype('uint8')
    return result

@njit(fastmath=True, cache=False)
def equalizeHist(img):
    flatten_img = img.flatten()
    cumulativeSum = cumsum(make_histogram(flatten_img))
    cumulativeSum_norm = normalize(cumulativeSum)
    img_new_his = cumulativeSum_norm[flatten_img]
    img_new = np.reshape(img_new_his, img.shape)
    return img_new

if __name__ == "__main__":
    img = cv2.imread('test.png', 0)
    result = equalizeHist(img)

    start = time.time()
    
    for _ in range(100):
        result = equalizeHist(img)
    end_time = (time.time() - start) * 1000

    cv2.imwrite('equalized.png', result)
    
    print(f"{end_time:.3f} ms")
    with open("he_numba.txt", "w") as file:
        file.write(f"{end_time:.3f}")