import cv2
import time
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('test.png', 0)
    result = cv2.equalizeHist(img) 

    start = time.time()
    
    for _ in range(100):
        result = cv2.equalizeHist(img) 
    end_time = (time.time() - start) * 1000

    cv2.imwrite('equalized.png', result)
    
    print(f"{end_time:.3f} ms")
    with open("he_opencv.txt", "w") as file:
        file.write(f"{end_time:.3f}")