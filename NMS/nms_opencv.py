import time
import numpy as np
from cv2.dnn import NMSBoxes

if __name__ == "__main__":
    iou_threshold = 0.5
    boxes = np.array([[10, 10, 20, 20, 0.9],
                      [15, 15, 25, 25, 0.8],
                      [30, 30, 40, 40, 0.7],
                      [35, 35, 45, 45, 0.6]])
    
    selected_boxes = NMSBoxes(boxes[:, :4], boxes[:, 4], iou_threshold, 0.5)

    # Test
    # Expected output:
    # [10, 10, 20, 20, 0.9]
    # [15, 15, 25, 25, 0.8]
    # [30, 30, 40, 40, 0.7]
    # [35, 35, 45, 45, 0.6]
    # for box in selected_boxes:
    #     print(box)

    # Test speed
    start = time.time()
    
    for _ in range(1000):
        NMSBoxes(boxes[:, :4], boxes[:, 4], iou_threshold, 0.5)
    end_time = (time.time() - start) * 1000

    print(f"{end_time:.3f} ms")
    with open("nms_opencv.txt", "w") as file:
        file.write(f"{end_time:.3f}")