import time
import numpy as np
from numba import njit

@njit(fastmath=True, cache=False)
def nms(boxes, threshold):

    keep = []
    if boxes.shape[0] != 0:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

    return keep

if __name__ == "__main__":
    iou_threshold = 0.5
    boxes = np.array([[10, 10, 20, 20, 0.9],
                      [15, 15, 25, 25, 0.8],
                      [30, 30, 40, 40, 0.7],
                      [35, 35, 45, 45, 0.6]])

    selected_boxes = nms(boxes, iou_threshold)

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
        nms(boxes, iou_threshold)
    end_time = (time.time() - start) * 1000

    print(f"{end_time:.3f} ms")
    with open("nms_numba.txt", "w") as file:
        file.write(f"{end_time:.3f}")