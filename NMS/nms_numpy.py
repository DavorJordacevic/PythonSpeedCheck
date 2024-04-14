import time
import numpy as np

def nms(boxes, threshold):
    if len(boxes) == 0:
        return []

    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    selected_boxes = []

    while len(boxes) > 0:
        best_box = boxes[0]
        selected_boxes.append(best_box)

        x1 = np.maximum(best_box[0], boxes[1:, 0])
        y1 = np.maximum(best_box[1], boxes[1:, 1])
        x2 = np.minimum(best_box[2], boxes[1:, 2])
        y2 = np.minimum(best_box[3], boxes[1:, 3])

        intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        area_box1 = (best_box[2] - best_box[0] + 1) * (best_box[3] - best_box[1] + 1)
        area_box2 = (boxes[1:, 2] - boxes[1:, 0] + 1) * (boxes[1:, 3] - boxes[1:, 1] + 1)
        union_area = area_box1 + area_box2 - intersection_area
        iou = intersection_area / union_area

        filtered_boxes = boxes[1:][iou <= threshold]
        boxes = filtered_boxes

    return np.array(selected_boxes)

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
    with open("nms_numpy.txt", "w") as file:
        file.write(f"{end_time:.3f}")