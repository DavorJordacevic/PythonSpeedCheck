import time

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area

    return iou

def nms(boxes, threshold):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    selected_boxes = []
    while len(boxes) > 0:
        best_box = boxes.pop(0)
        selected_boxes.append(best_box)
        ious = [calculate_iou(best_box, box) for box in boxes]
        filtered_boxes = [box for i, box in enumerate(boxes) if ious[i] <= threshold]
        boxes = filtered_boxes

    return selected_boxes

if __name__ == "__main__":
    iou_threshold = 0.5
    boxes = [[10, 10, 20, 20, 0.9],
             [15, 15, 25, 25, 0.8],
             [30, 30, 40, 40, 0.7],
             [35, 35, 45, 45, 0.6]]

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
    with open("nms_python.txt", "w") as file:
        file.write(f"{end_time:.3f}")