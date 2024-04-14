import time
import torch
import numpy as np
from torchvision.ops import nms

if __name__ == "__main__":
    iou_threshold = 0.5
    boxes = torch.tensor([[10, 10, 20, 20, 0.9],
                          [15, 15, 25, 25, 0.8],
                          [30, 30, 40, 40, 0.7],
                          [35, 35, 45, 45, 0.6]])

    selected_boxes = nms(boxes[:, :4], boxes[:, 4], iou_threshold)

    # Test
    # Expected output:
    # [10, 10, 20, 20, 0.9]
    # [15, 15, 25, 25, 0.8]
    # [30, 30, 40, 40, 0.7]
    # [35, 35, 45, 45, 0.6]
    for box in selected_boxes:
        print(box)

    # Test speed
    start = time.time()
    for _ in range(1000):
        nms(boxes[:, :4], boxes[:, 4], iou_threshold)
    end_time = (time.time() - start) * 1000

    print(f"{end_time:.3f} ms")
    with open("nms_torch.txt", "w") as file:
        file.write(f"{end_time:.3f}")