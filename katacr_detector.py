from ultralytics import YOLO
import numpy as np
import torch

class DualYOLODetector:
    def __init__(self, model1_path, model2_path):
        self.model1 = YOLO(model1_path)
        self.model2 = YOLO(model2_path)

    def detect(self, frame):
        results1 = self.model1(frame)
        results2 = self.model2(frame)
        return self.merge_results(results1, results2)
    
    def merge_results(self, r1, r2):
        # Extract detections from both models
        dets1 = r1[0].boxes  # assuming ultralytics format
        dets2 = r2[0].boxes

        # Convert to numpy arrays
        boxes1 = dets1.xyxy.cpu().numpy()
        confs1 = dets1.conf.cpu().numpy()
        classes1 = dets1.cls.cpu().numpy()

        boxes2 = dets2.xyxy.cpu().numpy()
        confs2 = dets2.conf.cpu().numpy()
        classes2 = dets2.cls.cpu().numpy()

        # Concatenate
        boxes = np.vstack([boxes1, boxes2])
        confs = np.hstack([confs1, confs2])
        classes = np.hstack([classes1, classes2])

        # Optional: apply NMS to remove overlaps
        from ultralytics.utils.ops import non_max_suppression
        
        merged = non_max_suppression(
            torch.tensor(boxes),
            torch.tensor(confs),
            torch.tensor(classes),
            iou_thres=0.5
        )

        return merged