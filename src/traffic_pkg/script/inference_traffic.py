#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospkg
import rospy
import torch
from PIL import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO


STATE_TO_ID = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "greenleft": 3,
    "redleft": 4,
    "redyellow": 5,
    "unknown": 6,
}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    traffic_type: str
    state_name: str
    det_conf: float
    cls_conf: float


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        return torch.device("cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_yolo_device_arg(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
        return "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def normalize_traffic_type(raw_name: str) -> Optional[str]:
    lowered = str(raw_name).strip().lower()
    if "ped" in lowered or "walker" in lowered:
        return "pedestrian"
    if "car" in lowered or "veh" in lowered:
        return "car"
    return None


def clip_bbox(
    box: Sequence[float],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(int(np.floor(x1)), img_w - 1))
    y1 = max(0, min(int(np.floor(y1)), img_h - 1))
    x2 = max(x1 + 1, min(int(np.ceil(x2)), img_w))
    y2 = max(y1 + 1, min(int(np.ceil(y2)), img_h))
    return x1, y1, x2, y2


def pad_bbox(
    box: Sequence[float],
    padding_ratio: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    padded = (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(img_w), x2 + pad_x),
        min(float(img_h), y2 + pad_y),
    )
    return clip_bbox(padded, img_w, img_h)


def draw_label(img_bgr: np.ndarray, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        img_bgr,
        (x1, max(0, y1 - text_h - 6)),
        (x1 + text_w + 4, y1),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img_bgr,
        text,
        (x1 + 2, y1 - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def color_for_label(label_name: str) -> Tuple[int, int, int]:
    if label_name.startswith("green"):
        return (0, 255, 0)
    if label_name.startswith("yellow"):
        return (0, 255, 255)
    if label_name == "unknown":
        return (192, 192, 192)
    return (0, 0, 255)


class TrafficLightRosNode:
    def __init__(self):
        rospy.init_node("traffic_light_inference_node", anonymous=True)
        pkg_path = rospkg.RosPack().get_path("traffic_pkg")

        self.image_topic = rospy.get_param("~image_topic", "/camera_3_undistorted/compressed")
        self.stage1_weights = os.path.join(pkg_path, "models", rospy.get_param("~stage1_weights", "0413_stage1.pt"))
        self.car_state_weights = os.path.join(
            pkg_path,
            "models",
            rospy.get_param("~car_state_weights", "0415_stage2_car_yolo.pt"),
        )
        self.ped_state_weights = os.path.join(
            pkg_path,
            "models",
            rospy.get_param("~ped_state_weights", "0415_stage2_ped_yolo.pt"),
        )
        self.det_conf = float(rospy.get_param("~det_conf", 0.25))
        self.car_det_conf = float(rospy.get_param("~car_det_conf", 0.5))
        self.ped_det_conf = float(rospy.get_param("~ped_det_conf", 0.3))
        self.cls_conf = float(rospy.get_param("~cls_conf", 0.0))
        self.iou = float(rospy.get_param("~iou", 0.7))
        self.padding_ratio = float(rospy.get_param("~padding_ratio", 0.1))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.device_arg = rospy.get_param("~device", "auto")
        self.cls_device_arg = rospy.get_param("~cls_device", "auto")
        self.pub_det_topic = rospy.get_param("~pub_detections_topic", "/traffic/detections")
        self.pub_img_topic = rospy.get_param("~pub_image_topic", "/traffic/image_bbox/compressed")

        self.detector_device = resolve_yolo_device_arg(self.device_arg)
        self.classifier_device = resolve_torch_device(self.cls_device_arg)

        self.stage1_detector = YOLO(self.stage1_weights)
        self.detector_names = getattr(self.stage1_detector, "names", {}) or {}

        self.car_classifier = self._load_classifier(self.car_state_weights)
        self.ped_classifier = self._load_classifier(self.ped_state_weights)

        self.pub_dets = rospy.Publisher(self.pub_det_topic, Float32MultiArray, queue_size=1)
        self.pub_img = rospy.Publisher(self.pub_img_topic, CompressedImage, queue_size=1)
        rospy.Subscriber(
            self.image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.spin()

    def _load_classifier(self, weights_path: str) -> Dict[str, object]:
        classifier = YOLO(weights_path)
        classifier.model.to(self.classifier_device)
        classifier.model.eval()
        preprocess = getattr(classifier.model, "transforms", None)
        if preprocess is None:
            raise RuntimeError("Classifier transforms are missing for {}".format(weights_path))

        names_obj = getattr(classifier, "names", None) or getattr(classifier.model, "names", None)
        if isinstance(names_obj, dict):
            class_names = [str(names_obj[idx]) for idx in sorted(names_obj)]
        elif isinstance(names_obj, (list, tuple)):
            class_names = [str(name) for name in names_obj]
        else:
            class_names = []
        if not class_names:
            raise RuntimeError("Classifier class names are missing for {}".format(weights_path))

        return {
            "model": classifier.model,
            "preprocess": preprocess,
            "class_names": class_names,
        }

    @torch.inference_mode()
    def _classify_crops(
        self,
        crops_bgr: List[np.ndarray],
        classifier_bundle: Dict[str, object],
    ) -> List[Tuple[str, float]]:
        if not crops_bgr:
            return []

        preprocess = classifier_bundle["preprocess"]
        model = classifier_bundle["model"]
        class_names = classifier_bundle["class_names"]

        batch_tensors = []
        for crop_bgr in crops_bgr:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            batch_tensors.append(preprocess(Image.fromarray(crop_rgb)))

        outputs = model(torch.stack(batch_tensors).to(self.classifier_device))
        probs = outputs[0] if isinstance(outputs, tuple) else outputs

        predictions = []
        for prob in probs:
            pred_index = int(torch.argmax(prob).item())
            pred_conf = float(prob[pred_index].item())
            predictions.append((str(class_names[pred_index]), pred_conf))
        return predictions

    @torch.inference_mode()
    def _collect_detections(self, image_bgr: np.ndarray) -> List[Detection]:
        result = self.stage1_detector.predict(
            source=image_bgr,
            conf=self.det_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.detector_device,
            save=False,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        img_h, img_w = image_bgr.shape[:2]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        candidates = []
        car_crops = []
        ped_crops = []
        car_indices = []
        ped_indices = []

        for box, det_conf, cls_id in zip(boxes, confs, cls_ids):
            raw_detector_name = str(self.detector_names.get(int(cls_id), str(cls_id)))
            traffic_type = normalize_traffic_type(raw_detector_name)
            if traffic_type is None:
                continue

            det_conf = float(det_conf)
            type_threshold = self.car_det_conf if traffic_type == "car" else self.ped_det_conf
            if det_conf < type_threshold:
                continue

            crop_bbox = pad_bbox(box.tolist(), self.padding_ratio, img_w, img_h)
            x1, y1, x2, y2 = crop_bbox
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            candidate = {
                "bbox": clip_bbox(box.tolist(), img_w, img_h),
                "traffic_type": traffic_type,
                "det_conf": det_conf,
            }
            candidate_index = len(candidates)
            candidates.append(candidate)
            if traffic_type == "car":
                car_crops.append(crop)
                car_indices.append(candidate_index)
            else:
                ped_crops.append(crop)
                ped_indices.append(candidate_index)

        predictions = [None] * len(candidates)
        for candidate_index, prediction in zip(car_indices, self._classify_crops(car_crops, self.car_classifier)):
            predictions[candidate_index] = prediction
        for candidate_index, prediction in zip(ped_indices, self._classify_crops(ped_crops, self.ped_classifier)):
            predictions[candidate_index] = prediction

        detections = []
        for candidate, prediction in zip(candidates, predictions):
            if prediction is None:
                continue
            state_name, cls_conf = prediction
            if cls_conf < self.cls_conf:
                continue
            detections.append(
                Detection(
                    bbox=candidate["bbox"],
                    traffic_type=candidate["traffic_type"],
                    state_name=state_name,
                    det_conf=float(candidate["det_conf"]),
                    cls_conf=float(cls_conf),
                )
            )

        return detections

    @torch.inference_mode()
    def image_callback(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return

        detections = self._collect_detections(img_bgr)
        vis = img_bgr.copy()
        out = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cls_id = STATE_TO_ID.get(detection.state_name, STATE_TO_ID["unknown"])

            cv2.rectangle(vis, (x1, y1), (x2, y2), color_for_label(detection.state_name), 2)
            draw_label(
                vis,
                "{} {} (det={:.2f}, cls={:.2f})".format(
                    detection.traffic_type,
                    detection.state_name,
                    detection.det_conf,
                    detection.cls_conf,
                ),
                x1,
                y1,
            )

            out.extend(
                [
                    float(cls_id),
                    float(detection.det_conf),
                    float(detection.cls_conf),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ]
            )

        arr = Float32MultiArray()
        arr.data = out
        self.pub_dets.publish(arr)

        vis_msg = CompressedImage()
        vis_msg.header = msg.header
        vis_msg.header.stamp = rospy.Time.now()
        vis_msg.format = "jpeg"
        vis_msg.data = np.array(cv2.imencode(".jpg", vis)[1]).tobytes()
        self.pub_img.publish(vis_msg)


if __name__ == "__main__":
    try:
        TrafficLightRosNode()
    except rospy.ROSInterruptException:
        pass
