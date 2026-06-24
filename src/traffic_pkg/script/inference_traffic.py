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
from autohyu_msgs.msg import TrafficLight, TrafficLights
from PIL import Image
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

from traffic_light_constants import (
    COLOR_GREEN,
    COLOR_GREEN_LEFT,
    COLOR_LEFT,
    COLOR_RED,
    COLOR_RED_GREEN,
    COLOR_RED_LEFT,
    COLOR_RED_YELLOW,
    COLOR_YELLOW,
    COLOR_YELLOW_GREEN,
    COLOR_YELLOW_LEFT,
    TYPE_RED_YELLOW_LEFT,
)


TRAFFIC_LIGHT_COLOR_BY_STATE = {
    "red": COLOR_RED,
    "yellow": COLOR_YELLOW,
    "redyellow": COLOR_RED_YELLOW,
    "green": COLOR_GREEN,
    "redgreen": COLOR_RED_GREEN,
    "yellowgreen": COLOR_YELLOW_GREEN,
    "left": COLOR_LEFT,
    "redleft": COLOR_RED_LEFT,
    "yellowleft": COLOR_YELLOW_LEFT,
    "greenleft": COLOR_GREEN_LEFT,
    "leftgreen": COLOR_GREEN_LEFT,
}

DEFAULT_STAGE1_WEIGHTS = "0608_stage1.pt"
DEFAULT_STAGE2_WEIGHTS = "0624_stage2_3.pt"
CAR_TRAFFIC_TYPE = "car"
GPU_YOLO_DEVICE = "0"
GPU_TORCH_DEVICE = torch.device("cuda:0")


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    traffic_type: str
    state_name: str
    det_conf: float
    cls_conf: float


def require_cuda_device() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("traffic_pkg requires a CUDA GPU; CPU inference is disabled.")
    torch.cuda.set_device(0)


def normalize_state_key(raw_name: str) -> str:
    return "".join(char for char in str(raw_name).strip().lower() if char.isalnum())


def is_car_traffic_light(raw_name: str) -> bool:
    lowered = str(raw_name).strip().lower()
    return "car" in lowered or "veh" in lowered


def infer_traffic_light_type() -> int:
    return TYPE_RED_YELLOW_LEFT


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
        self.stage1_weights = os.path.join(
            pkg_path,
            "models",
            rospy.get_param("~stage1_weights", DEFAULT_STAGE1_WEIGHTS),
        )
        stage2_weights_name = rospy.get_param("~stage2_weights", DEFAULT_STAGE2_WEIGHTS)
        self.stage2_weights = os.path.join(pkg_path, "models", stage2_weights_name)
        self.det_conf = float(rospy.get_param("~det_conf", 0.2))
        self.iou = float(rospy.get_param("~iou", 0.7))
        self.padding_ratio = float(rospy.get_param("~padding_ratio", 0.1))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.pub_det_topic = rospy.get_param("~pub_detections_topic", "/traffic/detections")
        self.pub_img_topic = rospy.get_param("~pub_image_topic", "/traffic/image_bbox/compressed")

        require_cuda_device()
        self.detector_device = GPU_YOLO_DEVICE
        self.classifier_device = GPU_TORCH_DEVICE

        self.stage1_detector = YOLO(self.stage1_weights)
        self.detector_names = getattr(self.stage1_detector, "names", {}) or {}

        self.stage2_classifier = self._load_classifier(self.stage2_weights)

        self.pub_dets = rospy.Publisher(self.pub_det_topic, TrafficLights, queue_size=1)
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
        crops = []

        for box, det_conf, cls_id in zip(boxes, confs, cls_ids):
            raw_detector_name = str(self.detector_names.get(int(cls_id), str(cls_id)))
            if not is_car_traffic_light(raw_detector_name):
                continue

            det_conf = float(det_conf)
            crop_bbox = pad_bbox(box.tolist(), self.padding_ratio, img_w, img_h)
            x1, y1, x2, y2 = crop_bbox
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            candidate = {
                "bbox": clip_bbox(box.tolist(), img_w, img_h),
                "traffic_type": CAR_TRAFFIC_TYPE,
                "det_conf": det_conf,
            }
            candidates.append(candidate)
            crops.append(crop)

        detections = []
        for candidate, prediction in zip(candidates, self._classify_crops(crops, self.stage2_classifier)):
            state_name, cls_conf = prediction
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

    def _to_traffic_light_msg(self, detection: Detection, detection_id: int, header) -> Optional[TrafficLight]:
        state_key = normalize_state_key(detection.state_name)
        color_value = TRAFFIC_LIGHT_COLOR_BY_STATE.get(state_key)
        if color_value is None:
            rospy.logdebug("Skipping unsupported traffic light state: %s", detection.state_name)
            return None

        msg = TrafficLight()
        msg.header.seq = header.seq
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id
        msg.id = str(detection_id)
        msg.type = infer_traffic_light_type()
        msg.color = color_value
        msg.bbox_xmin, msg.bbox_ymin, msg.bbox_xmax, msg.bbox_ymax = detection.bbox
        return msg

    @torch.inference_mode()
    def image_callback(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return

        detections = self._collect_detections(img_bgr)
        vis = img_bgr.copy()
        traffic_light_msgs = []

        for detection_id, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox

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

            traffic_light_msg = self._to_traffic_light_msg(detection, detection_id, msg.header)
            if traffic_light_msg is not None:
                traffic_light_msgs.append(traffic_light_msg)

        detections_msg = TrafficLights()
        detections_msg.header.seq = msg.header.seq
        detections_msg.header.stamp = msg.header.stamp
        detections_msg.header.frame_id = msg.header.frame_id
        detections_msg.lights = traffic_light_msgs
        self.pub_dets.publish(detections_msg)

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
