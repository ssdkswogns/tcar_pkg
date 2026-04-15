#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import rospkg
import rospy
import torch
from PIL import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO


TYPE_TO_ID = {"car": 0, "pedestrian": 1}
STATE_TO_ID = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "greenleft": 3,
    "redleft": 4,
    "redyellow": 5,
    "unknown": 6,
}

TRACK_CONFIRM_FRAMES = 3
TRACK_CONFIRM_CONF = {"car": 0.6, "pedestrian": 0.45}
TRACK_KEEP_CONF = {"car": 0.4, "pedestrian": 0.3}
TRACK_MAX_MISSED = 5
STATE_CONFIRM_FRAMES = 3
STATE_CONFIRM_CONF = 0.9
IMAGE_MSG_FIELDS = 8
REACQUIRE_FRAMES = 2


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    traffic_type: str
    state_name: str
    det_conf: float
    cls_conf: float


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    traffic_type: str
    state_name: str
    det_conf: float
    cls_conf: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    consecutive_strong_hits: int = 1
    confirmed: bool = False
    candidate_state_name: str | None = None
    candidate_state_hits: int = 0
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    reacquiring: bool = False
    reacquire_candidate_state_name: str | None = None
    reacquire_hits: int = 0


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        return torch.device("cpu")
    if device_arg.startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() == 0):
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_yolo_device_arg(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
        return "cpu"
    return device_arg


def pad_bbox(
    box: tuple[float, float, float, float], padding_ratio: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    px1 = max(0.0, min(x1 - pad_x, float(img_w)))
    py1 = max(0.0, min(y1 - pad_y, float(img_h)))
    px2 = max(0.0, min(x2 + pad_x, float(img_w)))
    py2 = max(0.0, min(y2 + pad_y, float(img_h)))

    return int(np.floor(px1)), int(np.floor(py1)), int(np.ceil(px2)), int(np.ceil(py2))


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def per_type_det_threshold(traffic_type: str, car_det_conf: float, ped_det_conf: float, default_conf: float) -> float:
    if traffic_type == "car":
        return float(car_det_conf)
    if traffic_type == "pedestrian":
        return float(ped_det_conf)
    return float(default_conf)


def is_strong_detection(detection: Detection) -> bool:
    return detection.det_conf >= TRACK_CONFIRM_CONF.get(detection.traffic_type, 0.6)


def should_keep_confirmed_track(detection: Detection) -> bool:
    return detection.det_conf > TRACK_KEEP_CONF.get(detection.traffic_type, 0.4)


def is_trackable_state(state_name: str) -> bool:
    return str(state_name).lower() != "unknown"


def should_keep_detection_for_tracking(
    traffic_type: str,
    state_name: str,
) -> bool:
    if is_trackable_state(state_name):
        return True
    return str(traffic_type) == "pedestrian"


def normalize_state_observation(track: TrackState, observed_state: str) -> str:
    if track.traffic_type != "pedestrian":
        return observed_state
    if observed_state != "unknown":
        return observed_state
    if track.state_name == "green" or track.candidate_state_name == "green":
        return "green"
    return observed_state


def update_track_state_label(
    track: TrackState,
    observed_state: str,
    cls_conf: float,
) -> tuple[str, str | None, int, float]:
    normalized_state = normalize_state_observation(track, observed_state)
    if not is_trackable_state(normalized_state):
        return track.state_name, None, 0, track.cls_conf
    if cls_conf < STATE_CONFIRM_CONF:
        return track.state_name, None, 0, track.cls_conf
    if normalized_state == track.state_name:
        return track.state_name, None, 0, cls_conf

    candidate_hits = track.candidate_state_hits + 1 if track.candidate_state_name == normalized_state else 1
    if candidate_hits >= STATE_CONFIRM_FRAMES:
        return normalized_state, None, 0, cls_conf
    return track.state_name, normalized_state, candidate_hits, track.cls_conf


def create_track(track_id: int, detection: Detection) -> TrackState:
    strong = is_strong_detection(detection)
    bbox = detection.bbox
    return TrackState(
        track_id=track_id,
        bbox=bbox,
        traffic_type=detection.traffic_type,
        state_name=detection.state_name,
        det_conf=detection.det_conf,
        cls_conf=detection.cls_conf,
        consecutive_strong_hits=1 if strong else 0,
        bbox_history=[bbox],
    )


def bbox_center_wh(bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = float(max(x2 - x1, 1))
    height = float(max(y2 - y1, 1))
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return center_x, center_y, width, height


def rect_from_center_wh(center_x: float, center_y: float, width: float, height: float) -> tuple[int, int, int, int]:
    width = max(width, 1.0)
    height = max(height, 1.0)
    x1 = int(round(center_x - width / 2.0))
    y1 = int(round(center_y - height / 2.0))
    x2 = int(round(center_x + width / 2.0))
    y2 = int(round(center_y + height / 2.0))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return (x1, y1, x2, y2)


def append_bbox_history(
    history: list[tuple[int, int, int, int]],
    bbox: tuple[int, int, int, int],
    maxlen: int = 3,
) -> list[tuple[int, int, int, int]]:
    next_history = list(history)
    next_history.append(bbox)
    return next_history[-maxlen:]


def predict_bbox_from_history(track: TrackState) -> tuple[int, int, int, int]:
    history = track.bbox_history or [track.bbox]
    if len(history) < 2:
        return track.bbox

    centers = [bbox_center_wh(bbox) for bbox in history]
    delta_cx = []
    delta_cy = []
    delta_w = []
    delta_h = []
    for previous, current in zip(centers[:-1], centers[1:]):
        delta_cx.append(current[0] - previous[0])
        delta_cy.append(current[1] - previous[1])
        delta_w.append(current[2] - previous[2])
        delta_h.append(current[3] - previous[3])

    last_cx, last_cy, last_w, last_h = centers[-1]
    next_cx = last_cx + (sum(delta_cx) / len(delta_cx))
    next_cy = last_cy + (sum(delta_cy) / len(delta_cy))
    next_w = max(1.0, last_w + (sum(delta_w) / len(delta_w)))
    next_h = max(1.0, last_h + (sum(delta_h) / len(delta_h)))
    return rect_from_center_wh(next_cx, next_cy, next_w, next_h)


def match_tracks_to_detections(
    active_tracks: list[TrackState],
    detections: list[Detection],
    min_iou: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(active_tracks):
        for detection_index, detection in enumerate(detections):
            if detection.traffic_type != track.traffic_type:
                continue
            overlap = bbox_iou(track.bbox, detection.bbox)
            if overlap < min_iou:
                continue
            candidates.append((overlap, track_index, detection_index))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matches: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for _, track_index, detection_index in candidates:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((track_index, detection_index))
    return matches


def update_tracks(
    active_tracks: list[TrackState],
    detections: list[Detection],
    track_min_iou: float,
    track_max_missed: int,
) -> list[TrackState]:
    matches = match_tracks_to_detections(active_tracks, detections, track_min_iou)
    matched_track_indices = {track_index for track_index, _ in matches}
    matched_detection_indices = {detection_index for _, detection_index in matches}
    next_tracks: list[TrackState] = []

    for track_index, detection_index in matches:
        previous_track = active_tracks[track_index]
        detection = detections[detection_index]
        matched_bbox = detection.bbox

        if previous_track.confirmed:
            if should_keep_confirmed_track(detection):
                stable_state_name = previous_track.state_name
                stable_cls_conf = previous_track.cls_conf
                candidate_state_name = previous_track.candidate_state_name
                candidate_state_hits = previous_track.candidate_state_hits
                reacquiring = previous_track.reacquiring
                reacquire_candidate_state_name = previous_track.reacquire_candidate_state_name
                reacquire_hits = previous_track.reacquire_hits

                if previous_track.misses > 0 or previous_track.reacquiring:
                    observed_state = detection.state_name
                    if is_trackable_state(observed_state):
                        if previous_track.reacquiring and previous_track.reacquire_candidate_state_name == observed_state:
                            reacquire_hits = previous_track.reacquire_hits + 1
                        else:
                            reacquire_hits = 1
                        reacquiring = True
                        reacquire_candidate_state_name = observed_state
                        if reacquire_hits >= REACQUIRE_FRAMES:
                            stable_state_name = observed_state
                            stable_cls_conf = detection.cls_conf
                            candidate_state_name = None
                            candidate_state_hits = 0
                            reacquiring = False
                            reacquire_candidate_state_name = None
                            reacquire_hits = 0
                    else:
                        reacquiring = True
                        reacquire_candidate_state_name = None
                        reacquire_hits = 0
                else:
                    stable_state_name, candidate_state_name, candidate_state_hits, stable_cls_conf = update_track_state_label(
                        previous_track,
                        observed_state=detection.state_name,
                        cls_conf=detection.cls_conf,
                    )
                    reacquiring = False
                    reacquire_candidate_state_name = None
                    reacquire_hits = 0

                next_tracks.append(
                    TrackState(
                        track_id=previous_track.track_id,
                        bbox=matched_bbox,
                        traffic_type=previous_track.traffic_type,
                        state_name=stable_state_name,
                        det_conf=detection.det_conf,
                        cls_conf=stable_cls_conf,
                        hits=previous_track.hits + 1,
                        misses=0,
                        age=previous_track.age + 1,
                        consecutive_strong_hits=previous_track.consecutive_strong_hits,
                        confirmed=True,
                        candidate_state_name=candidate_state_name,
                        candidate_state_hits=candidate_state_hits,
                        bbox_history=append_bbox_history(previous_track.bbox_history, matched_bbox),
                        reacquiring=reacquiring,
                        reacquire_candidate_state_name=reacquire_candidate_state_name,
                        reacquire_hits=reacquire_hits,
                    )
                )
            else:
                next_misses = previous_track.misses + 1
                if next_misses > track_max_missed:
                    continue
                predicted_bbox = predict_bbox_from_history(previous_track)
                next_tracks.append(
                    TrackState(
                        track_id=previous_track.track_id,
                        bbox=predicted_bbox,
                        traffic_type=previous_track.traffic_type,
                        state_name=previous_track.state_name,
                        det_conf=previous_track.det_conf,
                        cls_conf=previous_track.cls_conf,
                        hits=previous_track.hits,
                        misses=next_misses,
                        age=previous_track.age + 1,
                        consecutive_strong_hits=previous_track.consecutive_strong_hits,
                        confirmed=True,
                        candidate_state_name=previous_track.candidate_state_name,
                        candidate_state_hits=previous_track.candidate_state_hits,
                        bbox_history=append_bbox_history(previous_track.bbox_history, predicted_bbox),
                        reacquiring=previous_track.reacquiring,
                        reacquire_candidate_state_name=previous_track.reacquire_candidate_state_name,
                        reacquire_hits=previous_track.reacquire_hits,
                    )
                )
            continue

        strong_hits = previous_track.consecutive_strong_hits + 1 if is_strong_detection(detection) else 0
        confirmed = strong_hits >= TRACK_CONFIRM_FRAMES
        stable_state_name = previous_track.state_name
        stable_cls_conf = previous_track.cls_conf
        candidate_state_name = previous_track.candidate_state_name
        candidate_state_hits = previous_track.candidate_state_hits
        if confirmed:
            stable_state_name, candidate_state_name, candidate_state_hits, stable_cls_conf = update_track_state_label(
                previous_track,
                observed_state=detection.state_name,
                cls_conf=detection.cls_conf,
            )

        next_tracks.append(
            TrackState(
                track_id=previous_track.track_id,
                bbox=detection.bbox,
                traffic_type=previous_track.traffic_type,
                state_name=stable_state_name,
                det_conf=detection.det_conf,
                cls_conf=stable_cls_conf,
                hits=previous_track.hits + 1,
                misses=0,
                age=previous_track.age + 1,
                consecutive_strong_hits=strong_hits,
                confirmed=confirmed,
                candidate_state_name=candidate_state_name,
                candidate_state_hits=candidate_state_hits,
                bbox_history=append_bbox_history(previous_track.bbox_history, matched_bbox),
                reacquiring=False,
                reacquire_candidate_state_name=None,
                reacquire_hits=0,
            )
        )

    for track_index, track in enumerate(active_tracks):
        if track_index in matched_track_indices:
            continue
        next_misses = track.misses + 1
        if next_misses > track_max_missed:
            continue
        predicted_bbox = predict_bbox_from_history(track)
        next_tracks.append(
            TrackState(
                track_id=track.track_id,
                bbox=predicted_bbox,
                traffic_type=track.traffic_type,
                state_name=track.state_name,
                det_conf=track.det_conf,
                cls_conf=track.cls_conf,
                hits=track.hits,
                misses=next_misses,
                age=track.age + 1,
                consecutive_strong_hits=0 if not track.confirmed else track.consecutive_strong_hits,
                confirmed=track.confirmed,
                candidate_state_name=track.candidate_state_name,
                candidate_state_hits=track.candidate_state_hits,
                bbox_history=append_bbox_history(track.bbox_history, predicted_bbox),
                reacquiring=track.reacquiring,
                reacquire_candidate_state_name=track.reacquire_candidate_state_name,
                reacquire_hits=track.reacquire_hits,
            )
        )

    next_track_id = max((track.track_id for track in active_tracks), default=0) + 1
    for detection_index, detection in enumerate(detections):
        if detection_index in matched_detection_indices:
            continue
        if not is_trackable_state(detection.state_name):
            continue
        next_tracks.append(create_track(next_track_id, detection))
        next_track_id += 1

    next_tracks.sort(key=lambda item: item.track_id)
    return next_tracks


def confirmed_track_detections(tracks: list[TrackState], min_hits: int) -> list[TrackState]:
    return [
        track
        for track in tracks
        if track.confirmed and not track.reacquiring and is_trackable_state(track.state_name) and track.hits >= min_hits
    ]


def draw_label(img_bgr: np.ndarray, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 0, 0), -1)
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


def color_for_label(name: str) -> tuple[int, int, int]:
    if name.startswith("green"):
        return (0, 255, 0)
    if name.startswith("yellow"):
        return (0, 255, 255)
    return (0, 0, 255)


class TrafficLightIouRosNode:
    def __init__(self):
        rospy.init_node("traffic_light_iou_inference_node", anonymous=True)
        rp = rospkg.RosPack()
        pkg_path = rp.get_path("traffic_pkg")

        self.image_topic = rospy.get_param("~image_topic", "/camera_3_undistorted/compressed")
        self.stage1_weights = os.path.join(pkg_path, "models", rospy.get_param("~stage1_weights", "0413_stage1.pt"))
        self.car_state_weights = os.path.join(pkg_path, "models", rospy.get_param("~car_state_weights", "0415_stage2_car_yolo.pt"))
        self.ped_state_weights = os.path.join(pkg_path, "models", rospy.get_param("~ped_state_weights", "0415_stage2_ped_yolo.pt"))
        self.det_conf = float(rospy.get_param("~det_conf", 0.25))
        self.car_det_conf = float(rospy.get_param("~car_det_conf", 0.5))
        self.ped_det_conf = float(rospy.get_param("~ped_det_conf", 0.3))
        self.iou = float(rospy.get_param("~iou", 0.7))
        self.padding_ratio = float(rospy.get_param("~padding_ratio", 0.1))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.crop_pad_ratio = self.padding_ratio
        self.track_min_hits = int(rospy.get_param("~track_min_hits", 3))
        self.track_max_missed = int(rospy.get_param("~track_max_missed", TRACK_MAX_MISSED))
        self.track_min_iou = float(rospy.get_param("~track_min_iou", 0.01))
        self.device_arg = rospy.get_param("~device", "auto")
        self.cls_device_arg = rospy.get_param("~cls_device", "auto")
        self.pub_det_topic = rospy.get_param("~pub_detections_topic", "/traffic/detections")
        self.pub_img_topic = rospy.get_param("~pub_image_topic", "/traffic/image_bbox/compressed")

        self.detector_device = resolve_yolo_device_arg(self.device_arg)
        self.cls_device = resolve_torch_device(self.cls_device_arg)

        self.stage1_detector = YOLO(self.stage1_weights)
        self.detector_names = getattr(self.stage1_detector, "names", {}) or {}
        self.car_classifier = self._load_classifier(self.car_state_weights)
        self.ped_classifier = self._load_classifier(self.ped_state_weights)
        self.active_tracks: list[TrackState] = []

        rospy.loginfo("[TrafficIoU] Stage1: %s", self.stage1_weights)
        rospy.loginfo("[TrafficIoU] Car cls: %s", self.car_state_weights)
        rospy.loginfo("[TrafficIoU] Ped cls: %s", self.ped_state_weights)
        rospy.loginfo("[TrafficIoU] Detector device: %s", self.detector_device)
        rospy.loginfo("[TrafficIoU] Classifier device: %s", self.cls_device)

        self.pub_dets = rospy.Publisher(self.pub_det_topic, Float32MultiArray, queue_size=1)
        self.pub_img = rospy.Publisher(self.pub_img_topic, CompressedImage, queue_size=1)
        rospy.Subscriber(
            self.image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("[TrafficIoU] Ready. Subscribed to %s", self.image_topic)
        rospy.spin()

    def _load_classifier(self, weights_path: str) -> dict:
        model = YOLO(weights_path)
        model.model.to(self.cls_device)
        model.model.eval()
        names = getattr(model, "names", {}) or getattr(model.model, "names", {}) or {}
        class_names = [str(names[idx]) for idx in sorted(names)]
        return {
            "model": model.model,
            "preprocess": model.model.transforms,
            "class_names": class_names,
        }

    @torch.inference_mode()
    def _classify_crops(self, crops_bgr: list[np.ndarray], classifier_bundle: dict) -> list[tuple[str, float]]:
        if not crops_bgr:
            return []
        batch_tensors = []
        for crop_bgr in crops_bgr:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            batch_tensors.append(classifier_bundle["preprocess"](Image.fromarray(crop_rgb)))
        outputs = classifier_bundle["model"](torch.stack(batch_tensors).to(self.cls_device))
        probs = outputs[0] if isinstance(outputs, tuple) else outputs
        predictions: list[tuple[str, float]] = []
        for prob in probs:
            pred_idx = int(torch.argmax(prob).item())
            pred_conf = float(prob[pred_idx].item())
            predictions.append((classifier_bundle["class_names"][pred_idx], pred_conf))
        return predictions

    @torch.inference_mode()
    def _collect_detections(self, image_bgr: np.ndarray) -> list[Detection]:
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

        candidates: list[dict] = []
        car_crops: list[np.ndarray] = []
        ped_crops: list[np.ndarray] = []
        car_indices: list[int] = []
        ped_indices: list[int] = []

        for box, det_conf, cls_id in zip(boxes, confs, cls_ids):
            traffic_type = str(self.detector_names.get(int(cls_id), str(cls_id)))
            if float(det_conf) < per_type_det_threshold(traffic_type, self.car_det_conf, self.ped_det_conf, self.det_conf):
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            cx1, cy1, cx2, cy2 = pad_bbox((x1, y1, x2, y2), self.crop_pad_ratio, img_w, img_h)
            crop = image_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            candidate = {
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "traffic_type": traffic_type,
                "det_conf": float(det_conf),
            }
            candidate_index = len(candidates)
            candidates.append(candidate)
            if traffic_type == "car":
                car_crops.append(crop)
                car_indices.append(candidate_index)
            else:
                ped_crops.append(crop)
                ped_indices.append(candidate_index)

        predictions: list[tuple[str, float] | None] = [None] * len(candidates)
        for candidate_index, prediction in zip(car_indices, self._classify_crops(car_crops, self.car_classifier)):
            predictions[candidate_index] = prediction
        for candidate_index, prediction in zip(ped_indices, self._classify_crops(ped_crops, self.ped_classifier)):
            predictions[candidate_index] = prediction

        detections: list[Detection] = []
        for candidate, prediction in zip(candidates, predictions):
            if prediction is None:
                continue
            state_name, cls_conf = prediction
            if not should_keep_detection_for_tracking(candidate["traffic_type"], state_name):
                continue
            detections.append(
                Detection(
                    bbox=candidate["bbox"],
                    traffic_type=candidate["traffic_type"],
                    state_name=state_name,
                    det_conf=candidate["det_conf"],
                    cls_conf=float(cls_conf),
                )
            )
        return detections

    @torch.inference_mode()
    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            rospy.logwarn("[TrafficIoU] Failed to decode image")
            return

        detections = self._collect_detections(img_bgr)
        self.active_tracks = update_tracks(
            active_tracks=self.active_tracks,
            detections=detections,
            track_min_iou=self.track_min_iou,
            track_max_missed=self.track_max_missed,
        )
        confirmed_tracks = confirmed_track_detections(self.active_tracks, self.track_min_hits)

        vis = img_bgr.copy()
        out = []
        for track in confirmed_tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_for_label(track.state_name), 2)
            draw_label(
                vis,
                (
                    f"{track.traffic_type} {track.state_name} "
                    f"(det={track.det_conf:.2f}, cls={track.cls_conf:.2f}, id={track.track_id})"
                ),
                x1,
                y1,
            )
            out.extend(
                [
                    float(STATE_TO_ID.get(track.state_name, STATE_TO_ID["unknown"])),
                    float(track.det_conf),
                    float(track.cls_conf),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(TYPE_TO_ID.get(track.traffic_type, -1)),
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
        TrafficLightIouRosNode()
    except rospy.ROSInterruptException:
        pass
