#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import types
from dataclasses import dataclass

import cv2
import numpy as np
import rospkg
import rospy
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from torch import nn
from torchvision import transforms
from ultralytics import YOLO


IDX2LABEL = ["red", "yellow", "green", "greenleft", "redleft"]

PREP = transforms.Compose(
    [
        transforms.Resize((20, 70)),
        transforms.ToTensor(),
    ]
)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    box_conf: float
    cls_conf: float
    bbox: tuple[int, int, int, int]
    track_length: int = 1
    missed_frames: int = 0


@dataclass
class TrackState:
    bbox: tuple[int, int, int, int]
    cls_id: int
    cls_name: str
    box_conf: float
    cls_conf: float
    hit_streak: int
    last_frame_idx: int
    missed_frames: int = 0
    candidate_cls_id: int | None = None
    candidate_cls_name: str | None = None
    candidate_count: int = 0


class NIA_SEGNet_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = models.mobilenet_v2(weights=None)
        self.fcn.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fcn.last_channel, 5),
        )

    def forward(self, x):
        return self.fcn(x)


def install_lightning_stubs() -> None:
    pl = sys.modules.get("pytorch_lightning")
    if pl is None:
        pl = types.ModuleType("pytorch_lightning")

    callbacks = sys.modules.get("pytorch_lightning.callbacks")
    if callbacks is None:
        callbacks = types.ModuleType("pytorch_lightning.callbacks")

    early = sys.modules.get("pytorch_lightning.callbacks.early_stopping")
    if early is None:
        early = types.ModuleType("pytorch_lightning.callbacks.early_stopping")

    checkpoint = sys.modules.get("pytorch_lightning.callbacks.model_checkpoint")
    if checkpoint is None:
        checkpoint = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class EarlyStopping:  # pragma: no cover - compatibility stub
        pass

    class ModelCheckpoint:  # pragma: no cover - compatibility stub
        pass

    early.EarlyStopping = EarlyStopping
    checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.early_stopping = early
    callbacks.model_checkpoint = checkpoint
    pl.callbacks = callbacks

    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.callbacks.early_stopping"] = early
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = checkpoint


def load_checkpoint_compat(ckpt_path: str):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("pytorch_lightning"):
            raise
        install_lightning_stubs()
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu")


def load_classifier(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = load_checkpoint_compat(ckpt_path)
    state = ckpt.get("state_dict", ckpt)

    model = NIA_SEGNet_module()
    new_state = {}
    for key, value in state.items():
        if key.startswith("fcn.") or key.startswith("fc."):
            new_state[key] = value
        elif key.startswith("model."):
            new_state[key.replace("model.", "")] = value
        else:
            new_state[key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load classifier checkpoint cleanly. "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval().to(device)
    return model


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but this Python environment does not have CUDA-enabled PyTorch."
        )
    return device


def xyxy_to_int_bbox(
    xyxy: torch.Tensor,
    width: int,
    height: int,
    pad: int = 2,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.tolist()
    x1 = max(0, int(np.floor(x1) - pad))
    y1 = max(0, int(np.floor(y1) - pad))
    x2 = min(width - 1, int(np.ceil(x2) + pad))
    y2 = min(height - 1, int(np.ceil(y2) + pad))
    return x1, y1, x2, y2


def draw_label(img_bgr: np.ndarray, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        img_bgr,
        (x1, max(0, y1 - th - 6)),
        (x1 + tw + 4, y1),
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


def color_for_label(name: str) -> tuple[int, int, int]:
    if name.startswith("green"):
        return (0, 255, 0)
    if name.startswith("yellow"):
        return (0, 255, 255)
    return (0, 0, 255)


def bbox_iou(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

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


def should_relax_thresholds_for_bbox(
    bbox: tuple[int, int, int, int],
    previous_tracks: list[TrackState],
    track_iou_thres: float,
    min_track_frames: int,
) -> bool:
    for track in previous_tracks:
        if track.hit_streak < min_track_frames:
            continue
        if bbox_iou(track.bbox, bbox) >= track_iou_thres:
            return True
    return False


@torch.no_grad()
def run_inference_on_image(
    image_bgr: np.ndarray,
    yolo_model: YOLO,
    cls_model: nn.Module,
    device: torch.device,
    previous_tracks: list[TrackState],
    traffic_class_id: int,
    det_conf_thres: float,
    cls_conf_thres: float,
    min_box_aspect_ratio: float,
    track_iou_thres: float,
    min_track_frames: int,
    tracked_threshold_relaxation: float,
    crop_pad: int,
) -> list[Detection]:
    height, width = image_bgr.shape[:2]
    detections: list[Detection] = []

    result = yolo_model.predict(
        source=image_bgr,
        device=str(device),
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxys = result.boxes.xyxy
    confs = result.boxes.conf
    clses = result.boxes.cls

    for index in range(len(xyxys)):
        yolo_cls = int(clses[index].item())
        if yolo_cls != traffic_class_id:
            continue

        raw_x1, raw_y1, raw_x2, raw_y2 = xyxys[index].tolist()
        box_width = raw_x2 - raw_x1
        box_height = raw_y2 - raw_y1
        if box_width <= 0 or box_height <= 0:
            continue
        if (box_width / box_height) < min_box_aspect_ratio:
            continue

        x1, y1, x2, y2 = xyxy_to_int_bbox(
            xyxys[index],
            width=width,
            height=height,
            pad=crop_pad,
        )
        relax_thresholds = should_relax_thresholds_for_bbox(
            bbox=(x1, y1, x2, y2),
            previous_tracks=previous_tracks,
            track_iou_thres=track_iou_thres,
            min_track_frames=min_track_frames,
        )

        current_det_conf_thres = det_conf_thres
        current_cls_conf_thres = cls_conf_thres
        if relax_thresholds:
            current_det_conf_thres = max(0.0, det_conf_thres - tracked_threshold_relaxation)
            current_cls_conf_thres = max(0.0, cls_conf_thres - tracked_threshold_relaxation)

        box_conf = float(confs[index].item())
        if box_conf < current_det_conf_thres:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = PREP(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        logits = cls_model(tensor)
        prob = F.softmax(logits, dim=1)[0]
        cls_id = int(torch.argmax(prob).item())
        cls_conf = float(prob[cls_id].item())
        if cls_conf < current_cls_conf_thres:
            continue

        detections.append(
            Detection(
                cls_id=cls_id,
                cls_name=IDX2LABEL[cls_id],
                box_conf=box_conf,
                cls_conf=cls_conf,
                bbox=(x1, y1, x2, y2),
            )
        )

    return detections


def update_tracks(
    detections: list[Detection],
    previous_tracks: list[TrackState],
    frame_idx: int,
    track_iou_thres: float,
    min_track_frames: int,
    track_persist_after_frames: int,
    track_max_missed_frames: int,
    color_change_frames: int,
) -> tuple[list[TrackState], list[Detection]]:
    updated_tracks: list[TrackState] = []
    matched_track_indices: set[int] = set()
    confirmed_detections: list[Detection] = []

    for detection in detections:
        best_track_index = None
        best_iou = 0.0

        for track_index, track in enumerate(previous_tracks):
            if track_index in matched_track_indices:
                continue
            if track.last_frame_idx != frame_idx - 1:
                continue

            overlap = bbox_iou(track.bbox, detection.bbox)
            if overlap < track_iou_thres or overlap <= best_iou:
                continue

            best_iou = overlap
            best_track_index = track_index

        if best_track_index is None:
            updated_track = TrackState(
                bbox=detection.bbox,
                cls_id=detection.cls_id,
                cls_name=detection.cls_name,
                box_conf=detection.box_conf,
                cls_conf=detection.cls_conf,
                hit_streak=1,
                last_frame_idx=frame_idx,
                missed_frames=0,
                candidate_cls_id=None,
                candidate_cls_name=None,
                candidate_count=0,
            )
        else:
            matched_track_indices.add(best_track_index)
            previous_track = previous_tracks[best_track_index]
            stable_cls_id = previous_track.cls_id
            stable_cls_name = previous_track.cls_name
            stable_cls_conf = previous_track.cls_conf
            candidate_cls_id = previous_track.candidate_cls_id
            candidate_cls_name = previous_track.candidate_cls_name
            candidate_count = previous_track.candidate_count

            if detection.cls_id == previous_track.cls_id:
                stable_cls_conf = detection.cls_conf
                candidate_cls_id = None
                candidate_cls_name = None
                candidate_count = 0
            elif previous_track.candidate_cls_id == detection.cls_id:
                candidate_cls_id = detection.cls_id
                candidate_cls_name = detection.cls_name
                candidate_count = previous_track.candidate_count + 1
                if candidate_count >= color_change_frames:
                    stable_cls_id = detection.cls_id
                    stable_cls_name = detection.cls_name
                    stable_cls_conf = detection.cls_conf
                    candidate_cls_id = None
                    candidate_cls_name = None
                    candidate_count = 0
            else:
                candidate_cls_id = detection.cls_id
                candidate_cls_name = detection.cls_name
                candidate_count = 1

            updated_track = TrackState(
                bbox=detection.bbox,
                cls_id=stable_cls_id,
                cls_name=stable_cls_name,
                box_conf=detection.box_conf,
                cls_conf=stable_cls_conf,
                hit_streak=previous_track.hit_streak + 1,
                last_frame_idx=frame_idx,
                missed_frames=0,
                candidate_cls_id=candidate_cls_id,
                candidate_cls_name=candidate_cls_name,
                candidate_count=candidate_count,
            )

        updated_tracks.append(updated_track)
        if updated_track.hit_streak >= min_track_frames:
            confirmed_detections.append(
                Detection(
                    cls_id=updated_track.cls_id,
                    cls_name=updated_track.cls_name,
                    box_conf=updated_track.box_conf,
                    cls_conf=updated_track.cls_conf,
                    bbox=updated_track.bbox,
                    track_length=updated_track.hit_streak,
                    missed_frames=updated_track.missed_frames,
                )
            )

    for track_index, track in enumerate(previous_tracks):
        if track_index in matched_track_indices:
            continue
        if track.hit_streak < track_persist_after_frames:
            continue
        if track.missed_frames >= track_max_missed_frames:
            continue

        carried_track = TrackState(
            bbox=track.bbox,
            cls_id=track.cls_id,
            cls_name=track.cls_name,
            box_conf=track.box_conf,
            cls_conf=track.cls_conf,
            hit_streak=track.hit_streak,
            last_frame_idx=frame_idx,
            missed_frames=track.missed_frames + 1,
            candidate_cls_id=None,
            candidate_cls_name=None,
            candidate_count=0,
        )
        updated_tracks.append(carried_track)
        confirmed_detections.append(
            Detection(
                cls_id=carried_track.cls_id,
                cls_name=carried_track.cls_name,
                box_conf=carried_track.box_conf,
                cls_conf=carried_track.cls_conf,
                bbox=carried_track.bbox,
                track_length=carried_track.hit_streak,
                missed_frames=carried_track.missed_frames,
            )
        )

    return updated_tracks, confirmed_detections


def draw_confirmed_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    if not detections:
        return image_bgr

    vis = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_for_label(detection.cls_name), 2)
        draw_label(
            vis,
            (
                f"{detection.cls_name} "
                f"(box={detection.box_conf:.2f}, cls={detection.cls_conf:.2f}, "
                f"trk={detection.track_length}, miss={detection.missed_frames})"
            ),
            x1,
            y1,
        )
    return vis


class TrafficLightRosNode:
    def __init__(self):
        rospy.init_node("traffic_light_inference_node", anonymous=True)
        rp = rospkg.RosPack()
        pkg_path = rp.get_path("traffic_pkg")

        self.image_topic = rospy.get_param("~image_topic", "/camera_3_undistorted/compressed")
        self.yolo_weights = rospy.get_param("~yolo_weights", "last.pt")
        self.classifier_ckpt = rospy.get_param("~classifier_ckpt", "mobilenet.ckpt")
        self.traffic_class_id = int(rospy.get_param("~traffic_class_id", 1))
        self.det_conf_thres = float(rospy.get_param("~det_conf_thres", 0.6))
        self.cls_conf_thres = float(rospy.get_param("~cls_conf_thres", 0.9))
        self.min_box_aspect_ratio = float(rospy.get_param("~min_box_aspect_ratio", 2.0))
        self.track_iou_thres = float(rospy.get_param("~track_iou_thres", 0.1))
        self.min_track_frames = int(rospy.get_param("~min_track_frames", 3))
        self.track_persist_after_frames = int(
            rospy.get_param("~track_persist_after_frames", 10)
        )
        self.track_max_missed_frames = int(
            rospy.get_param("~track_max_missed_frames", 5)
        )
        self.tracked_threshold_relaxation = float(
            rospy.get_param("~tracked_threshold_relaxation", 0.2)
        )
        self.color_change_frames = int(rospy.get_param("~color_change_frames", 4))
        self.crop_pad = int(rospy.get_param("~crop_pad", 2))
        self.device = infer_device(rospy.get_param("~device", "auto"))
        self.pub_det_topic = rospy.get_param("~pub_detections_topic", "/traffic/detections")
        self.pub_img_topic = rospy.get_param("~pub_image_topic", "/traffic/image_bbox/compressed")

        self.yolo_weights = os.path.join(pkg_path, "models", self.yolo_weights)
        self.classifier_ckpt = os.path.join(pkg_path, "models", self.classifier_ckpt)

        rospy.loginfo("[Traffic] Device: %s", self.device)
        rospy.loginfo("[Traffic] YOLO weights: %s", self.yolo_weights)
        rospy.loginfo("[Traffic] Classifier ckpt: %s", self.classifier_ckpt)

        self.yolo = YOLO(self.yolo_weights)
        self.cls_model = load_classifier(self.classifier_ckpt, self.device)
        self.active_tracks: list[TrackState] = []
        self.frame_index = 0

        self.pub_dets = rospy.Publisher(self.pub_det_topic, Float32MultiArray, queue_size=1)
        self.pub_img = rospy.Publisher(self.pub_img_topic, CompressedImage, queue_size=1)
        rospy.Subscriber(
            self.image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("[Traffic] Ready. Subscribed to %s", self.image_topic)
        rospy.spin()

    @torch.no_grad()
    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            rospy.logwarn("[Traffic] Failed to decode image")
            return

        self.frame_index += 1

        # conf & w:h ratio filter, tracker
        raw_detections = run_inference_on_image(
            image_bgr=img_bgr,
            yolo_model=self.yolo,
            cls_model=self.cls_model,
            device=self.device,
            previous_tracks=self.active_tracks,
            traffic_class_id=self.traffic_class_id,
            det_conf_thres=self.det_conf_thres,
            cls_conf_thres=self.cls_conf_thres,
            min_box_aspect_ratio=self.min_box_aspect_ratio,
            track_iou_thres=self.track_iou_thres,
            min_track_frames=self.min_track_frames,
            tracked_threshold_relaxation=self.tracked_threshold_relaxation,
            crop_pad=self.crop_pad,
        )
        # tracking filter, color change handling
        self.active_tracks, detections = update_tracks(
            detections=raw_detections,
            previous_tracks=self.active_tracks,
            frame_idx=self.frame_index,
            track_iou_thres=self.track_iou_thres,
            min_track_frames=self.min_track_frames,
            track_persist_after_frames=self.track_persist_after_frames,
            track_max_missed_frames=self.track_max_missed_frames,
            color_change_frames=self.color_change_frames,
        )

        vis = draw_confirmed_detections(img_bgr, detections)

        out = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            out.extend(
                [
                    float(detection.cls_id),
                    float(detection.box_conf),
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
