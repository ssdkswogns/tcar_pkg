#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import cv2
import numpy as np


CLASS_NAMES = ("green", "red", "redleft", "redyellow", "unknown", "yellow")
DEFAULT_PROB_THRESHOLD = 0.7

HSV_PRESETS = {
    "loose": {"s": 30, "v": 55},
    "mid": {"s": 45, "v": 75},
    "strict": {"s": 65, "v": 105},
    "bright": {"s": 35, "v": 135},
}

TREE_NODES = {
    0: ("rgb_m35_v75_red_top", 25.5, 1, 36, (0.16666666666666258, 0.16666666666666236, 0.1666666666666636, 0.16666666666666274, 0.16666666666664573, 0.1666666666666629)),
    1: ("rgb_m15_v40_green_right_share", 0.19072726927697659, 2, 19, (0.501943683498969, 0.0036024666279830387, 0.0, 0.0, 0.49445384987297186, 0.0)),
    2: ("bbox_aspect", 4.0645833015441895, 3, 16, (0.049598776068521463, 0.006949194025826752, 0.0, 0.0, 0.9434520299056588, 0.0)),
    3: ("det_conf", 0.8048499822616577, 4, 11, (0.01794454793334333, 0.0072282613834370405, 0.0, 0.0, 0.974827190683221, 0.0)),
    4: ("hsv_bright_green_bottom", 12.5, 5, 10, (0.004604620381989528, 0.00741916705088504, 0.0, 0.0, 0.9879762125671274, 0.0)),
    5: ("rgb_m15_v40_red_right_minus_left", -64.0, 6, 9, (0.0, 0.0049930028423241565, 0.0, 0.0, 0.9950069971576766, 0.0)),
    6: ("hsv_loose_red_share", 0.77161505818367, 7, 8, (0.0, 0.29045073678127714, 0.0, 0.0, 0.709549263218723, 0.0)),
    7: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    8: (None, -2.0, -1, -1, (0.0, 0.6579937817051218, 0.0, 0.0, 0.3420062182948781, 0.0)),
    9: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    10: (None, -2.0, -1, -1, (0.49029983855307135, 0.2633308941511112, 0.0, 0.0, 0.24636926729581715, 0.0)),
    11: ("hsv_loose_red_largest_per_crop", 0.004945359192788601, 12, 15, (0.523034231578034, 0.0, 0.0, 0.0, 0.47696576842196475, 0.0)),
    12: ("hsv_v_std", 44.155099868774414, 13, 14, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    13: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    14: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    15: (None, -2.0, -1, -1, (0.8995525727069352, 0.0, 0.0, 0.0, 0.1004474272930649, 0.0)),
    16: ("hsv_loose_green_cx", 0.6217391192913055, 17, 18, (0.8378363361069053, 0.0, 0.0, 0.0, 0.16216366389309486, 0.0)),
    17: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    18: (None, -2.0, -1, -1, (0.9853299135804485, 0.0, 0.0, 0.0, 0.014670086419551405, 0.0)),
    19: ("bbox_area", 517.0, 20, 21, (0.9888543994439618, 0.0, 0.0, 0.0, 0.011145600556038901, 0.0)),
    20: (None, -2.0, -1, -1, (0.4487723214285715, 0.0, 0.0, 0.0, 0.5512276785714286, 0.0)),
    21: ("rgb_m35_v75_green_cx", 0.7779353260993958, 22, 33, (0.9916624741120641, 0.0, 0.0, 0.0, 0.008337525887936669, 0.0)),
    22: ("hsv_s_std", 62.81975173950195, 23, 24, (0.9956926762905937, 0.0, 0.0, 0.0, 0.004307323709406751, 0.0)),
    23: (None, -2.0, -1, -1, (0.8995525727069352, 0.0, 0.0, 0.0, 0.1004474272930649, 0.0)),
    24: ("hsv_strict_green_cy", 0.5547457337379456, 25, 30, (0.9972639251887679, 0.0, 0.0, 0.0, 0.002736074811232172, 0.0)),
    25: ("hsv_h_std", 7.823981761932373, 26, 27, (0.9991395808155267, 0.0, 0.0, 0.0, 0.0008604191844733638, 0.0)),
    26: (None, -2.0, -1, -1, (0.9781551036294639, 0.0, 0.0, 0.0, 0.02184489637053615, 0.0)),
    27: ("cls_conf", 0.9958909749984741, 28, 29, (0.9997054594308086, 0.0, 0.0, 0.0, 0.0002945405691914507, 0.0)),
    28: (None, -2.0, -1, -1, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    29: (None, -2.0, -1, -1, (0.9899507609668755, 0.0, 0.0, 0.0, 0.01004923903312444, 0.0)),
    30: ("rgb_m15_v40_red_plus_green", 153.5, 31, 32, (0.9583824366732432, 0.0, 0.0, 0.0, 0.041617563326756805, 0.0)),
    31: (None, -2.0, -1, -1, (0.86480557467309, 0.0, 0.0, 0.0, 0.13519442532690984, 0.0)),
    32: (None, -2.0, -1, -1, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    33: ("hsv_h_mean", 102.00343322753906, 34, 35, (0.8704091204271591, 0.0, 0.0, 0.0, 0.1295908795728407, 0.0)),
    34: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    35: (None, -2.0, -1, -1, (0.9641144501278772, 0.0, 0.0, 0.0, 0.035885549872122766, 0.0)),
    36: ("hsv_strict_green_bottom", 4.5, 37, 62, (0.0, 0.24772610500778808, 0.24951689612832814, 0.24951689612832686, 0.0037232066072369107, 0.24951689612832711)),
    37: ("hsv_h_p90", 140.5, 38, 49, (0.0, 0.32281367496090885, 0.014214160195588561, 0.3291124783747809, 0.0047472080939411345, 0.32911247837478125)),
    38: ("crop_w", 381.5, 39, 46, (0.0, 0.4342914895934788, 0.0018671903011640673, 0.0, 0.0018170394549751293, 0.5620242806503842)),
    39: ("rgb_m35_v75_green_left", 42.5, 40, 43, (0.0, 0.0878033447978216, 0.0, 0.0, 0.003390225428154776, 0.9088064297740237)),
    40: ("hsv_loose_yellow_count", 2.5, 41, 42, (0.0, 0.9683623127575981, 0.0, 0.0, 0.03163768724240192, 0.0)),
    41: (None, -2.0, -1, -1, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    42: (None, -2.0, -1, -1, (0.0, 0.4665274393781181, 0.0, 0.0, 0.5334725606218819, 0.0)),
    43: ("hsv_loose_green_left", 748.5, 44, 45, (0.0, 0.0, 0.0, 0.0, 0.0005735808653423323, 0.9994264191346578)),
    44: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 0.0014899428855227215, 0.9985100571144773)),
    45: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    46: ("rgb_m25_v55_red_cy", 0.9284152984619141, 47, 48, (0.0, 0.8344874235187969, 0.004023806559510617, 0.0, 0.0, 0.1614887699216928)),
    47: (None, -2.0, -1, -1, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    48: (None, -2.0, -1, -1, (0.0, 0.10916683649325931, 0.02165720818898073, 0.0, 0.0, 0.8691759553177602)),
    49: ("hsv_loose_green_bottom", 32.0, 50, 57, (0.0, 0.16529164516649156, 0.03166085814163984, 0.7941598583861318, 0.008887638305738863, 0.0)),
    50: ("hsv_loose_green_bottom", 3.0, 51, 56, (0.0, 0.018575365120721415, 0.0, 0.970562827557693, 0.010861807321585733, 0.0)),
    51: ("hsv_loose_red_largest", 120.0, 52, 55, (0.0, 0.6310173015810736, 0.0, 0.0, 0.3689826984189255, 0.0)),
    52: ("rgb_m15_v40_green_cx", 0.04740109480917454, 53, 54, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    53: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    54: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    55: (None, -2.0, -1, -1, (0.0, 0.9505910165484635, 0.0, 0.0, 0.04940898345153666, 0.0)),
    56: (None, -2.0, -1, -1, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    57: ("rgb_m35_v75_yellow_per_crop", 0.157047837972641, 58, 61, (0.0, 0.8258030908143776, 0.1741969091856227, 0.0, 0.0, 0.0)),
    58: ("hsv_mid_green_h", 0.337354838848114, 59, 60, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    59: (None, -2.0, -1, -1, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    60: (None, -2.0, -1, -1, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    61: (None, -2.0, -1, -1, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    62: ("hsv_mid_green_per_crop", 0.022759990766644478, 63, 66, (0.0, 0.012340963474308256, 0.9871458783329775, 0.0, 0.0005131581927143349, 0.0)),
    63: ("rgb_m25_v55_yellow_largest_per_crop", 0.006765821715816855, 64, 65, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    64: (None, -2.0, -1, -1, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    65: (None, -2.0, -1, -1, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    66: (None, -2.0, -1, -1, (0.0, 0.41147420254840994, 0.5714160022100843, 0.0, 0.017109795241505632, 0.0)),
}


def color_masks_hsv(image_bgr: np.ndarray, s_min: int, v_min: int) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_val = (s >= s_min) & (v >= v_min)
    return {
        "red": (((h <= 12) | (h >= 168)) & sat_val).astype(np.uint8),
        "green": ((h >= 36) & (h <= 95) & sat_val).astype(np.uint8),
        "yellow": ((h >= 14) & (h <= 38) & sat_val).astype(np.uint8),
    }


def color_masks_rgb(image_bgr: np.ndarray, margin: int, min_value: int) -> Dict[str, np.ndarray]:
    b, g, r = cv2.split(image_bgr)
    bright = np.maximum.reduce([r, g, b]) >= min_value
    return {
        "red": ((r > g + margin) & (r > b + margin) & bright).astype(np.uint8),
        "green": ((g > r + margin) & (g > b + margin) & bright).astype(np.uint8),
        "yellow": (
            (r > b + margin)
            & (g > b + margin)
            & (np.abs(r.astype(np.int16) - g.astype(np.int16)) <= 85)
            & bright
        ).astype(np.uint8),
    }


def largest_component_stats(mask: np.ndarray) -> Dict[str, float]:
    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0.0,
            "largest": 0.0,
            "components": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "w": 0.0,
            "h": 0.0,
            "aspect": 0.0,
        }

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return {
            "count": float(count),
            "largest": 0.0,
            "components": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "w": 0.0,
            "h": 0.0,
            "aspect": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    w = float(stats[best, cv2.CC_STAT_WIDTH])
    h = float(stats[best, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[best]
    mh, mw = mask.shape[:2]
    return {
        "count": float(count),
        "largest": float(stats[best, cv2.CC_STAT_AREA]),
        "components": float(num_labels - 1),
        "cx": float(cx / max(1, mw)),
        "cy": float(cy / max(1, mh)),
        "w": float(w / max(1, mw)),
        "h": float(h / max(1, mh)),
        "aspect": float(w / max(1.0, h)),
    }


def add_mask_features(features: Dict[str, float], prefix: str, masks: Dict[str, np.ndarray]) -> None:
    colors = ("red", "green", "yellow")
    area = float(next(iter(masks.values())).size)
    total_count = 1e-6 + sum(float(masks[color].sum()) for color in colors)
    thirds = {}

    for color in colors:
        mask = masks[color]
        h, w = mask.shape[:2]
        left = mask[:, : w // 3].sum()
        mid = mask[:, w // 3 : (2 * w) // 3].sum()
        right = mask[:, (2 * w) // 3 :].sum()
        top = mask[: h // 2, :].sum()
        bottom = mask[h // 2 :, :].sum()
        stats = largest_component_stats(mask)

        for key, value in stats.items():
            features["{}_{}_{}".format(prefix, color, key)] = value
        features["{}_{}_per_crop".format(prefix, color)] = stats["count"] / area
        features["{}_{}_largest_per_crop".format(prefix, color)] = stats["largest"] / area
        features["{}_{}_share".format(prefix, color)] = stats["count"] / total_count
        features["{}_{}_left".format(prefix, color)] = float(left)
        features["{}_{}_mid".format(prefix, color)] = float(mid)
        features["{}_{}_right".format(prefix, color)] = float(right)
        features["{}_{}_top".format(prefix, color)] = float(top)
        features["{}_{}_bottom".format(prefix, color)] = float(bottom)
        features["{}_{}_left_share".format(prefix, color)] = float(left) / (stats["count"] + 1e-6)
        features["{}_{}_right_share".format(prefix, color)] = float(right) / (stats["count"] + 1e-6)
        thirds[color] = (float(left), float(mid), float(right))

    red_total = features["{}_red_count".format(prefix)]
    green_total = features["{}_green_count".format(prefix)]
    yellow_total = features["{}_yellow_count".format(prefix)]
    features["{}_red_plus_yellow".format(prefix)] = red_total + yellow_total
    features["{}_red_plus_green".format(prefix)] = red_total + green_total
    features["{}_max_minus_second".format(prefix)] = (
        max(red_total, green_total, yellow_total)
        - sorted([red_total, green_total, yellow_total])[1]
    )
    features["{}_red_right_minus_left".format(prefix)] = thirds["red"][2] - thirds["red"][0]
    features["{}_green_right_minus_left".format(prefix)] = thirds["green"][2] - thirds["green"][0]
    features["{}_yellow_mid_minus_edges".format(prefix)] = thirds["yellow"][1] - max(
        thirds["yellow"][0],
        thirds["yellow"][2],
    )


def extract_color_features(
    crop_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    det_conf: float = 0.0,
    cls_conf: float = 0.0,
) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, int(x2) - int(x1))
    bbox_h = max(1, int(y2) - int(y1))
    crop_h, crop_w = crop_bgr.shape[:2]
    features = {
        "det_conf": float(det_conf),
        "cls_conf": float(cls_conf),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "bbox_area": float(bbox_w * bbox_h),
        "bbox_aspect": float(bbox_w / float(bbox_h)),
        "crop_w": float(crop_w),
        "crop_h": float(crop_h),
        "crop_area": float(max(1, crop_w * crop_h)),
        "csv_red_pixels": 0.0,
        "csv_green_pixels": 0.0,
        "csv_yellow_pixels": 0.0,
        "csv_red_share": 0.0,
        "csv_green_share": 0.0,
        "csv_yellow_share": 0.0,
    }

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    for channel_name, channel in zip(("h", "s", "v"), cv2.split(hsv)):
        features["hsv_{}_mean".format(channel_name)] = float(channel.mean())
        features["hsv_{}_std".format(channel_name)] = float(channel.std())
        features["hsv_{}_p90".format(channel_name)] = float(np.percentile(channel, 90))
        features["hsv_{}_p99".format(channel_name)] = float(np.percentile(channel, 99))

    for preset, cfg in HSV_PRESETS.items():
        add_mask_features(
            features,
            "hsv_{}".format(preset),
            color_masks_hsv(crop_bgr, cfg["s"], cfg["v"]),
        )

    for margin, min_value in ((15, 40), (25, 55), (35, 75)):
        add_mask_features(
            features,
            "rgb_m{}_v{}".format(margin, min_value),
            color_masks_rgb(crop_bgr, margin, min_value),
        )

    return features


class ColorDecisionTreeClassifier:
    def __init__(self, prob_threshold: float = DEFAULT_PROB_THRESHOLD):
        self.prob_threshold = float(prob_threshold)

    def classify(
        self,
        crop_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        det_conf: float = 0.0,
        cls_conf: float = 0.0,
    ) -> Tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        features = extract_color_features(crop_bgr, bbox, det_conf, cls_conf)
        node_id = 0

        while True:
            feature_name, threshold, left_id, right_id, probs = TREE_NODES[node_id]
            if feature_name is None:
                class_index = int(np.argmax(np.asarray(probs, dtype=np.float32)))
                confidence = float(probs[class_index])
                class_name = CLASS_NAMES[class_index]
                if class_name != "unknown" and confidence >= self.prob_threshold:
                    return class_name, confidence
                return "unknown", confidence

            feature_value = features.get(feature_name, 0.0)
            node_id = left_id if feature_value <= threshold else right_id
