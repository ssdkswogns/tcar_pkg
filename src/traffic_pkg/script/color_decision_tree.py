#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import cv2
import numpy as np


CLASS_NAMES = ("red", "green", "unknown")
DEFAULT_PROB_THRESHOLD = 0.7

RGB_RULE_SPECS = {
    "rgb15n40_s30v30": (15.0, 0.40, 30, 30),
    "rgb20n40_s40v40": (20.0, 0.40, 40, 40),
    "rgb25n42_s40v50": (25.0, 0.42, 40, 50),
    "rgb30n44_s50v60": (30.0, 0.44, 50, 60),
}

HSV_RULE_SPECS = {
    "hsv_s30v70": (30, 70),
    "hsv_s40v40": (40, 40),
    "hsv_s50v90": (50, 90),
    "hsv_s60v60": (60, 60),
}

TREE_NODES = {
    0: ("rgb15n40_s30v30_largest_g_share", 0.19999999552965164, 1, 22, (0.022754746497902295, 0.06456659318779777, 0.9126786603142999)),
    1: ("hsv_s40v40_r_count", 199.0, 2, 19, (0.02462675080806526, 0.004155764198861013, 0.9712174849930737)),
    2: ("rgb30n44_s50v60_count_r_share", 0.9999999701976776, 3, 16, (0.004723851513600756, 0.00425146636224068, 0.9910246821241586)),
    3: ("hsv_s40v40_g_largest", 7.5, 4, 9, (0.0012660731948565776, 0.00427299703264095, 0.9944609297725024)),
    4: ("rgb25n42_s40v50_count_y_minus_max_rg", -22.5, 5, 8, (0.000758617419752501, 0.0, 0.9992413825802475)),
    5: ("rgb20n40_s40v40_largest_max_per_bbox", 0.08149781078100204, 6, 7, (0.01937046004842615, 0.0, 0.9806295399515739)),
    6: (None, None, -1, -1, (0.5161290322580645, 0.0, 0.4838709677419355)),
    7: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    8: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    9: ("hsv_s40v40_largest_y_minus_max_rg", -25.5, 10, 13, (0.0038240917782026767, 0.02581261950286807, 0.9703632887189293)),
    10: ("hsv_s40v40_r_count", 114.5, 11, 12, (0.010825439783491205, 0.06224627875507442, 0.9269282814614344)),
    11: (None, None, -1, -1, (0.0, 0.0701219512195122, 0.9298780487804879)),
    12: (None, None, -1, -1, (0.0963855421686747, 0.0, 0.9036144578313253)),
    13: ("hsv_s30v70_count_r_per_bbox", 0.0065022604539990425, 14, 15, (0.0, 0.005912786400591279, 0.9940872135994088)),
    14: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    15: (None, None, -1, -1, (0.0, 0.037122969837587005, 0.962877030162413)),
    16: ("hsv_s30v70_count_g_minus_r", -74.0, 17, 18, (0.6875, 0.0, 0.3125)),
    17: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    18: (None, None, -1, -1, (1.0, 0.0, 0.0)),
    19: ("hsv_s60v60_count_max_per_bbox", 0.025230729952454567, 20, 21, (0.8888888888888888, 0.0, 0.1111111111111111)),
    20: (None, None, -1, -1, (1.0, 0.0, 0.0)),
    21: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    22: ("hsv_s40v40_largest_y_per_bbox", 0.025586970150470734, 23, 40, (0.0, 0.7988774555659495, 0.2011225444340505)),
    23: ("rgb15n40_s30v30_largest_max_per_bbox", 0.1583712324500084, 24, 37, (0.0, 0.9038461538461539, 0.09615384615384616)),
    24: ("hsv_s50v90_count_g_per_bbox", 0.007874779403209686, 25, 30, (0.0, 0.8622366288492707, 0.13776337115072934)),
    25: ("hsv_s40v40_largest_max_color", 77.5, 26, 29, (0.0, 0.9578820697954272, 0.0421179302045728)),
    26: ("hsv_s30v70_count_y_minus_max_rg", -19.0, 27, 28, (0.0, 0.8582995951417004, 0.1417004048582996)),
    27: (None, None, -1, -1, (0.0, 0.34782608695652173, 0.6521739130434783)),
    28: (None, None, -1, -1, (0.0, 0.9751243781094527, 0.024875621890547265)),
    29: (None, None, -1, -1, (0.0, 1.0, 0.0)),
    30: ("rgb25n42_s40v50_largest_g_per_crop", 0.007554452167823911, 31, 34, (0.0, 0.6650124069478908, 0.3349875930521092)),
    31: ("hsv_s60v60_largest_top_margin", 70.5, 32, 33, (0.0, 0.05405405405405406, 0.9459459459459459)),
    32: (None, None, -1, -1, (0.0, 0.16666666666666666, 0.8333333333333334)),
    33: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    34: ("hsv_s40v40_largest_y_minus_max_rg", -139.5, 35, 36, (0.0, 0.8024316109422492, 0.19756838905775076)),
    35: (None, None, -1, -1, (0.0, 0.8905109489051095, 0.10948905109489052)),
    36: (None, None, -1, -1, (0.0, 0.36363636363636365, 0.6363636363636364)),
    37: ("rgb30n44_s50v60_g_largest", 580.0, 38, 39, (0.0, 0.9843260188087775, 0.01567398119122257)),
    38: (None, None, -1, -1, (0.0, 1.0, 0.0)),
    39: (None, None, -1, -1, (0.0, 0.5454545454545454, 0.45454545454545453)),
    40: ("hsv_s30v70_count_top_margin", 132.0, 41, 42, (0.0, 0.06015037593984962, 0.9398496240601504)),
    41: (None, None, -1, -1, (0.0, 0.0, 1.0)),
    42: (None, None, -1, -1, (0.0, 0.6153846153846154, 0.38461538461538464)),
}


def _largest_connected_area(mask: np.ndarray) -> int:
    if not bool(mask.any()):
        return 0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if component_count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def _add_mask_stats(stats: Dict[str, float], rule_name: str, masks: Dict[str, np.ndarray]) -> None:
    for color_name, mask in masks.items():
        count = int(mask.sum())
        stats["{}_{}_count".format(rule_name, color_name)] = float(count)
        stats["{}_{}_largest".format(rule_name, color_name)] = float(
            _largest_connected_area(mask) if count else 0
        )


def _extract_mask_stats(crop_bgr: np.ndarray) -> Dict[str, float]:
    bgr_float = crop_bgr.astype(np.float32)
    b, g, r = cv2.split(bgr_float)
    denom = r + g + b + 1e-6
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    stats = {}
    for rule_name, (excess, norm, saturation_min, value_min) in RGB_RULE_SPECS.items():
        saturated = (s >= saturation_min) & (v >= value_min)
        masks = {
            "r": (r - np.maximum(g, b) > excess) & (r / denom > norm) & saturated,
            "g": (g - np.maximum(r, b) > excess) & (g / denom > norm) & saturated,
            "y": (
                (r - b > excess)
                & (g - b > excess)
                & (np.abs(r - g) < 70.0)
                & ((r + g) / denom > 0.70)
                & saturated
            ),
        }
        _add_mask_stats(stats, rule_name, masks)

    for rule_name, (saturation_min, value_min) in HSV_RULE_SPECS.items():
        saturated = (s >= saturation_min) & (v >= value_min)
        masks = {
            "r": ((h <= 10) | (h >= 170)) & saturated,
            "g": (h >= 35) & (h <= 95) & saturated,
            "y": (h >= 15) & (h <= 38) & saturated,
        }
        _add_mask_stats(stats, rule_name, masks)

    return stats


def extract_color_features(crop_bgr: np.ndarray, bbox_area: float) -> Dict[str, float]:
    bbox_area = max(float(bbox_area), 1.0)
    crop_area = max(float(crop_bgr.shape[0] * crop_bgr.shape[1]), 1.0)
    features = _extract_mask_stats(crop_bgr)

    for rule_name in sorted(set(RGB_RULE_SPECS) | set(HSV_RULE_SPECS)):
        for component_name in ("count", "largest"):
            red = features["{}_r_{}".format(rule_name, component_name)]
            green = features["{}_g_{}".format(rule_name, component_name)]
            yellow = features["{}_y_{}".format(rule_name, component_name)]
            total = red + green + yellow + 1e-6
            max_color = max(red, green, yellow)
            second_color = sorted((red, green, yellow))[1]
            prefix = "{}_{}".format(rule_name, component_name)

            features["{}_r_per_bbox".format(prefix)] = red / bbox_area
            features["{}_g_per_bbox".format(prefix)] = green / bbox_area
            features["{}_y_per_bbox".format(prefix)] = yellow / bbox_area
            features["{}_r_per_crop".format(prefix)] = red / crop_area
            features["{}_g_per_crop".format(prefix)] = green / crop_area
            features["{}_y_per_crop".format(prefix)] = yellow / crop_area
            features["{}_r_share".format(prefix)] = red / total
            features["{}_g_share".format(prefix)] = green / total
            features["{}_y_share".format(prefix)] = yellow / total
            features["{}_r_minus_g".format(prefix)] = red - green
            features["{}_g_minus_r".format(prefix)] = green - red
            features["{}_y_minus_max_rg".format(prefix)] = yellow - max(red, green)
            features["{}_max_color".format(prefix)] = max_color
            features["{}_max_per_bbox".format(prefix)] = max_color / bbox_area
            features["{}_max_per_crop".format(prefix)] = max_color / crop_area
            features["{}_top_margin".format(prefix)] = max_color - second_color

    return features


class ColorDecisionTreeClassifier:
    def __init__(self, prob_threshold: float = DEFAULT_PROB_THRESHOLD):
        self.prob_threshold = float(prob_threshold)

    def classify(self, crop_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        x1, y1, x2, y2 = bbox
        bbox_area = max(float(x2 - x1) * float(y2 - y1), 1.0)
        features = extract_color_features(crop_bgr, bbox_area)
        node_id = 0

        while True:
            feature_name, threshold, left_id, right_id, probs = TREE_NODES[node_id]
            if feature_name is None:
                class_index = int(np.argmax(np.asarray(probs, dtype=np.float32)))
                confidence = float(probs[class_index])
                if class_index != 2 and confidence >= self.prob_threshold:
                    return CLASS_NAMES[class_index], confidence
                return "unknown", confidence

            feature_value = features.get(feature_name, 0.0)
            node_id = left_id if feature_value <= threshold else right_id
