import os
import re

import rospkg


_REQUIRED_CONSTANTS = (
    "TYPE_RED_YELLOW_GREEN",
    "TYPE_RED_YELLOW_LEFT",
    "TYPE_RED_YELLOW_LEFT_GREEN",
    "TYPE_PED_RED_GREEN",
    "TYPE_YELLOW_YELLOW_YELLOW",
    "COLOR_RED",
    "COLOR_YELLOW",
    "COLOR_RED_YELLOW",
    "COLOR_GREEN",
    "COLOR_RED_GREEN",
    "COLOR_YELLOW_GREEN",
    "COLOR_LEFT",
    "COLOR_RED_LEFT",
    "COLOR_YELLOW_LEFT",
    "COLOR_GREEN_LEFT",
)

_CONSTANT_PATTERN = re.compile(
    r"static\s+constexpr\s+std::uint16_t\s+([A-Z0-9_]+)\s*=\s*([0-9]+)\s*;"
)


def _candidate_header_paths():
    package_path = rospkg.RosPack().get_path("autohyu_msgs")
    return (
        os.path.join(package_path, "include", "autohyu_msgs", "traffic_light_constants.h"),
        os.path.abspath(
            os.path.join(
                package_path,
                os.pardir,
                os.pardir,
                "include",
                "autohyu_msgs",
                "traffic_light_constants.h",
            )
        ),
    )


def _load_header_constants():
    for header_path in _candidate_header_paths():
        if not os.path.exists(header_path):
            continue

        constants = {}
        with open(header_path, "r", encoding="utf-8") as header_file:
            for line in header_file:
                match = _CONSTANT_PATTERN.search(line)
                if match:
                    constants[match.group(1)] = int(match.group(2))

        missing = [name for name in _REQUIRED_CONSTANTS if name not in constants]
        if missing:
            raise RuntimeError(
                "Missing traffic light constants in {}: {}".format(
                    header_path,
                    ", ".join(missing),
                )
            )
        return constants

    raise RuntimeError("traffic_light_constants.h was not found in autohyu_msgs")


_CONSTANTS = _load_header_constants()
globals().update({name: _CONSTANTS[name] for name in _REQUIRED_CONSTANTS})

__all__ = list(_REQUIRED_CONSTANTS)
