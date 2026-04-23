#!/usr/bin/env python3

import os
import re
import zipfile

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
STAMP_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<sec>\d+)_(?P<nsec>\d+)$")


def compression_format_for(filename):
    extension = os.path.splitext(filename)[1].lower()
    if extension in {".jpg", ".jpeg"}:
        return "jpeg"
    if extension == ".png":
        return "png"
    return "compressed"


def param_as_bool(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_stamp_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = STAMP_PATTERN.match(stem)
    if not match:
        return None
    return rospy.Time(int(match.group("sec")), int(match.group("nsec")))


def load_image_members(zip_handle):
    members = []
    for info in zip_handle.infolist():
        if info.is_dir():
            continue
        extension = os.path.splitext(info.filename)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            members.append(info)
    members.sort(key=lambda item: item.filename)
    return members


def build_raw_image_message(image_bgr, header):
    message = Image()
    message.header = header
    message.height = image_bgr.shape[0]
    message.width = image_bgr.shape[1]
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = image_bgr.shape[1] * image_bgr.shape[2]
    message.data = image_bgr.tobytes()
    return message


def has_subscribers(compressed_publisher, raw_publisher, publish_raw):
    if compressed_publisher.get_num_connections() > 0:
        return True
    if publish_raw and raw_publisher is not None and raw_publisher.get_num_connections() > 0:
        return True
    return False


def main():
    rospy.init_node("camera3_zip_publisher")

    zip_path = rospy.get_param("~zip_path", "/data/camera_3__compressed.zip")
    topic_name = rospy.get_param("~topic", "/camera_3_undistorted/compressed")
    raw_topic_name = rospy.get_param("~raw_topic", "/camera_3_undistorted")
    frame_id = rospy.get_param("~frame_id", "camera_3")
    rate_hz = float(rospy.get_param("~rate", 30.0))
    startup_delay = float(rospy.get_param("~startup_delay", 1.0))
    wait_for_subscribers = param_as_bool("~wait_for_subscribers", False)
    use_filename_stamp = param_as_bool("~use_filename_stamp", True)
    publish_raw = param_as_bool("~publish_raw", True)
    loop = param_as_bool("~loop", False)
    start_index = max(0, int(rospy.get_param("~start_index", 0)))
    end_index = int(rospy.get_param("~end_index", -1))

    if rate_hz <= 0.0:
        rospy.logfatal("~rate must be > 0.0, got %.3f", rate_hz)
        return

    if not os.path.isfile(zip_path):
        rospy.logfatal("Zip file not found: %s", zip_path)
        return

    publisher = rospy.Publisher(topic_name, CompressedImage, queue_size=10)
    raw_publisher = None
    if publish_raw:
        raw_publisher = rospy.Publisher(raw_topic_name, Image, queue_size=10)

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = load_image_members(archive)
        if not members:
            rospy.logfatal("No image files found in %s", zip_path)
            return

        if end_index >= 0:
            members = members[start_index:end_index]
        else:
            members = members[start_index:]

        if not members:
            rospy.logfatal(
                "No images selected after slicing: start_index=%d end_index=%d",
                start_index,
                end_index,
            )
            return

        rospy.loginfo(
            "Loaded %d images from %s for %s at %.2f Hz",
            len(members),
            zip_path,
            topic_name,
            rate_hz,
        )
        if publish_raw:
            rospy.loginfo("Raw preview topic: %s", raw_topic_name)
        rospy.loginfo("First image: %s", members[0].filename)
        rospy.loginfo("Last image: %s", members[-1].filename)

        if startup_delay > 0.0:
            rospy.sleep(startup_delay)

        if wait_for_subscribers:
            while not rospy.is_shutdown() and not has_subscribers(publisher, raw_publisher, publish_raw):
                rospy.loginfo_throttle(
                    2.0,
                    "Waiting for a subscriber on %s%s",
                    topic_name,
                    " or " + raw_topic_name if publish_raw else "",
                )
                rospy.sleep(0.1)

        rate = rospy.Rate(rate_hz)
        sequence = 0
        cycle = 0

        while not rospy.is_shutdown():
            cycle += 1
            rospy.loginfo("Starting publish cycle %d", cycle)

            for info in members:
                if rospy.is_shutdown():
                    break

                stamp = parse_stamp_from_filename(info.filename) if use_filename_stamp else None
                if stamp is None:
                    stamp = rospy.Time.now()

                header = Header()
                header.seq = sequence
                header.stamp = stamp
                header.frame_id = frame_id

                compressed_bytes = archive.read(info.filename)

                message = CompressedImage()
                message.header = header
                message.format = compression_format_for(info.filename)
                message.data = compressed_bytes
                publisher.publish(message)

                if publish_raw:
                    np_buffer = np.frombuffer(compressed_bytes, dtype=np.uint8)
                    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        rospy.logwarn_throttle(2.0, "Failed to decode %s for raw publish", info.filename)
                    else:
                        raw_publisher.publish(build_raw_image_message(image_bgr, header))

                sequence += 1
                rate.sleep()

            if not loop:
                break

        rospy.loginfo("Finished publishing %d messages", sequence)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
