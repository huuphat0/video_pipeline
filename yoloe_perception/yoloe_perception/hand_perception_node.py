from __future__ import annotations

import threading
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


HAND_LANDMARK_IDS = tuple(range(21))
PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)
HAND_LANDMARK_CONNECTIONS = (
    # Lòng bàn tay
    (0, 1),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),
    # Ngón cái
    (1, 2),
    (2, 3),
    (3, 4),
    # Ngón trỏ
    (5, 6),
    (6, 7),
    (7, 8),
    # Ngón giữa
    (9, 10),
    (10, 11),
    (11, 12),
    # Ngón áp út
    (13, 14),
    (14, 15),
    (15, 16),
    # Ngón út
    (17, 18),
    (18, 19),
    (19, 20),
)


class HandPerceptionNode(Node):
    """Detect selected hand landmarks from the latest RealSense RGB frame."""

    def __init__(self) -> None:
        super().__init__("hand_perception_node")

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._latest_pair: Optional[tuple[Image, Image]] = None
        self._camera_info: Optional[CameraInfo] = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_warning_reported = False

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._color_subscriber = Subscriber(
            self,
            Image,
            "/camera/camera/color/image_raw",
            qos_profile=qos_profile_sensor_data,
        )
        self._depth_subscriber = Subscriber(
            self,
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            qos_profile=qos_profile_sensor_data,
        )
        self._rgbd_sync = ApproximateTimeSynchronizer(
            [self._color_subscriber, self._depth_subscriber],
            queue_size=10,
            slop=0.05,
        )
        self._rgbd_sync.registerCallback(self._rgbd_callback)
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self._publisher = self.create_publisher(
            Image,
            "/hand/annotated_image",
            qos_profile_sensor_data,
        )
        self._palm_center_2d_publisher = self.create_publisher(
            Point,
            "/hand/palm_center_2d",
            qos_profile_sensor_data,
        )
        self._palm_center_3d_publisher = self.create_publisher(
            PointStamped,
            "/hand/palm_center_3d",
            qos_profile_sensor_data,
        )
        self._palm_center_camera_link_publisher = self.create_publisher(
            PointStamped,
            "/hand/palm_center_camera_link",
            qos_profile_sensor_data,
        )

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            "Hand RGB-D perception started -> /hand/annotated_image, "
            "/hand/palm_center_2d, /hand/palm_center_3d, "
            "/hand/palm_center_camera_link"
        )

    def _rgbd_callback(self, color_message: Image, depth_message: Image) -> None:
        # Chỉ giữ cặp RGB-D mới nhất để không tạo hàng đợi ảnh cũ.
        with self._lock:
            self._latest_pair = (color_message, depth_message)
        self._frame_event.set()

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._frame_event.wait(timeout=0.1):
                continue

            with self._lock:
                pair = self._latest_pair
                self._latest_pair = None
                self._frame_event.clear()

            if pair is None:
                continue

            try:
                self._process_pair(*pair)
            except Exception as error:  # Keep the worker alive after a bad frame.
                self.get_logger().error(f"Hand processing error: {error}")

    def _process_pair(
        self,
        color_message: Image,
        depth_message: Image,
    ) -> None:
        frame = self.bridge.imgmsg_to_cv2(
            color_message,
            desired_encoding="bgr8",
        )
        depth = self.bridge.imgmsg_to_cv2(
            depth_message,
            desired_encoding="passthrough",
        )
        depth_scale = 1.0 if depth_message.encoding == "32FC1" else 0.001
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if result.multi_hand_landmarks:
            height, width = frame.shape[:2]
            for hand_index, landmarks in enumerate(result.multi_hand_landmarks):
                pixel_points = {}
                for landmark_id in HAND_LANDMARK_IDS:
                    landmark = landmarks.landmark[landmark_id]
                    u = min(width - 1, max(0, round(landmark.x * width)))
                    v = min(height - 1, max(0, round(landmark.y * height)))
                    pixel_points[landmark_id] = (u, v)

                for start_id, end_id in HAND_LANDMARK_CONNECTIONS:
                    cv2.line(
                        frame,
                        pixel_points[start_id],
                        pixel_points[end_id],
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                for u, v in pixel_points.values():
                    cv2.circle(frame, (u, v), 5, (0, 255, 0), -1)

                palm_u = int(round(np.mean([
                    pixel_points[index][0] for index in PALM_LANDMARK_IDS
                ])))
                palm_v = int(round(np.mean([
                    pixel_points[index][1] for index in PALM_LANDMARK_IDS
                ])))
                cv2.circle(frame, (palm_u, palm_v), 9, (0, 0, 255), -1)

                palm_2d = Point()
                palm_2d.x = float(palm_u)
                palm_2d.y = float(palm_v)
                palm_2d.z = 0.0
                self._palm_center_2d_publisher.publish(palm_2d)

                z_m = self._estimate_depth_m(
                    depth,
                    palm_u,
                    palm_v,
                    depth_scale,
                )
                camera_info = self._camera_info
                if z_m is not None and camera_info is not None:
                    fx = float(camera_info.k[0])
                    fy = float(camera_info.k[4])
                    cx = float(camera_info.k[2])
                    cy = float(camera_info.k[5])
                    if fx > 0.0 and fy > 0.0:
                        palm_3d = PointStamped()
                        palm_3d.header = color_message.header
                        palm_3d.point.x = (palm_u - cx) * z_m / fx
                        palm_3d.point.y = (palm_v - cy) * z_m / fy
                        palm_3d.point.z = z_m
                        self._palm_center_3d_publisher.publish(palm_3d)
                        self._publish_camera_link_point(palm_3d)

        output = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        output.header = color_message.header
        self._publisher.publish(output)

    def _publish_camera_link_point(self, palm_3d: PointStamped) -> None:
        try:
            transform = self._tf_buffer.lookup_transform(
                "camera_link",
                palm_3d.header.frame_id,
                Time.from_msg(palm_3d.header.stamp),
            )
            palm_camera_link = do_transform_point(palm_3d, transform)
            self._palm_center_camera_link_publisher.publish(
                palm_camera_link
            )
            self._tf_warning_reported = False
        except TransformException as error:
            if not self._tf_warning_reported:
                self.get_logger().warning(
                    "Cannot transform palm center to camera_link: "
                    f"{error}"
                )
                self._tf_warning_reported = True

    @staticmethod
    def _estimate_depth_m(
        depth: np.ndarray,
        u: int,
        v: int,
        depth_scale: float,
    ) -> Optional[float]:
        """Estimate robust palm depth from a 3x3 window using median/MAD."""
        height, width = depth.shape[:2]
        x1, x2 = max(0, u - 1), min(width, u + 2)
        y1, y2 = max(0, v - 1), min(height, v + 2)
        values_m = depth[y1:y2, x1:x2].astype(np.float64).ravel()
        values_m *= depth_scale
        values_m = values_m[
            np.isfinite(values_m)
            & (values_m >= 0.2)
            & (values_m <= 4.0)
        ]
        if values_m.size == 0:
            return None

        median = float(np.median(values_m))
        deviations = np.abs(values_m - median)
        mad = float(np.median(deviations))
        if mad > 0.0:
            values_m = values_m[deviations <= 3.0 * mad]
        if values_m.size == 0:
            return None
        return float(np.median(values_m))

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._frame_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._hands.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
