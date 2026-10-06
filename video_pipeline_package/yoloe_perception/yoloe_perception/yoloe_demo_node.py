"""YOLOE RGB-D baseline dedicated to human-demonstration experiments.

The processing pipeline is intentionally inherited unchanged from the stable
realtime node.  Only the ROS node name and output namespace are different, so
future demonstration-specific outputs can be added without colliding with the
realtime topics.
"""

from __future__ import annotations

from collections import deque

import rclpy
import numpy as np

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from tf2_geometry_msgs import do_transform_point
from yoloe_perception.yoloe_depth_node import YoloEDepthNode


class YoloEDemoNode(YoloEDepthNode):
    def __init__(self) -> None:
        super().__init__(
            node_name="yoloe_demo_node",
            output_namespace="/demo",
        )
        self.bottle_top_publisher = self.create_publisher(
            PointStamped,
            "/demo/bottle_top",
            10,
        )
        self.bottle_bottom_publisher = self.create_publisher(
            PointStamped,
            "/demo/bottle_bottom",
            10,
        )
        self.bottle_length_publisher = self.create_publisher(
            Float32,
            "/demo/bottle_length",
            10,
        )
        self.filtered_top: np.ndarray | None = None
        self.filtered_bottom: np.ndarray | None = None
        self.length_history: deque[float] = deque(maxlen=15)
        self.get_logger().info(
            "YOLOE demonstration baseline started on /demo"
        )

    def publish_additional_geometry(
        self,
        object_points: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        header,
        transform,
    ) -> None:
        """Estimate robust bottle endpoints and publish them in camera_link."""
        if object_points.shape[0] < 100:
            return

        unit_axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(unit_axis))
        if norm <= 1e-9:
            return
        unit_axis /= norm

        projections = (
            object_points.astype(np.float64, copy=False) - center
        ) @ unit_axis
        projections = projections[np.isfinite(projections)]
        if projections.size < 100:
            return

        # Giữ nhiều điểm ở nắp/đáy hơn, nhưng vẫn loại 1% điểm cực trị ở
        # mỗi phía để một vài điểm depth nhiễu không kéo dài chai bất thường.
        bottom_s, top_s = np.percentile(projections, (1.0, 99.0))
        if top_s <= bottom_s:
            return

        top = center + float(top_s) * unit_axis
        bottom = center + float(bottom_s) * unit_axis

        alpha = self.pose_filter_alpha
        if self.filtered_top is not None and self.filtered_bottom is not None:
            top = alpha * top + (1.0 - alpha) * self.filtered_top
            bottom = alpha * bottom + (1.0 - alpha) * self.filtered_bottom

        self.filtered_top = top
        self.filtered_bottom = bottom

        top_optical = self._make_point_stamped(top, header)
        bottom_optical = self._make_point_stamped(bottom, header)
        top_camera_link = do_transform_point(top_optical, transform)
        bottom_camera_link = do_transform_point(bottom_optical, transform)

        self.bottle_top_publisher.publish(top_camera_link)
        self.bottle_bottom_publisher.publish(bottom_camera_link)

        current_length_m = float(np.linalg.norm(top - bottom))
        self.length_history.append(current_length_m)

        length = Float32()
        length.data = float(np.median(self.length_history))
        self.bottle_length_publisher.publish(length)

    @staticmethod
    def _make_point_stamped(point: np.ndarray, header) -> PointStamped:
        message = PointStamped()
        message.header = header
        message.point.x = float(point[0])
        message.point.y = float(point[1])
        message.point.z = float(point[2])
        return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloEDemoNode()

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
