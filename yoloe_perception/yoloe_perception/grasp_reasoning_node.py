from __future__ import annotations

from collections import deque

import numpy as np
import rclpy

from geometry_msgs.msg import Point32, PointStamped, PolygonStamped
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32


class GraspReasoningNode(Node):
    """Ghép hình học chai và palm để suy ra vị trí nắm theo chiều cao."""

    def __init__(self) -> None:
        super().__init__("grasp_reasoning_node")

        self.length_history: deque[float] = deque(maxlen=15)

        self.pose_subscriber = Subscriber(
            self,
            PoseStamped,
            "/yoloe/object_pose_camera_link",
            qos_profile=qos_profile_sensor_data,
        )
        self.length_subscriber = Subscriber(
            self,
            Vector3Stamped,
            "/yoloe/bottle_length_raw",
            qos_profile=qos_profile_sensor_data,
        )
        self.palm_subscriber = Subscriber(
            self,
            PointStamped,
            "/hand/palm_center_camera_link",
            qos_profile=qos_profile_sensor_data,
        )
        self.synchronizer = ApproximateTimeSynchronizer(
            [
                self.pose_subscriber,
                self.length_subscriber,
                self.palm_subscriber,
            ],
            queue_size=30,
            slop=0.10,
        )
        self.synchronizer.registerCallback(self.reasoning_callback)

        self.filtered_length_publisher = self.create_publisher(
            Vector3Stamped,
            "/grasp/bottle_length_filtered",
            qos_profile_sensor_data,
        )
        self.endpoints_publisher = self.create_publisher(
            PolygonStamped,
            "/grasp/bottle_endpoints",
            qos_profile_sensor_data,
        )
        self.ratio_publisher = self.create_publisher(
            Float32,
            "/grasp/height_ratio",
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Grasp reasoning started: median window=15 samples"
        )

    def reasoning_callback(
        self,
        pose: PoseStamped,
        raw_length: Vector3Stamped,
        palm: PointStamped,
    ) -> None:
        if pose.header.frame_id != palm.header.frame_id:
            self.get_logger().warning(
                "Pose and palm frames differ; waiting for camera_link data",
                throttle_duration_sec=2.0,
            )
            return

        length_m = float(raw_length.vector.x)
        if not np.isfinite(length_m) or not 0.05 <= length_m <= 0.50:
            return

        self.length_history.append(length_m)
        filtered_length_m = float(np.median(self.length_history))

        center = np.asarray(
            (
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ),
            dtype=np.float64,
        )
        rotation = self.quaternion_to_rotation_matrix(
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        bottle_axis = rotation[:, 2]
        bottle_axis /= np.linalg.norm(bottle_axis)

        half_length = 0.5 * filtered_length_m
        bottom = center - half_length * bottle_axis
        top = center + half_length * bottle_axis

        palm_xyz = np.asarray(
            (palm.point.x, palm.point.y, palm.point.z),
            dtype=np.float64,
        )
        bottle_vector = top - bottom
        denominator = float(np.dot(bottle_vector, bottle_vector))
        if denominator < 1e-12:
            return
        ratio = float(
            np.clip(
                np.dot(palm_xyz - bottom, bottle_vector) / denominator,
                0.0,
                1.0,
            )
        )

        length_message = Vector3Stamped()
        length_message.header = pose.header
        length_message.vector.x = filtered_length_m
        self.filtered_length_publisher.publish(length_message)

        endpoints = PolygonStamped()
        endpoints.header = pose.header
        endpoints.polygon.points = [
            Point32(x=float(bottom[0]), y=float(bottom[1]), z=float(bottom[2])),
            Point32(x=float(top[0]), y=float(top[1]), z=float(top[2])),
        ]
        self.endpoints_publisher.publish(endpoints)

        ratio_message = Float32()
        ratio_message.data = ratio
        self.ratio_publisher.publish(ratio_message)

    @staticmethod
    def quaternion_to_rotation_matrix(
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> np.ndarray:
        quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        qx, qy, qz, qw = quaternion / norm
        return np.asarray(
            (
                (
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ),
                (
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ),
                (
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ),
            ),
            dtype=np.float64,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspReasoningNode()
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
