from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointField
from sensor_msgs_py import point_cloud2
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLOE
from visualization_msgs.msg import Marker


class YoloEDepthNode(Node):
    def __init__(
        self,
        node_name: str = "yoloe_depth_node",
        output_namespace: str = "/yoloe",
    ) -> None:
        super().__init__(node_name)
        self.output_namespace = output_namespace.rstrip("/")

        self.bridge = CvBridge()
        self.device = 0 if torch.cuda.is_available() else "cpu"

        self.model = YOLOE("yoloe-26s-seg.pt")
        self.model.set_classes(["bottle"])

        self.latest_color: Optional[np.ndarray] = None
        self.latest_color_header = None
        self.latest_depth: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.profile_count = 0
        self.profile_sums = {
            "inference": 0.0,
            "gpu_to_cpu": 0.0,
            "mask_cloud": 0.0,
            "pca": 0.0,
            "render_publish": 0.0,
            "total": 0.0,
        }

        self.depth_scale_to_m = 0.001
        self.filtered_center: Optional[np.ndarray] = None
        self.filtered_surface_center: Optional[np.ndarray] = None
        self.filtered_bottle_axis: Optional[np.ndarray] = None
        # Lọc nhiễu pose (không phải tracking): 90% mới, 10% trạng thái cũ.
        self.pose_filter_alpha = 0.9

        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_warning_reported = False

        self.publisher = self.create_publisher(
            Image,
            f"{self.output_namespace}/annotated_image",
            qos_profile_sensor_data,
        )
        self.pointcloud_publisher = self.create_publisher(
            point_cloud2.PointCloud2,
            f"{self.output_namespace}/bottle_points",
            qos_profile_sensor_data,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            f"{self.output_namespace}/object_pose",
            qos_profile_sensor_data,
        )
        self.camera_link_pose_publisher = self.create_publisher(
            PoseStamped,
            f"{self.output_namespace}/object_pose_camera_link",
            qos_profile_sensor_data,
        )
        self.pose_text_publisher = self.create_publisher(
            Marker,
            f"{self.output_namespace}/object_pose_text",
            qos_profile_sensor_data,
        )
        self.surface_center_publisher = self.create_publisher(
            PointStamped,
            f"{self.output_namespace}/surface_center",
            qos_profile_sensor_data,
        )

        # Một PointCloud2 chứa mọi vật trong frame. `instance_id` dùng để
        # tách từng vật, còn `rgb` giúp RViz hiển thị mỗi vật bằng một màu.
        self.pointcloud_fields = [
            PointField(
                name="x", offset=0, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="y", offset=4, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="z", offset=8, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="rgb", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="instance_id",
                offset=16,
                datatype=PointField.UINT32,
                count=1,
            ),
        ]
        self.pointcloud_dtype = point_cloud2.dtype_from_fields(
            self.pointcloud_fields
        )
        self.instance_colors_rgb = (
            (255, 64, 64),
            (64, 255, 64),
            (64, 128, 255),
            (255, 192, 64),
            (192, 64, 255),
            (64, 255, 255),
            (255, 64, 192),
            (160, 255, 64),
        )

        self.color_subscriber = Subscriber(
            self,
            Image,
            "/camera/camera/color/image_raw",
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_subscriber = Subscriber(
            self,
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            qos_profile=qos_profile_sensor_data,
        )

        # Chỉ ghép các frame RGB và depth có timestamp gần nhau (tối đa 50 ms).
        self.rgbd_sync = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            queue_size=10,
            slop=0.05,
        )
        self.rgbd_sync.registerCallback(self.rgbd_callback)

        self.create_subscription(

            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,

            qos_profile_sensor_data,
        )


        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,

        )
        self.worker.start()

        self.get_logger().info("Simple YOLOE depth node started")

    def rgbd_callback(self, color_msg: Image, depth_msg: Image) -> None:
        """Nhận một cặp RGB-depth đã được ghép gần nhau theo timestamp."""
        try:
            frame = self.bridge.imgmsg_to_cv2(
                color_msg,
                desired_encoding="bgr8",
            )
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            with self.lock:
                self.latest_color = frame
                self.latest_color_header = color_msg.header
                self.latest_depth = depth

                if depth_msg.encoding == "32FC1":
                    self.depth_scale_to_m = 1.0
                else:
                    self.depth_scale_to_m = 0.001

            self.new_frame_event.set()

        except Exception as error:
            self.get_logger().error(
                f"RGB-D conversion error: {error}"
            )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()

            if self.stop_event.is_set():
                break

            with self.lock:
                if (
                    self.latest_color is None
                    or self.latest_depth is None
                ):
                    continue

                frame = self.latest_color
                color_header = self.latest_color_header
                depth = self.latest_depth
                depth_scale = self.depth_scale_to_m

            if self.camera_info is None:
                continue

            self.process_frame(
                frame,
                depth,
                depth_scale,
                color_header,
            )

    def process_frame(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        depth_scale: float,
        color_header,
    ) -> None:
        total_start = time.perf_counter()
        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                imgsz=320,
                conf=0.30,
                device=self.device,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - total_start) * 1000.0

        display = frame.copy()
        result = results[0]
        cloud_chunks = []
        pose_candidate = None
        gpu_to_cpu_ms = 0.0
        mask_cloud_ms = 0.0
        pca_ms = 0.0

        if (
            result.boxes is not None
            and len(result.boxes) > 0
            and result.masks is not None
        ):
            transfer_start = time.perf_counter()
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy()
            gpu_to_cpu_ms = (
                time.perf_counter() - transfer_start
            ) * 1000.0

            # Mức 1: ID là thứ tự detection trong frame hiện tại. ID này có
            # thể đổi giữa các frame; tracking ổn định sẽ là bước tiếp theo.
            detection_count = min(len(boxes), len(masks))
            for instance_id in range(detection_count):
                box = boxes[instance_id]
                confidence = float(confidences[instance_id])
                class_id = int(class_ids[instance_id])
                object_mask = cv2.resize(
                    masks[instance_id],
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5

                x1, y1, x2, y2 = box.astype(int)
                height, width = frame.shape[:2]
                x1 = int(np.clip(x1, 0, width - 1))
                x2 = int(np.clip(x2, 0, width))
                y1 = int(np.clip(y1, 0, height - 1))
                y2 = int(np.clip(y2, 0, height))
                if x1 >= x2 or y1 >= y2:
                    continue

                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                class_name = result.names.get(
                    class_id,
                    str(class_id),
                )
                color_rgb = self.instance_colors_rgb[
                    instance_id % len(self.instance_colors_rgb)
                ]
                color_bgr = tuple(reversed(color_rgb))

                if object_mask.shape == display.shape[:2]:
                    original_pixels = display[object_mask].astype(
                        np.float32,
                        copy=False,
                    )
                    display[object_mask] = (
                        0.65 * original_pixels
                        + 0.35 * np.asarray(color_bgr, dtype=np.float32)
                    ).astype(np.uint8)

                z_front_m = self.estimate_depth_at_center(
                    depth=depth,
                    u=u,
                    v=v,
                    depth_scale=depth_scale,
                )

                if z_front_m is None:
                    label = (
                        f"ID={instance_id} {class_name} "
                        f"{confidence:.2f} XYZ=N/A"
                    )
                else:
                    x_front_m, y_front_m = self.pixel_to_xy(
                        u,
                        v,
                        z_front_m,
                    )
                    surface_center_xyz = np.asarray(
                        (x_front_m, y_front_m, z_front_m),
                        dtype=np.float64,
                    )
                    radius_m = self.estimate_radius_from_mask(
                        object_mask=object_mask,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        z_front_m=z_front_m,
                    )
                    z_center_m = z_front_m
                    if radius_m is not None:
                        z_center_m += radius_m

                    x_m, y_m = self.pixel_to_xy(
                        u,
                        v,
                        z_center_m,
                    )
                    center_xyz = np.asarray(
                        (x_m, y_m, z_center_m),
                        dtype=np.float64,
                    )
                    label = (
                        f"ID={instance_id} {class_name} {confidence:.2f} "
                        f"X={x_m:.3f} "
                        f"Y={y_m:.3f} "
                        f"Z={z_center_m:.3f}m"
                    )
                    cloud_start = time.perf_counter()
                    object_points = self.points_from_mask(
                        depth=depth,
                        object_mask=object_mask,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        reference_z_m=z_front_m,
                        depth_scale=depth_scale,
                    )
                    mask_cloud_ms += (
                        time.perf_counter() - cloud_start
                    ) * 1000.0
                    if object_points.size > 0:
                        cloud_chunks.append(
                            self.make_colored_cloud_chunk(
                                points=object_points,
                                instance_id=instance_id,
                                color_rgb=color_rgb,
                            )
                        )
                        pca_start = time.perf_counter()
                        pca_result = self.estimate_bottle_axis(object_points)
                        pca_ms += (
                            time.perf_counter() - pca_start
                        ) * 1000.0
                        if pca_result is not None:
                            bottle_axis, axis_confidence = pca_result
                            candidate = (
                                confidence,
                                surface_center_xyz,
                                center_xyz,
                                bottle_axis,
                                axis_confidence,
                                object_points,
                            )
                            if (
                                pose_candidate is None
                                or confidence > pose_candidate[0]
                            ):
                                pose_candidate = candidate

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color_bgr,
                    2,
                )
                cv2.circle(display, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2,
                )

        publish_start = time.perf_counter()
        output = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        output.header = color_header
        self.publisher.publish(output)

        if cloud_chunks:
            cloud_points = np.concatenate(cloud_chunks, axis=0)
        else:
            cloud_points = np.empty(0, dtype=self.pointcloud_dtype)

        cloud = point_cloud2.create_cloud(
            color_header,
            self.pointcloud_fields,
            cloud_points,
        )
        self.pointcloud_publisher.publish(cloud)

        if pose_candidate is not None:
            (
                _,
                surface_center_xyz,
                center_xyz,
                bottle_axis,
                _,
                object_points,
            ) = pose_candidate
            self.publish_filtered_pose(
                surface_center_xyz=surface_center_xyz,
                center_xyz=center_xyz,
                bottle_axis=bottle_axis,
                object_points=object_points,
                header=color_header,
            )
        render_publish_ms = (
            time.perf_counter() - publish_start
        ) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        self.record_profile(
            inference=inference_ms,
            gpu_to_cpu=gpu_to_cpu_ms,
            mask_cloud=mask_cloud_ms,
            pca=pca_ms,
            render_publish=render_publish_ms,
            total=total_ms,
        )

    def record_profile(self, **measurements: float) -> None:
        """Log thời gian trung bình mỗi 30 frame để tránh spam terminal."""
        self.profile_count += 1
        for name, value in measurements.items():
            self.profile_sums[name] += value

        if self.profile_count < 30:
            return

        averages = {
            name: value / self.profile_count
            for name, value in self.profile_sums.items()
        }
        processing_fps = 1000.0 / max(averages["total"], 1e-6)
        self.get_logger().info(
            "PROFILE avg/30: "
            f"inference={averages['inference']:.1f}ms, "
            f"gpu_to_cpu={averages['gpu_to_cpu']:.1f}ms, "
            f"mask_cloud={averages['mask_cloud']:.1f}ms, "
            f"pca={averages['pca']:.1f}ms, "
            f"render_publish={averages['render_publish']:.1f}ms, "
            f"total={averages['total']:.1f}ms, "
            f"processing_fps={processing_fps:.1f}"
        )
        self.profile_count = 0
        for name in self.profile_sums:
            self.profile_sums[name] = 0.0

    def estimate_radius_from_mask(
        self,
        object_mask: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        z_front_m: float,
    ) -> Optional[float]:
        """Ước lượng bán kính chai từ median bề rộng mask vùng thân."""
        box_height = y2 - y1
        if box_height < 5:
            return None

        # Tránh vùng nắp và đáy; lấy nhiều lát ngang ở phần thân chai.
        body_top = y1 + int(0.35 * box_height)
        body_bottom = y1 + int(0.75 * box_height)
        row_widths = []
        for row in range(body_top, body_bottom):
            columns = np.flatnonzero(object_mask[row, x1:x2])
            if columns.size >= 2:
                row_widths.append(int(columns[-1] - columns[0] + 1))

        if len(row_widths) < 5:
            return None

        width_px = float(np.median(row_widths))
        fx = float(self.camera_info.k[0])
        if fx <= 0.0:
            return None

        diameter_m = width_px * z_front_m / fx
        # Sanity gate cho vật dạng chai; nếu mask tạo kích thước vô lý thì
        # giữ tâm bề mặt thay vì hiệu chỉnh sai.
        if not 0.02 <= diameter_m <= 0.20:
            return None
        return 0.5 * diameter_m

    @staticmethod
    def estimate_bottle_axis(
        points: np.ndarray,
    ) -> Optional[tuple[np.ndarray, float]]:
        """Lấy trục dài PCA nếu cloud đủ điểm và đủ tính kéo dài."""
        if points.shape[0] < 100:
            return None

        lower, upper = np.percentile(points, (2.0, 98.0), axis=0)
        trimmed = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if trimmed.shape[0] < 100:
            return None

        # Cloud vẫn publish đầy đủ; riêng PCA chỉ cần một mẫu phân bố đều.
        # Giới hạn số điểm giúp giảm covariance/eigendecomposition latency.
        max_pca_points = 2500
        if trimmed.shape[0] > max_pca_points:
            sample_indices = np.linspace(
                0,
                trimmed.shape[0] - 1,
                max_pca_points,
                dtype=np.int32,
            )
            trimmed = trimmed[sample_indices]

        centered = trimmed.astype(np.float64) - np.mean(trimmed, axis=0)
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        bottle_axis = eigenvectors[:, order[0]]

        axis_confidence = float(
            eigenvalues[0] / max(float(eigenvalues[1]), 1e-12)
        )
        if axis_confidence < 1.5:
            return None

        bottle_axis /= np.linalg.norm(bottle_axis)
        return bottle_axis, axis_confidence

    def publish_filtered_pose(
        self,
        surface_center_xyz: np.ndarray,
        center_xyz: np.ndarray,
        bottle_axis: np.ndarray,
        object_points: np.ndarray,
        header,
    ) -> None:
        """Lọc tâm/trục PCA, dựng hệ trục vật và publish PoseStamped."""
        axis = bottle_axis.astype(np.float64, copy=True)
        axis /= np.linalg.norm(axis)

        if self.filtered_bottle_axis is not None:
            # PCA không phân biệt a và -a; chọn dấu gần frame trước nhất.
            if np.dot(axis, self.filtered_bottle_axis) < 0.0:
                axis = -axis
            alpha = self.pose_filter_alpha
            axis = alpha * axis + (1.0 - alpha) * self.filtered_bottle_axis
            axis /= np.linalg.norm(axis)
            center = (
                alpha * center_xyz
                + (1.0 - alpha) * self.filtered_center
            )
            surface_center = (
                alpha * surface_center_xyz
                + (1.0 - alpha) * self.filtered_surface_center
            )
        else:
            # Optical frame có Y hướng xuống; quy ước trục chai hướng lên ảnh.
            if axis[1] > 0.0:
                axis = -axis
            center = center_xyz.astype(np.float64, copy=True)
            surface_center = surface_center_xyz.astype(
                np.float64,
                copy=True,
            )

        self.filtered_bottle_axis = axis
        self.filtered_center = center
        self.filtered_surface_center = surface_center

        object_z = axis
        reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(np.dot(reference, object_z))) > 0.95:
            reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

        object_x = np.cross(reference, object_z)
        object_x /= np.linalg.norm(object_x)
        object_y = np.cross(object_z, object_x)
        object_y /= np.linalg.norm(object_y)
        rotation_matrix = np.column_stack((object_x, object_y, object_z))
        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rotation_matrix)

        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = float(center[0])
        pose.pose.position.y = float(center[1])
        pose.pose.position.z = float(center[2])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        surface_point = PointStamped()
        surface_point.header = header
        surface_point.point.x = float(surface_center[0])
        surface_point.point.y = float(surface_center[1])
        surface_point.point.z = float(surface_center[2])
        self.surface_center_publisher.publish(surface_point)
        self.pose_publisher.publish(pose)

        try:
            transform = self.tf_buffer.lookup_transform(
                "camera_link",
                pose.header.frame_id,
                Time.from_msg(pose.header.stamp),
            )
            camera_link_pose = do_transform_pose_stamped(pose, transform)
            self.camera_link_pose_publisher.publish(camera_link_pose)
            self.publish_pose_text(camera_link_pose)
            self.publish_additional_geometry(
                object_points=object_points,
                center=center,
                axis=axis,
                header=header,
                transform=transform,
            )
            self.tf_warning_reported = False
        except TransformException as error:
            if not self.tf_warning_reported:
                self.get_logger().warning(
                    "Cannot transform object pose to camera_link: "
                    f"{error}"
                )
                self.tf_warning_reported = True

    def publish_additional_geometry(
        self,
        object_points: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        header,
        transform,
    ) -> None:
        """Extension hook for nodes that need extra object geometry."""
        return None

    def publish_pose_text(self, pose: PoseStamped) -> None:
        """Hiển thị quaternion camera_link cạnh hệ trục Pose trong RViz."""
        quaternion = pose.pose.orientation
        marker = Marker()
        marker.header = pose.header
        marker.ns = "object_pose_quaternion"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = pose.pose.position.x
        marker.pose.position.y = pose.pose.position.y
        marker.pose.position.z = pose.pose.position.z + 0.08
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.025
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = (
            "camera_link quaternion\n"
            f"qx={quaternion.x:+.3f}  qy={quaternion.y:+.3f}\n"
            f"qz={quaternion.z:+.3f}  qw={quaternion.w:+.3f}"
        )
        self.pose_text_publisher.publish(marker)

    @staticmethod
    def rotation_matrix_to_quaternion(
        rotation_matrix: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Đổi ma trận quay 3x3 thành quaternion theo thứ tự ROS xyzw."""
        matrix = np.asarray(rotation_matrix, dtype=np.float64)
        trace = float(np.trace(matrix))

        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                scale = 2.0 * np.sqrt(
                    1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
                )
                qw = (matrix[2, 1] - matrix[1, 2]) / scale
                qx = 0.25 * scale
                qy = (matrix[0, 1] + matrix[1, 0]) / scale
                qz = (matrix[0, 2] + matrix[2, 0]) / scale
            elif index == 1:
                scale = 2.0 * np.sqrt(
                    1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
                )
                qw = (matrix[0, 2] - matrix[2, 0]) / scale
                qx = (matrix[0, 1] + matrix[1, 0]) / scale
                qy = 0.25 * scale
                qz = (matrix[1, 2] + matrix[2, 1]) / scale
            else:
                scale = 2.0 * np.sqrt(
                    1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
                )
                qw = (matrix[1, 0] - matrix[0, 1]) / scale
                qx = (matrix[0, 2] + matrix[2, 0]) / scale
                qy = (matrix[1, 2] + matrix[2, 1]) / scale
                qz = 0.25 * scale

        quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        return tuple(float(value) for value in quaternion)

    def make_colored_cloud_chunk(
        self,
        points: np.ndarray,
        instance_id: int,
        color_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        """Đóng gói XYZ, màu RGB và ID của một detection."""
        chunk = np.empty(points.shape[0], dtype=self.pointcloud_dtype)
        chunk["x"] = points[:, 0]
        chunk["y"] = points[:, 1]
        chunk["z"] = points[:, 2]

        red, green, blue = color_rgb
        packed_rgb = (red << 16) | (green << 8) | blue
        rgb_as_float = np.asarray(
            [packed_rgb], dtype=np.uint32
        ).view(np.float32)[0]
        chunk["rgb"] = rgb_as_float
        chunk["instance_id"] = instance_id
        return chunk

    def estimate_depth_at_center(
        self,
        depth: np.ndarray,
        u: int,
        v: int,
        depth_scale: float,
        window_size: int = 3,
    ) -> Optional[float]:
        """Ước lượng Z bằng median/MAD trong vùng nhỏ quanh tâm bbox."""
        radius = window_size // 2
        height, width = depth.shape[:2]

        x1 = max(0, u - radius)
        x2 = min(width, u + radius + 1)
        y1 = max(0, v - radius)
        y2 = min(height, v + radius + 1)

        if x1 >= x2 or y1 >= y2:
            return None

        values_m = (
            depth[y1:y2, x1:x2]
            .astype(np.float32, copy=False)
            .reshape(-1)
            * depth_scale
        )
        values_m = values_m[
            np.isfinite(values_m)
            & (values_m >= 0.2)
            & (values_m <= 4.0)
        ]

        if values_m.size < 3:
            return None

        median_m = float(np.median(values_m))
        deviations_m = np.abs(values_m - median_m)
        mad_m = float(np.median(deviations_m))

        # 1.4826 * MAD xấp xỉ độ lệch chuẩn khi nhiễu gần phân phối chuẩn.
        robust_sigma_m = 1.4826 * mad_m
        threshold_m = float(np.clip(3.0 * robust_sigma_m, 0.005, 0.05))
        filtered_m = values_m[deviations_m <= threshold_m]

        if filtered_m.size < 3:
            return None

        return float(np.median(filtered_m))

    def points_from_mask(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        reference_z_m: float,
        depth_scale: float,
        stride: int = 2,
    ) -> np.ndarray:
        """Tạo point cloud từ mask chai và lọc depth bằng median/MAD."""
        height, width = depth.shape[:2]
        if object_mask.shape != (height, width):
            return np.empty((0, 3), dtype=np.float32)

        left = max(0, min(x1, width - 1))
        right = max(0, min(x2, width))
        top = max(0, min(y1, height - 1))
        bottom = max(0, min(y2, height))
        if left >= right or top >= bottom:
            return np.empty((0, 3), dtype=np.float32)

        # Quét mọi pixel trong bbox; chỉ pixel thuộc mask và có depth hợp lệ
        # mới được chuyển thành điểm 3D.
        u_values = np.arange(left, right, stride, dtype=np.int32)
        v_values = np.arange(top, bottom, stride, dtype=np.int32)
        u_grid, v_grid = np.meshgrid(u_values, v_values)
        z_grid = (
            depth[v_grid, u_grid].astype(np.float32, copy=False)
            * depth_scale
        )

        sampled_mask = object_mask[v_grid, u_grid]
        initially_valid = (
            sampled_mask
            & np.isfinite(z_grid)
            & (z_grid >= 0.2)
            & (z_grid <= 4.0)
            & (np.abs(z_grid - reference_z_m) <= 0.15)
        )
        values_m = z_grid[initially_valid]
        if values_m.size < 10:
            return np.empty((0, 3), dtype=np.float32)

        median_m = float(np.median(values_m))
        deviations_m = np.abs(values_m - median_m)
        mad_m = float(np.median(deviations_m))
        robust_sigma_m = 1.4826 * mad_m
        threshold_m = float(np.clip(3.0 * robust_sigma_m, 0.01, 0.08))

        valid = initially_valid & (np.abs(z_grid - median_m) <= threshold_m)

        z_m = z_grid[valid]
        u = u_grid[valid].astype(np.float32)
        v = v_grid[valid].astype(np.float32)
        k = self.camera_info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy
        return np.column_stack((x_m, y_m, z_m)).astype(
            np.float32,
            copy=False,
        )

    def pixel_to_xy(
        self,
        u: int,
        v: int,
        z_m: float,
    ) -> tuple[float, float]:
        k = self.camera_info.k

        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy

        return x_m, y_m

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self.new_frame_event.set()

        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = YoloEDepthNode()

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
    
