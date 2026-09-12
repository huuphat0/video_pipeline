from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthResult:
    """Kết quả ước lượng depth tại một điểm."""

    z_mm: float
    std_mm: float
    mad_mm: float
    robust_sigma_mm: float
    threshold_mm: float

    valid_count: int
    filtered_count: int
    total_count: int

    reliable: bool
    reason: str


class DepthEstimator:
    """
    Ước lượng depth bền vững quanh một pixel bằng vùng 3x3.

    Quy trình:
        1. Lấy vùng 3x3 quanh tâm.
        2. Đổi depth về đơn vị mm.
        3. Bỏ giá trị không hợp lệ.
        4. Kiểm tra số pixel hợp lệ.
        5. Tính median ban đầu.
        6. Tính MAD.
        7. Quy đổi MAD thành robust sigma.
        8. Tạo ngưỡng outlier thích nghi.
        9. Loại outlier.
        10. Lấy median lần hai làm Z.
        11. Tính std sau lọc.
        12. Đánh giá độ tin cậy.
    """

    def __init__(
        self,
        window_size: int = 3,
        min_depth_mm: float = 200.0,
        max_depth_mm: float = 2000.0,
        min_valid_pixels: int = 6,
        min_filtered_pixels: int = 5,
        sigma_factor: float = 3.0,
        min_threshold_mm: float = 5.0,
        max_threshold_mm: float = 30.0,
        max_std_mm: float = 10.0,
    ) -> None:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size phải là số lẻ và >= 3")

        self.window_size = window_size

        # Khoảng depth mà hệ thống cho phép sử dụng.
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm

        # Số pixel tối thiểu cần có trước và sau lọc.
        self.min_valid_pixels = min_valid_pixels
        self.min_filtered_pixels = min_filtered_pixels

        # Hệ số cho quy tắc sigma.
        self.sigma_factor = sigma_factor

        # Giới hạn ngưỡng outlier.
        self.min_threshold_mm = min_threshold_mm
        self.max_threshold_mm = max_threshold_mm

        # Giới hạn độ lệch chuẩn sau lọc.
        self.max_std_mm = max_std_mm

    def estimate(
        self,
        depth_image: np.ndarray,
        u: int,
        v: int,
        depth_scale_to_mm: float = 1.0,
    ) -> Optional[DepthResult]:
        """
        Ước lượng depth quanh pixel (u, v).

        Tham số
        -------
        depth_image:
            Ma trận depth H x W.

        u:
            Tọa độ ngang của pixel.

        v:
            Tọa độ dọc của pixel.

        depth_scale_to_mm:
            Hệ số đổi dữ liệu depth sang mm.

            Ví dụ:
            - ảnh 16UC1 theo mm: scale = 1.0
            - ảnh 32FC1 theo mét: scale = 1000.0

        Trả về
        -------
        DepthResult:
            Kết quả xử lý depth.

        None:
            Tâm nằm quá sát mép ảnh, không lấy đủ vùng xử lý.
        """

        if depth_image.ndim != 2:
            raise ValueError("depth_image phải là ma trận 2 chiều")

        height, width = depth_image.shape
        radius = self.window_size // 2

        # Không đủ vùng 3x3 nếu tâm nằm sát mép ảnh.
        if (
            u < radius
            or u >= width - radius
            or v < radius
            or v >= height - radius
        ):
            return None

        # ----------------------------------------
        # Bước 1: lấy vùng depth quanh tâm bbox
        # ----------------------------------------
        region = depth_image[
            v - radius : v + radius + 1,
            u - radius : u + radius + 1,
        ]

        total_count = int(region.size)

        # Chuyển về float và đơn vị mm.
        values_mm = (
            region.astype(np.float32).reshape(-1)
            * depth_scale_to_mm
        )

        # ----------------------------------------
        # Bước 2: bỏ dữ liệu không hợp lệ
        # ----------------------------------------
        valid_mask = (
            np.isfinite(values_mm)
            & (values_mm > 0.0)
            & (values_mm >= self.min_depth_mm)
            & (values_mm <= self.max_depth_mm)
        )

        valid_values = values_mm[valid_mask]
        valid_count = int(valid_values.size)

        if valid_count < self.min_valid_pixels:
            return DepthResult(
                z_mm=float("nan"),
                std_mm=float("nan"),
                mad_mm=float("nan"),
                robust_sigma_mm=float("nan"),
                threshold_mm=float("nan"),
                valid_count=valid_count,
                filtered_count=0,
                total_count=total_count,
                reliable=False,
                reason="Không đủ pixel depth hợp lệ",
            )

        # ----------------------------------------
        # Bước 3: median ban đầu
        # ----------------------------------------
        median_initial_mm = float(np.median(valid_values))

        # Khoảng cách của từng điểm tới median.
        deviations_mm = np.abs(
            valid_values - median_initial_mm
        )

        # ----------------------------------------
        # Bước 4: MAD
        # ----------------------------------------
        mad_mm = float(np.median(deviations_mm))

        # Robust sigma:
        # 1.4826 là hệ số quy đổi MAD sang mức gần sigma
        # khi nhiễu gần phân phối Gaussian.
        robust_sigma_mm = 1.4826 * mad_mm

        # ----------------------------------------
        # Bước 5: ngưỡng outlier thích nghi
        # ----------------------------------------
        raw_threshold_mm = (
            self.sigma_factor * robust_sigma_mm
        )

        threshold_mm = float(
            np.clip(
                raw_threshold_mm,
                self.min_threshold_mm,
                self.max_threshold_mm,
            )
        )

        # Giữ những điểm nằm đủ gần median.
        filtered_mask = deviations_mm <= threshold_mm
        filtered_values = valid_values[filtered_mask]

        filtered_count = int(filtered_values.size)

        if filtered_count < self.min_filtered_pixels:
            return DepthResult(
                z_mm=float("nan"),
                std_mm=float("nan"),
                mad_mm=mad_mm,
                robust_sigma_mm=robust_sigma_mm,
                threshold_mm=threshold_mm,
                valid_count=valid_count,
                filtered_count=filtered_count,
                total_count=total_count,
                reliable=False,
                reason="Sau lọc còn quá ít pixel",
            )

        # ----------------------------------------
        # Bước 6: Z đại diện sau lọc
        # ----------------------------------------
        z_mm = float(np.median(filtered_values))

        # ----------------------------------------
        # Bước 7: đánh giá độ phân tán sau lọc
        # ----------------------------------------
        std_mm = float(np.std(filtered_values))

        reliable = std_mm <= self.max_std_mm

        if reliable:
            reason = "Depth ổn định"
        else:
            reason = "Depth sau lọc vẫn phân tán lớn"

        return DepthResult(
            z_mm=z_mm,
            std_mm=std_mm,
            mad_mm=mad_mm,
            robust_sigma_mm=robust_sigma_mm,
            threshold_mm=threshold_mm,
            valid_count=valid_count,
            filtered_count=filtered_count,
            total_count=total_count,
            reliable=reliable,
            reason=reason,
        )
