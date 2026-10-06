# Tóm tắt source `video_pipeline/`

Pipeline nhận diện skill thao tác tay từ video người (nhánh mask Grounded-SAM,
không dùng VLM) và so sánh/chuyển điểm nắm sang góc camera khác. Đo trên pixel
+ depth, kết quả tất định.

## Cấu trúc

```
video_pipeline/
├── data_io/            đọc dữ liệu vào (.mcap, depth, DexYCB)
├── perception/         hình học: điểm chạm, pose 3D, pose tay (phụ)
├── skill_inference/    suy luận skill + sub-skill + xuất tham số điểm nắm
├── visualization/      render video, timeline, thẻ vật (grasp card)
├── vlm/                pipeline CŨ dùng VLM (không dùng nữa, giữ tham chiếu)
└── models/             trọng số MediaPipe (chỉ cho hand_pose_masked)
```

## Luồng chạy (1 video)

```
.mcap ─ data_io/mcap_to_mp4.py ─────────────→ .mp4 (màu)
      └ data_io/depth_from_mcap.py ─────────→ depth.npz (mm + fx,fy,cx,cy)
                  │
   (sam_dino/gsam2_video.py ở ngoài repo này → masks_gsam2.json)
                  │
skill_inference/hoi_skill_inference.py ─→ skills_lowlevel.json   (nhãn skill theo frame)
skill_inference/subskill_inference.py  ─→ subskills.json         (Approach/Align/Close_Gripper, Orient_Tilt/...)
perception/object_pose.py              ─→ pose 3D từng frame      (center_mm, axis, tilt_deg)
skill_inference/skill_params.py        ─→ grasp_points / grasp_episodes / segments
visualization/render_hoi_video.py      ─→ video có nhãn + chấm điểm chạm + trục 3D
```

## Nhóm 1 — Nhận diện skill (`skill_inference/`)

### `hoi_skill_inference.py` — skill mức thấp
- Đầu vào: `masks_gsam2.json` (+ depth tuỳ chọn).
- Gán nhãn từng frame bằng luật hình học: `Reach, Contact, Grasp, Lift, Place,
  MoveToTarget, Pour, Release, Retract, Idle` (Push/Pull có nhãn nhưng chưa đủ dữ liệu).
- Hàm chính: `frame_features` (đặc trưng từng frame), `pick_active_object` (vật
  đang thao tác), `build_signals` (chuỗi tín hiệu đã làm mịn), `classify_frames`
  (luật gán nhãn), `merge_short_segments` / `to_segments` (gộp thành đoạn).
- Phụ trợ hình học 3D: `backproject_mask_xyz` (mask+depth → đám mây điểm, lọc
  ngoại lai Tukey k=3), `principal_axis_3d` (PCA 3D), `axis_tilt_deg`,
  `pixel_to_xyz_mm` / `project_xyz_to_pixel`, `load_depth`, `load_intrinsics`.

### `subskill_inference.py` — sub-skill
- Tách bên trong `Grasp` → `Approach / Align / Close_Gripper` (dựa khoảng cách
  depth tay–vật, `compute_gap`).
- Tách bên trong `Pour` → `Orient_Tilt / Hold_Pour_Angle / Return_Upright`
  (dựa góc trục chính khỏi tư thế thẳng, `compute_dev`).
- Hàm: `find_grasp_episodes`, `find_pour_episodes`, `split_*_subskills`.

### `skill_params.py` — tham số xuất cho robot
- `grasp_points_per_object`: 1 điểm nắm gộp (trung vị) cho MỖI vật trong video.
- `grasp_episodes`: 1 điểm nắm cho MỖI LẦN CẦM (dùng `validated_contacts`).
- `build` (`--mode segments`): tham số theo từng đoạn skill.
- Mỗi điểm nắm có: `along, part, side` (tương đối với vật — truyền được sang
  video khác), `grasp_point_px` (chỉ đúng cho video này), `grasp_point_xyz_mm`
  (hệ camera, chỉ đúng khi camera đứng yên).
- CLI: `--mode {segments, grasp_points, grasp_episodes}`.

## Nhóm 2 — Hình học điểm chạm và pose (`perception/`)

### `contact_point.py` — điểm nắm trên 1 frame
- `find_contact`: vùng chạm = mask vật ∩ mask tay nới 6px.
- `describe_location` / `_pca_axis`: chiếu điểm chạm lên trục chính vật → `along`
  (0 đáy, 1 đỉnh), `side`, `part`. Vật tròn dùng phương đứng ảnh.
- `apply_grasp_point`: chiều ngược — áp `along/side` đã học lên mask vật MỚI
  → pixel (+ XYZ nếu có depth). Dùng cho bước robot sau này.
- `contact_point_xyz_mm`: điểm trên trục → patch depth 7×7 (median) → XYZ mm.
- `approach_direction`: hướng tay tới vật (chỉ dùng TRƯỚC khi chạm).
- `contact_info`: gộp các bước trên, dùng cho renderer.

### `object_pose.py` — pose vật từng frame
- Xuất `center_mm` (tâm khối), `axis` (hướng trục 3D), `tilt_deg`,
  `elongation`, `axis_confident` (chỉ tin trục khi elongation ≥ 2.0).
- Thiếu roll quanh trục dài (không ảnh hưởng vật đối xứng như chai/ly).

### `hand_pose_masked.py` — ĐÃ BÁC BỎ
- Đo tư thế ngón bằng MediaPipe. Kết quả không dùng được (tay đeo găng, curl
  không đổi). Giữ làm bằng chứng. Không nằm trong luồng chính.

## Nhóm 3 — So sánh/chuyển điểm nắm (`visualization/grasp_card.py`)

Đây là phần "so sánh thế nắm" giữa 2 góc nhìn — **chưa hoàn chỉnh**.

- Tạo **thẻ vật** cho mỗi lần cầm: ảnh vật lúc CHƯA bị chạm + mask vùng nắm
  + đám mây vùng nắm trong **hệ của vật**.
- Lý do không dùng điểm nắm cũ: px/xyz đổi theo camera, `side` đảo khi camera
  đối diện, `along` chỉ ổn khi camera nhìn ngang.
- Quy trình cho mỗi lần cầm `[a, b)`:
  1. `find_reference_frame` — frame trước `a` mà tay chưa phủ vật.
  2. `hold_window` — các frame vật còn đứng yên tại chỗ cũ.
  3. `occlusion_mask` — vùng bị tay che, bỏ phiếu 3 tín hiệu (mask tay,
     depth 8–80mm, ngoại hình DINO/LAB), cần ≥ 2.
  4. `grasp_bands` — vùng nắm = vùng che ở đủ số frame đã nắm chắc, tách thành
     mảng dọc trục vật.
  5. `object_frame_3d`, `fit_table_plane`, `compute_anchors` — dựng hệ vật
     (mặt bàn, đáy B, đỉnh T).
  6. `measure_view` + `resolve_grasp` — đo lại ở góc mới, kiểm tra chéo B–T,
     trả vị trí mảng nắm.
  7. `dino_features`, `draw_anchors`, `render_preview`, `build_card` — DINOv2
     và ảnh xem trước.
- CLI: `list_episodes`, `main`.

**Còn thiếu**: docstring nhắc `grasp_transfer.py` (so khớp cả vật sang góc
camera khác rồi chuyển nhãn vùng nắm). File này **chưa có trong repo**. Hiện
chỉ có bước tạo thẻ.

## Nhóm 4 — Đầu vào/ra và hiển thị

| File | Chức năng |
|---|---|
| `data_io/mcap_to_mp4.py` | Rút luồng màu ROS2 bag (.mcap) thành .mp4 |
| `data_io/depth_from_mcap.py` | Rút depth căn theo frame màu (ghép theo timestamp), lưu fx,fy,cx,cy |
| `data_io/dexycb_to_video.py` | Chuyển dữ liệu DexYCB thành video + depth |
| `visualization/render_hoi_video.py` | Vẽ mask, nhãn skill, HUD, chấm điểm chạm, trục 3D, thanh timeline; `--dump-contacts` xuất điểm chạm từng frame |
| `visualization/timeline_overlay.py` | Thêm thanh timeline vào video đã render (hậu xử lý) |

## Nhóm 5 — Pipeline cũ (`vlm/`, không dùng)

`segment_and_overlay_vi_v3.py`, `infer_task_name.py`, 2 file `.md`: pipeline
dùng VLM (Ollama) + MediaPipe. Đã thay bằng nhánh mask. Giữ làm tham chiếu.

## Phụ thuộc bên ngoài repo

- `sam_dino/gsam2_video.py`, `sam_dino/merge_object_ids.py` — sinh
  `masks_gsam2.json` (Grounding DINO + SAM2). Nằm ngoài `github/`.
- `sam_dino/gsam2_frame.py` — bản 1 frame, dùng cho `apply_grasp_point`.
- Môi trường có `torch`, `transformers`, `mcap`, `mcap-ros2-support`, `cv2`.
