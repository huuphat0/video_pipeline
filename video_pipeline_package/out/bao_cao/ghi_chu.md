## Cấu trúc thư mục `video_pipeline/`

| Thư mục | File | Chức năng |
|---|---|---|
| `data_io/` | `mcap_to_mp4.py`, `depth_from_mcap.py`, `dexycb_to_video.py` | Chuyển dữ liệu vào |
| `perception/` | `contact_point.py`, `object_pose.py`, `hand_pose_masked.py` | Đo tay / vật |
| `skill_inference/` | `hoi_skill_inference.py`, `subskill_inference.py`, `skill_params.py`, `skills_lowlevel.json` | Suy luận skill từ mask |
| `vlm/` | `segment_and_overlay_vi_v3.py`, `infer_task_name.py`, 2 file `.md` | Pipeline VLM |
| `visualization/` | `render_hoi_video.py`, `timeline_overlay.py`, `grasp_card.py` | Xuất video / ảnh |
| `models/` | `hand_landmarker.task` | Trọng số MediaPipe |
