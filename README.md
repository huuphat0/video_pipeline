## Cài đặt

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ollama + model VLM
ollama pull qwen3-vl:8b-instruct     # nhanh, 6.1 GB
ollama pull qwen2.5vl:32b            # chính xác hơn, 21 GB
```

`video_pipeline/hand_landmarker.task` (trọng số MediaPipe, 7.5 MB) đã kèm sẵn
trong repo nên không cần tải thêm. Font tiếng Việt lấy từ hệ thống
(`DejaVuSans-Bold.ttf`); thiếu font thì overlay tự fallback sang font mặc định
của PIL.

## Chạy

```bash
cd video_pipeline
python3 segment_and_overlay_vi_v3.py /duong/dan/video.mp4 \
    --model qwen3-vl:8b-instruct \
    --out out.mp4 --log log.json --task-log task_log.json
```

Model mặc định trong script là `qwen2.5vl:32b` (nặng, 21 GB) — luôn truyền
`--model` nếu bạn muốn dùng bản 8b nhẹ hơn.

Các cờ hay dùng khác:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--interval` | `1.5` | khoảng cách giữa các mốc lấy mẫu (giây) — nhỏ hơn thì chi tiết hơn nhưng chậm hơn |
| `--skip-task-log` | tắt | bỏ bước suy luận tên task, chỉ xuất timeline |
| `--skills` / `--names-vi` | `../skill_ontology/*.yaml` | đường dẫn ontology, chỉ cần đổi khi chạy ngoài thư mục `video_pipeline/` |

### Mỗi lần chạy sinh ra 3 file

| File | Cờ | Nội dung |
|---|---|---|
| `*.mp4` | `--out` | video gốc + overlay skill / vật thể / vị trí chạm |
| `*.json` timeline | `--log` | từng mốc thời gian: skill, vật thể, bbox, kết quả xác thực MediaPipe |
| `task_log*.json` | `--task-log` | tên task suy luận cho cả video + chuỗi skill đã rút gọn |

`--task-log` **tích luỹ (append)**, không ghi đè: chạy lại cùng video sẽ thêm
một bản ghi mới vào cuối file. Bỏ hẳn bước suy luận task bằng `--skip-task-log`.

Nếu không truyền cờ nào, file mặc định (`annotated_output_vi_v3.mp4`,
`segments_log_vi_v3.json`, `video_task_log.json`) rơi thẳng vào
`video_pipeline/` — nhớ xoá sau khi test.

## Cấu trúc

```
video_pipeline/
  segment_and_overlay_vi_v3.py       # pipeline chính
  infer_task_name.py                 # suy luận tên task từ chuỗi skill
  hand_landmarker.task               # trọng số MediaPipe
  SEGMENT_OVERLAY_V3_CACH_HOAT_DONG.md   # kiến trúc & luồng dữ liệu
  SEGMENT_OVERLAY_V3_IMPROVEMENTS.md     # lịch sử lỗi đã gặp và cách sửa
skill_ontology/
  skills.yaml                        # 16 skill + precondition/effect
  skill_names_vi.yaml                # tên tiếng Việt
```

Tất cả đều cần để chạy, trừ 2 file `.md` là tài liệu thuần.

## Hạn chế đã biết

- **MediaPipe có thể không phát hiện được tay ở một số video** (đã gặp trường
  hợp 0/6 mốc). Khi đó toàn bộ tầng xác thực hình học không chạy, vị trí chạm
  trong log chỉ là chữ của VLM và không đáng tin. Cần mở rộng burst retry hoặc
  hạ ngưỡng detect.
- Với chai, chỉ `nắp` được nhận diện riêng (`BOTTLE_TOUCH_PARTS`); `đáy` bị
  loại bỏ có chủ đích vì bbox vật thể luôn bị cắt cụt tại chỗ tay che, gây
  false positive. Mọi trường hợp không khớp đều mặc định trả `"thân"`.
- Kết quả VLM **không tất định** — chạy lại cùng video có thể ra chữ khác ở
  vài mốc.
- Output dùng codec `mp4v` của OpenCV; nếu không phát được thì convert:
  `ffmpeg -i out.mp4 -c:v libx264 -pix_fmt yuv420p out_h264.mp4`

