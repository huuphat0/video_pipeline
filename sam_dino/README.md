# Nhánh A — Grounded-SAM (Grounding DINO + SAM 2.1)

Text prompt → Grounding DINO ra box ở keyframe → SAM 2.1 lan truyền mask cho
mọi frame. Không train gì, hoàn toàn zero-shot.

## Chạy

```bash
cd video_pipeline_ai/sam_dino
../../venv/bin/python gsam2_video.py ../video_test2.mp4 \
    --prompt "hand . plastic water bottle . glass cup ." \
    --interval 1.5 --outdir out_gsam2_test2
```

Tham số đáng chỉnh:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--prompt` | (bắt buộc) | danh từ viết thường, ngăn bằng ` . `. Vật không có trong prompt = vô hình |
| `--interval` | 1.5 | giây giữa 2 lần chạy Grounding DINO. Nhỏ hơn = bám tốt hơn khi cảnh đổi nhanh, chậm hơn |
| `--box-threshold` | 0.30 | hạ xuống nếu bị sót vật; tăng lên nếu nhiều box rác |
| `--text-threshold` | 0.25 | ngưỡng gán cụm từ cho box |
| `--dedup-iou` | 0.55 | cùng nhãn, IoU trên ngưỡng này = trùng, giữ score cao hơn |
| `--dedup-contain` | 0.80 | cùng nhãn, box nằm gọn trong box đã giữ quá tỉ lệ này thì bỏ |
| `--max-side` | 720 | thu nhỏ cạnh dài. Tăng lên 1080 nếu vật nhỏ bị sót |
| `--max-seconds` | — | cắt ngắn video để thử nhanh |

## Đầu ra (trong `--outdir`)

- `annotated_gsam2.mp4` — mask màu + box + nhãn `label #id score`
- `masks_gsam2.json` — mask từng frame, nén RLE (giải bằng `common.decode_rle`)
- `timing_gsam2.json` — ms/keyframe, ms/frame, fps tổng

## Cơ chế tracking

Video được chia thành chunk giữa 2 keyframe. Mỗi chunk mở một SAM2 video
session riêng, nạp box ở frame đầu chunk rồi propagate tới hết chunk. ID toàn
cục được nối giữa các chunk qua hai vòng:

1. cùng nhãn + IoU bbox >= 0.5 tại frame giao nhau;
2. detection còn thừa, nếu nhãn đó chỉ còn đúng MỘT object cũ chưa dùng thì nối luôn.

Vòng 2 là cần thiết: khi vật bị tay che hoặc xoay mạnh giữa 2 keyframe, IoU tụt
dưới ngưỡng và nếu không có nó thì cùng một vật bị tách thành 2 ID.

Tracker nhớ box cuối của **mọi** object đã thấy, không chỉ chunk liền trước —
một keyframe sót detection thì chunk sau vẫn nối lại được ID cũ.

`common.py` cũng được nhánh B dùng (`../sam_dinov2/common.py` là symlink tới
file này), qua cờ `link_by_label`. Nhánh A nối ID theo nhãn + IoU; nhánh B ở chế
độ gom cụm nối chỉ theo IoU, vì tên cụm không phân biệt được vật.

Khử box trùng ở keyframe xét cả IoU lẫn **mức độ bao hàm** (`box_containment`),
vì Grounding DINO hay ra nhiều box lồng nhau cho cùng một vật với IoU chỉ ~0.5 —
không đủ để IoU đơn thuần bắt được.

## Kết quả trên video_test2.mp4 (cảnh rót nước, 297 frame, 720x405)

- 3 object: `#0 plastic water bottle`, `#1 hand`, `#2 glass cup` — đúng số vật thật
- ID ổn định qua cả 7 chunk
- Phủ 297/297 frame (100%)
- Detect 337 ms/keyframe, track 152 ms/frame, tổng 47.4s → **6.3 fps end-to-end**
- Ở keyframe 180 chai ra 3 box chồng nhau (score 0.44 / 0.30 / 0.29); hai bước
  trên khử hết mà không phải nâng `--box-threshold` — điều quan trọng, vì box
  thật của chai lúc bị tay che chỉ có score 0.44.

## Phụ thuộc

`transformers>=5`, `torch`, `torchvision`, `opencv-python`, `numpy`.
Weights tự tải về HF cache: `IDEA-Research/grounding-dino-base` (1.8G),
`facebook/sam2.1-hiera-large` (857M).
