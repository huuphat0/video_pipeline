# Nhánh B — SAM 2 + DINOv2

SAM 2 sinh mask tự động ở keyframe bằng lưới điểm prompt (**không có text
prompt**) → crop từng mask → DINOv2 nhúng thành vector 768 chiều → gán nhãn →
bbox đưa vào **cùng tracker SAM2 của nhánh A**.

`common.py` là **symlink** tới `../sam_dino/common.py`, không phải bản copy: hai
nhánh phải chạy đúng cùng một tracker thì so sánh mới có nghĩa.

## Môi trường

Venv riêng, tách khỏi venv chung của repo:

```bash
video_pipeline_ai/sam_dinov2/venv/    # torch 2.14.0+cu130, torchvision 0.29.0,
                                      # transformers 5.x, scikit-learn, opencv, matplotlib
```

## Hai chế độ

### 1. Gom cụm — không cần chuẩn bị gì

```bash
venv/bin/python sam_dinov2_video.py ../video_test2.mp4 \
    --n-clusters 6 --save-crops --outdir out_test2
```

DINOv2 gom các mask thành cụm, đặt tên `cluster_0..k`. Xem `clusters.png` để
biết cụm nào là vật gì. `--save-crops` lưu từng crop vào `out_test2/crops/<cụm>/`.

### 2. One-shot — có ảnh mẫu

Chọn vài crop tốt từ bước 1 xếp vào `refs/<tên lớp>/`:

```
refs/bottle/*.png   refs/hand/*.png   refs/cup/*.png
```

```bash
venv/bin/python sam_dinov2_video.py ../video_test2.mp4 \
    --refs refs --outdir out_test2_refs
```

Mask nào cosine < `--sim-threshold` với mọi lớp thì bị **bỏ** (mặc định) — đây
là thứ dọn sạch nền. `--keep-unknown` để giữ lại.

Vòng làm việc tự nhiên: chạy chế độ 1 → nhìn `clusters.png` → nhặt crop vào
`refs/` → chạy chế độ 2.

## Tham số

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--points-per-side` | 16 | lưới 16×16 = 256 điểm prompt. Cao hơn = nhiều mask hơn, chậm hơn |
| `--pred-iou-thresh` | 0.80 | ngưỡng điểm chất lượng mask của SAM |
| `--min-area-frac` / `--max-area-frac` | 0.0015 / 0.15 | lọc mask quá vụn / quá lớn |
| `--nms-iou` | 0.75 | khử mask trùng theo mask-IoU |
| `--max-objects` | 8 | số mask giữ mỗi keyframe |
| `--rank` | `iou` | chọn mask theo điểm chất lượng SAM. `area` (diện tích) dễ trúng tường/sàn |
| `--sim-threshold` | 0.55 | cosine tối thiểu để nhận nhãn ở chế độ `--refs` |
| `--keep-unknown` | tắt | giữ cả mask không khớp lớp nào |
| `--n-clusters` | 6 | số cụm KMeans khi không có `--refs` |

## Đầu ra

- `annotated_sam_dinov2.mp4`
- `masks_sam_dinov2.json` — mask từng frame, RLE
- `embeddings.npz` — `emb (N,768)`, `label`, `conf`, `frame_idx`, `bbox`.
  **Chỉ nhánh B có.** Dùng lại được để tìm mọi lần một vật xuất hiện, đo độ
  giống giữa 2 vật, hoặc gán nhãn về sau mà không phải chạy lại model
- `clusters.png` (chế độ gom cụm) — lưới crop theo cụm
- `crops/<nhãn>/*.png` (khi có `--save-crops`)
- `timing_sam_dinov2.json`

## Kết quả trên video_test2.mp4 (297 frame, 720x405)

**Chế độ gom cụm (k=6):** cụm rất sạch — `cluster_1` = chai (7/7),
`cluster_4` = cốc (7/7), `cluster_3` = tay (7/8); `cluster_0/2/5` = nắp, bàn
phím, dây cáp, tường.

Nhưng ra **21 ID cho 8 vật**. Vật thật giữ ID xuyên suốt; số ID phình ra là do
mask **nền**: mỗi keyframe AMG rơi vào một vùng tường/dây cáp khác nhau, IoU
giữa các chunk quá thấp nên bị cấp ID mới. Đây là giới hạn thật của AMG — nó
không có khái niệm "cùng một vật qua thời gian" — không vá bằng tracker được.
Cách đúng là loại nền ngay từ đầu, tức là chế độ `--refs`.

**Chế độ one-shot (3 ảnh mẫu mỗi lớp, lấy từ chính clusters.png):**

- Giữ 3/8 mask mỗi keyframe, đúng `hand` / `cup` / `bottle`
- **3 object**, ID ổn định, phủ 297/297 frame
- AMG 687 ms/keyframe, DINOv2 86 ms/keyframe, track 148 ms/frame

## So với nhánh A trên cùng video

| | Nhánh A (Grounded-SAM) | Nhánh B (SAM + DINOv2, `--refs`) |
|---|---|---|
| Phải cung cấp | danh sách tên | 3 ảnh mẫu mỗi lớp |
| Object | 3, đúng | 3, đúng |
| Phủ frame | 297/297 | 297/297 |
| Chi phí keyframe | 338 ms | 773 ms (AMG + DINOv2) |
| Sản phẩm phụ | — | `embeddings.npz` |

Khác biệt đáng chú ý về **hình dạng mask**: nhánh A prompt bằng box nên mask
`hand` bám đúng bàn tay; nhánh B để SAM tự cắt nên `hand` gồm cả cẳng tay —
SAM coi đó là một vùng liền. Không cái nào sai, nhưng nếu bạn cần đúng bàn tay
thì nhánh A cho ra thứ bạn muốn mà không phải hậu xử lý.
