# `segment_and_overlay_vi_v3.py` hoạt động như thế nào?

Tài liệu này giải thích **cách chạy của chương trình** (kiến trúc, luồng dữ
liệu, từng hàm làm gì). Nếu bạn muốn biết **vì sao** từng cơ chế được thêm
vào (lịch sử lỗi đã gặp và cách sửa), xem `SEGMENT_OVERLAY_V3_IMPROVEMENTS.md`.

---

## 0. Tóm tắt trong 1 đoạn

Đầu vào là 1 file video người đang thao tác tay với vật thể. Chương trình lấy
mẫu video mỗi `--interval` giây (mặc định 1.5s), với mỗi mốc thời gian hỏi
model thị giác **Qwen2.5-VL** (chạy local qua Ollama) đang diễn ra *skill* gì
và *vật thể* nào, sau đó dùng **MediaPipe Hands** + *object grounding* để xác
minh bằng hình học xem tay có **thật sự chạm** vật hay không và chạm vào **bộ
phận nào**. Kết quả đi qua nhiều bước khử nhiễu rồi được ghi ra 2 thứ: 1 file
JSON timeline và 1 video có chữ overlay.

```
video.mp4
   │
   ├─(1) Lấy mẫu frame ────────────► sample_frames + retry burst + tail peek
   │
   ├─(2) VLM phân loại thô ────────► [(skill, object, touch_location_thô), ...]
   │
   ├─(3) Khử nhiễu 4 lớp ──────────► skill + tên vật thể đã ổn định
   │
   ├─(4) Xác minh chạm (hybrid) ───► vị trí chạm đã kiểm chứng hình học
   │
   ├─(5) Render overlay ───────────► annot_*.mp4
   └─(6) Suy luận tên task ────────► log_*.json + video_task_log.json
```

---

## 1. Hai model được dùng, và vì sao cần cả hai

| Model | Chạy ở đâu | Giỏi việc gì | Dở việc gì |
|---|---|---|---|
| **Qwen2.5-VL 32B** (`MODEL_NAME`, qua Ollama `localhost:11434`) | GPU+CPU, ~29s/request | Hiểu **ngữ nghĩa**: đây là hành động gì, vật này tên gì, vật nằm ở đâu (bbox) | Suy luận **hình học tinh vi**: hay bịa ra "đang cầm nắp chai" dù tay còn cách xa |
| **MediaPipe HandLandmarker** (`hand_landmarker.task`, ~7.6MB) | CPU local, vài chục ms | Định vị **21 điểm** trên bàn tay cực chính xác | Không hiểu ngữ nghĩa — không biết vật cạnh tay là cái gì |

Nguyên tắc xuyên suốt file: **mỗi model chỉ làm đúng việc nó giỏi, còn quyết
định cuối cùng do code tự tính** (đo khoảng cách / chồng lấn bbox), không để
VLM tự phát biểu kết luận bằng lời văn.

---

## 2. Giai đoạn 1 — Lấy mẫu frame (`process_video`, phần đầu)

Đọc tuần tự toàn bộ video 1 lượt và gom 3 loại frame:

| Biến | Là gì | Dùng để làm gì |
|---|---|---|
| `sample_frames` | `[(timestamp, frame)]` mỗi `interval_sec` giây | Mốc chính để phân loại |
| `sample_retry_frames` | 3+ frame ngay sau mỗi mốc (~0.15s) | Thử lại MediaPipe khi frame chính không phát hiện được tay |
| `tail_peek_frame` | 1 frame ở ~0.5s sau mốc **cuối cùng** | Làm "khung sau" cho cửa sổ trượt của mốc cuối |

`tail_peek_frame` tồn tại vì cửa sổ trượt cần đủ 3 khung *trước–giữa–sau*; ở
mốc cuối video không còn khung sau nên VLM dễ nhầm **Lift** (đi lên) với
**Place** (đi xuống) — cả hai trông giống hệt nhau trên ảnh tĩnh.

---

## 3. Giai đoạn 2 — VLM phân loại thô (`classify_window`)

Với mỗi mốc `i`, gửi **1 request duy nhất** cho Qwen2.5-VL gồm **3 ảnh**:
`sample_frames[i-1]`, `sample_frames[i]`, `sample_frames[i+1]` (hoặc
`tail_peek_frame` ở mốc cuối).

Prompt (`build_window_prompt`) yêu cầu 3 trường trong 1 JSON:

```json
{"skill": "Grasp", "object": "cốc", "touch_location": "quai cốc"}
```

- **`skill`** — chỉ được chọn trong danh sách từ `skills.yaml`, **kèm cả phần
  `description` gốc** của từng skill (không chỉ tên trần) để VLM phân biệt
  được các cặp dễ nhầm. Prompt nhấn mạnh phải nhìn **hướng chuyển động** giữa
  3 khung. Nếu không rõ thì trả `"Unknown"` chứ không đoán bừa.
- **`object`** — tên tiếng Việt tự do (không giới hạn danh sách cố định vì vật
  thể nhà bếp quá đa dạng).
- **`touch_location`** — mô tả tự do vị trí chạm. **Trường này chỉ dùng làm
  phương án dự phòng** — đây chính là chỗ VLM hallucination nặng nhất, sẽ được
  thay bằng cơ chế hybrid ở giai đoạn 4.

`parse_skill_object` bóc JSON ra khỏi text (kể cả khi model nói lan man quanh
nó), đối chiếu tên skill với danh sách hợp lệ (không phân biệt hoa thường), và
trả `"Unknown"` nếu VLM bịa tên skill không tồn tại. `temperature=0.1` để giảm
ngẫu nhiên.

> **Quan trọng:** mỗi mốc là 1 lần suy luận **độc lập**, VLM **không có bộ nhớ**
> giữa các mốc. Đây là nguồn gốc của mọi nhiễu mà giai đoạn 3 phải xử lý.

---

## 4. Giai đoạn 3 — Khử nhiễu, chạy đúng 4 bước theo thứ tự

```python
smoothed = smooth_skill_labels(raw_labels)          # 1. skill
smoothed = smooth_isolated_object_noise(smoothed)   # 2. tên vật, mốc lẻ
smoothed = normalize_object_synonyms(smoothed)      # 3. tên vật, đồng nghĩa
smoothed = smooth_object_labels(smoothed)           # 4. tên vật, theo chuỗi
```

| # | Hàm | Xử lý gì |
|---|---|---|
| 1 | `smooth_skill_labels` | 1 mốc có skill khác hẳn cả 2 mốc lân cận (mà 2 mốc đó giống nhau) → sửa theo lân cận |
| 2 | `smooth_isolated_object_noise` | Tên vật thể nhiễu ở **1 mốc lẻ**, không phụ thuộc skill (vd mốc đầu tiên gọi bừa "bóng") |
| 3 | `normalize_object_synonyms` | Ánh xạ **từ đồng nghĩa** về 1 tên chuẩn qua `OBJECT_SYNONYM_GROUPS` (vd "ly"/"cốc thuỷ tinh"/"bình thuỷ tinh" → cùng 1 tên). Độc lập theo mốc nên đúng cả khi vật bị bỏ dở giữa chừng rồi cầm lại |
| 4 | `smooth_object_labels` | Ràng buộc **vật lý theo chuỗi**: trong 1 chuỗi `HOLDING_SKILLS` liên tục, vật không thể tự đổi tên trừ khi mốc kế tiếp cũng xác nhận tên mới. Bên trong còn gọi `_bridge_chain_reversions`: nếu tên đổi rồi **quay lại đúng tên cũ**, cả đoạn giữa được vá về tên ban đầu |

**Vì sao khử nhiễu skill phải chạy TRƯỚC khi tính vị trí chạm?** Vì nếu 1 mốc
được bước 1 sửa thành `Grasp`, nó phải được đưa qua quy trình xác minh chạm ở
giai đoạn 4. Làm ngược lại thì mốc đó giữ nguyên mô tả thô chưa kiểm chứng.

---

## 5. Giai đoạn 4 — Xác minh chạm hybrid (`describe_touch_location_grounded`)

Chỉ chạy cho các mốc có `skill ∈ TOUCH_DISPLAY_SKILLS` (Grasp, Close, Open,
Press, Release, Insert, Remove, Rotate) **hoặc** `skill == "Reach"`. Các skill
khác (Pour, Lift, MoveToTarget...) không tốn request nào.

### Bước 1 — Tay ở đâu? (MediaPipe, không qua VLM)

`detect_fingertips_px_multi([frame_chính] + retry_burst)` thử lần lượt từng
frame cho tới khi phát hiện được tay. Hai chi tiết quan trọng:

- Lấy **cả 21 landmark** (`HAND_CONTACT_LANDMARK_IDS = range(21)`), không chỉ 5
  đầu ngón — khi nắm chặt, đầu ngón cong vào bị che khuất, còn khớp ngón và
  lòng bàn tay vẫn thấy rõ và vẫn sát vật.
- `min_hand_detection_confidence = 0.1` (mặc định 0.5) — đã kiểm chứng: tư thế
  nắm chặt chai ở conf 0.5 và 0.3 đều trả **0 tay**, conf 0.1 mới bắt được mà
  landmark vẫn chính xác.

→ Không frame nào thấy tay: trả `(fallback_text, verified=False)`.

### Bước 2 — Vật thể ở đâu? (VLM grounding)

`locate_object_bbox_px` hỏi VLM trả bbox pixel theo **đúng định dạng
`bbox_2d`** mà Qwen2.5-VL được huấn luyện riêng để xuất ra. Đây là lý do
**bắt buộc phải dùng họ model Qwen2.5-VL** — đổi sang Llama-3.2-Vision sẽ mất
toàn bộ khả năng định vị này.

→ Không định vị được bbox: trả `(fallback_text, verified=False)`.

### Bước 3 — Có chạm không? (code tự tính)

```python
contact_margin_px = max(25, 0.12 * max(rộng_bbox, cao_bbox))
point, dist = closest_point_to_bbox(fingertips, bbox)
if dist > contact_margin_px:  return "", True   # xác minh CHƯA chạm
```

Ngưỡng **co giãn theo kích thước vật thể**, không phải số pixel cố định — vật
to hơn hoặc video độ phân giải cao hơn thì ngưỡng lớn theo tỉ lệ. Nếu quá xa
thì dừng luôn, **không tốn thêm request VLM nào** — chặn hallucination tận gốc
thay vì lọc sau.

### Bước 4 — Chạm vào bộ phận nào? (`classify_touch_region_grounded`)

Thay vì hỏi VLM "bạn đang chạm vào đâu" (đã thất bại nhiều lần), code hỏi VLM
định vị **bbox riêng cho TỪNG bộ phận đặc trưng** của loại vật thể đó:

```
touch_parts_for_object("cốc")      → ["quai", "miệng"]     (CUP_TOUCH_PARTS)
touch_parts_for_object("chai nước") → ["nắp"]              (BOTTLE_TOUCH_PARTS)
```

(Phân loại theo `HANDLE_KEYWORDS` / `NO_HANDLE_KEYWORDS` trong tên vật thể.)

Rồi chấm điểm theo thứ tự ưu tiên:

1. **Chồng lấn** (`bbox_overlap_ratio`) giữa `hand_region_bbox` — bbox bao trọn
   **toàn bộ** landmark bàn tay, không phải 1 điểm — với bbox từng bộ phận.
   Đáng tin hơn 1 điểm vì khi ngón bị che khuất, MediaPipe có thể đặt sai vị
   trí từng khớp riêng lẻ nhưng cả *vùng* tay vẫn bao đúng chỗ đang cầm.
2. Không có chồng lấn đáng kể → fallback sang **khoảng cách** từ điểm chạm gần
   nhất tới bbox bộ phận.
3. **Loại bỏ** bbox bộ phận nào chiếm >60% diện tích bbox tổng — dấu hiệu VLM
   không tách riêng được, trả nhầm cả vật thể.
4. Không bộ phận nào khớp → mặc định **"thân"**.

> **"thân" cố tình KHÔNG có trong danh sách so khớp.** Nếu cho nó thi đấu, bbox
> "thân" luôn chiếm gần hết vật thể nên luôn thắng ở khoảng cách ≈ 0, xoá sạch
> khả năng phân biệt nắp/quai. **"đáy" cũng bị bỏ**: khi cầm giữa thân, bbox
> vật thể luôn bị cắt cụt ngay chỗ tay che, nên điểm chạm luôn rơi gần "đáy
> nhìn thấy được" giả tạo → false positive liên tục.

### Bước 5 — Xử lý kết quả trả về ở `process_video`

Hàm trả `(text, verified)`. `verified=True` nghĩa là MediaPipe **đã** thấy tay
và đo được (kể cả khi kết luận là chưa chạm). Logic ở caller:

| Tình huống | Xử lý |
|---|---|
| `skill == "Reach"` mà **verified + có vị trí** | **Tự nâng cấp thành `Grasp`** — theo định nghĩa Reach là tiến gần *chưa chạm*, có chạm thật thì không còn là Reach |
| `verified` nhưng text rỗng, và mốc trước cùng vật vừa xác thực có chạm | **Mượn** `last_verified_touch` — bbox VLM xê dịch nhẹ giữa các lần gọi có thể đẩy khoảng cách vượt ngưỡng dù tay vẫn đang giữ nguyên vật |
| `verified` và có text | Dùng text, và **lưu lại** làm `last_verified_touch` |
| Không verified, nhưng cùng chuỗi cùng vật đã có vị trí xác thực | **Mượn** `last_verified_touch` (tin hơn mô tả tự do của VLM) |
| Không verified, chưa từng xác thực | Dùng `fallback_text` thô của VLM (chưa kiểm chứng) |
| Skill không thuộc nhóm chạm | Xoá `last_verified_*` (kết thúc chuỗi) |

---

## 6. Giai đoạn 5 — Render video overlay

Overlay có **2 dòng riêng biệt**, không dính chung 1 câu:

- **Dòng 1** (trắng, luôn hiện): `Skill: Cầm nắm - cốc`
- **Dòng 2** (vàng cam `(0,220,255)`, chỉ chớp): vị trí chạm, vd `quai cốc`

Chữ tiếng Việt được vẽ bằng **PIL** (`put_vietnamese_text`) chứ không phải
`cv2.putText` — OpenCV không render được dấu tiếng Việt.

Dòng 2 chỉ hiện khi **cả 3** điều kiện đúng:

1. Có `touch_location`;
2. `skill ∈ TOUCH_DISPLAY_SKILLS`;
3. `is_new_touch_event[sample_idx]` — **đây là mốc ĐẦU TIÊN của lần chạm này**
   (giá trị khác mốc liền trước). Giữ nguyên "thân" suốt 5 mốc thì chỉ chớp
   **1 lần** lúc bắt đầu, không nháy 5 lần.

Hiệu ứng chớp dùng **alpha blending** mượt thay vì bật/tắt nhị phân:

```
touch_flash_sec = min(0.6, interval/2)   # tổng thời gian hiện
fade_in_sec     = min(0.15, flash/3)     # 0 → 1
fade_out_sec    = min(0.3,  flash/2)     # 1 → 0
```

Video ghi bằng codec `mp4v`; nếu máy không phát được thì convert sang H.264
(script tự in gợi ý lệnh `ffmpeg` ở cuối).

---

## 7. Giai đoạn 6 — Xuất file & suy luận tên task

**`--log log_test6.json`** — timeline, mỗi mốc 1 bản ghi:

```json
{ "time_sec": 4.5, "skill_en": "Grasp", "skill_vi": "Cầm nắm",
  "object_vi": "cốc", "touch_location_vi": "quai" }
```

**`video_task_log.json`** — trừ khi có `--skip-task-log`, script tự gọi
`infer_task_name(skill_sequence, model_name=MODEL_NAME)` từ `infer_task_name.py`:
rút gọn chuỗi skill (bỏ lặp liên tiếp), hỏi LLM đặt 1 tên task tổng quát cho cả
video, rồi `append_task_log` **tích luỹ** (append, không ghi đè) để giữ lịch sử
qua nhiều video. Truyền `model_name` để dùng đúng model đang cấu hình (32B) thay
vì mặc định 7B riêng của file đó.

---

## 8. Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `video_path` | *(bắt buộc)* | Video đầu vào |
| `--skills` | `../skill_ontology/skills.yaml` | Ontology skill (tên **và** description — description được đưa thẳng vào prompt) |
| `--names-vi` | `../skill_ontology/skill_names_vi.yaml` | Bảng dịch tên skill sang tiếng Việt để hiển thị |
| `--interval` | `1.5` | Khoảng cách giữa 2 mốc lấy mẫu (giây) |
| `--out` | `annotated_output_vi_v3.mp4` | Video overlay đầu ra |
| `--log` | `segments_log_vi_v3.json` | Timeline JSON đầu ra |
| `--task-log` | `video_task_log.json` | File tích luỹ task_name qua nhiều video |
| `--skip-task-log` | *(tắt)* | Bỏ qua bước suy luận tên task |

### Chạy thử

```bash
source ../../venv/bin/activate
python3 segment_and_overlay_vi_v3.py ../video_test6.mp4 \
  --skills ../../skill_ontology/skills.yaml \
  --names-vi ../../skill_ontology/skill_names_vi.yaml \
  --out annot_test6.mp4 \
  --log log_test6.json
```

**Điều kiện cần:** Ollama đang chạy ở `localhost:11434` với model
`qwen2.5vl:32b` đã pull; `pip install mediapipe`; file `hand_landmarker.task`
nằm cùng thư mục với script.

---

## 9. Chi phí & giới hạn

**Số request VLM cho mỗi mốc thời gian:**

| Bước | Số request |
|---|---|
| `classify_window` (skill + object + fallback) | 1 |
| `locate_object_bbox_px` (chỉ khi skill thuộc nhóm chạm) | 0 hoặc 1 |
| Định vị bộ phận (chỉ khi đã xác minh có chạm) | 0 → 2 (1/bộ phận) |

Với 32B (~29s/request), video 25–30s có thể mất **15–20 phút**. MediaPipe không
tốn request nào.

**Giới hạn đã xác nhận (không phải bug):**

- MediaPipe vẫn có thể miss hoàn toàn ở vài góc quay dù đã hạ ngưỡng — nếu đó
  là mốc chạm **duy nhất** (không có chuỗi liền kề để mượn), hệ thống buộc phải
  fallback về mô tả tự do chưa kiểm chứng.
- Ngay lúc **vừa chạm một vật mới**, chưa có lịch sử trong chuỗi để mượn, ảnh
  lại hay bị mờ do chuyển động → có thể trống vị trí chạm ở 1-2 mốc.
- Cơ chế thử nhiều frame trong ~0.15s đôi khi bắt trúng khoảnh khắc chạm rất
  thoáng qua, trong khi chữ vẫn hiện đủ 0.6s → lệch nhẹ so với lúc tay còn
  chạm thật trên màn hình.
- Phân biệt vật thể **không đảm bảo 100% ổn định qua mọi lần chạy** do VLM suy
  luận độc lập không có bộ nhớ — đây là giới hạn thật của kiến trúc, không vá
  hết được bằng heuristic văn bản/hình học.
