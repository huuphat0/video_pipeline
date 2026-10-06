# Timeline – Suy luận yếu tố bắt buộc / linh hoạt trong skill

Kế hoạch chi tiết 6 bước: `suy_luan_constraint_skill.md`.

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Schema + checklist 11 chiều | ✅ Xong (06/10/2026) |
| 2 | Client gọi Ollama | ✅ Xong (06/10/2026) |
| 3 | Bước A – Phân tích cảnh (VLM) | ⏳ Chưa làm |
| 4 | Bước B – Suy luận bắt buộc / linh hoạt (LLM) | ⏳ Chưa làm |
| 5 | Kiểm tra tự động + đối chiếu số đo depth | ⏳ Chưa làm |
| 6 | Chạy thử, so sánh model trên video demo_20260912 | ⏳ Chưa làm |

---

## 05–06/10/2026 – Chuẩn bị

### Đã làm
- **Đọc 3 bài báo và rút ra những gì dùng được:**
  - **ReKep:** cách chia `path` / `subgoal`, và ý tưởng viết constraint thành hàm trên keypoint.
  - **CP-Gen:** nguyên tắc "chuyển động tự do là linh hoạt".
  - **Li & Brock 2026:** bốn kiểu ràng buộc tiếp xúc.
- **Chọn model chạy local qua Ollama, không dùng API trả phí:**
  - Bước A dùng `qwen3-vl:8b-instruct`: chiếm khoảng 6 GB VRAM, đo được khoảng 74 token/s.
  - Bước B dùng `qwen3.5:9b`: chiếm khoảng 6,6 GB VRAM, đo được khoảng 64 token/s.
  - Hai model chạy cùng lúc vừa trong 16 GB của RTX 5060 Ti.
- **Viết kế hoạch 6 bước** vào `suy_luan_constraint_skill.md`.
- **Sắp xếp lại `github/video_pipeline/`** thành 6 thư mục theo chức năng (`data_io`, `perception`, `skill_inference`, `vlm`, `visualization`, `models`). Sửa import và đường dẫn tương ứng. Chạy lại trên video demo, kết quả trùng với output cũ. Danh sách thư mục ghi ở `ghi_chu.md`.

### Thay đổi so với dự kiến
- **Không dùng GPT-4o.** Không có OpenAI API key. Ngoài ra OpenAI đã ngừng `chatgpt-4o-latest` trên API từ 02/2026.
- **Đổi video test từ `2.mcap` (v2) sang `demo_20260912_172945_0.mcap`.**
  - Lý do: `2.mcap` không ghi topic depth, chỉ có ảnh màu, 38 khung trong 8,9 s.
  - Video demo có depth đã align, khoảng 30 fps, có skill Pour.
  - Video demo cũng đã được xử lý sẵn (mask, depth `.npz`, 11 đoạn skill, `tilt_deg` từng khung), nên bước 6 không phải chạy lại GSAM2.
- **Kế hoạch bổ sung phần đối chiếu với số đo thật** (bước 5) nhờ có depth.

---

## 06/10/2026 – Bước 1: Schema + checklist

### Đã làm
File `video_pipeline/constraint_schema.py`:

| Thành phần | Mục đích |
|---|---|
| `SCENE_VLM_SCHEMA` | Khuôn cho VLM mô tả cảnh: thuộc tính vật, môi trường, quan hệ không gian |
| `MEASURED_FIELDS` | Các khóa số đo depth do code ghép vào scene |
| `DIMENSIONS` | Checklist 11 chiều, kèm câu hỏi phản chứng và khóa số đo để đối chiếu |
| `CONSTRAINT_SCHEMA` | Khuôn đầu ra bước B. Cả 11 chiều bắt buộc, nên Ollama ép model trả lời đủ |
| `FIELD_GUIDE` | Định nghĩa từng trường, để đưa vào prompt |
| `EVIDENCE_PATTERN` | Định dạng bằng chứng |
| `validate()` | Kiểm tra theo schema, liệt kê đủ mọi lỗi |

**Kiểm tra:**
- `python3 constraint_schema.py` tự kiểm tra đạt: ví dụ mẫu hợp lệ, và bắt được các lỗi cố ý cài vào.
- Ollama nhận cả hai schema qua tham số `format`. Đầu ra của cả hai model đều hợp lệ theo schema.

**Kết quả chạy thử bước B** trên đoạn MoveToTarget (chai sang cốc, skill sau là Pour):
- Chọn đúng đích: `target_id = 0`, là cốc.
- Constraint:
  - `tilt(obj1) <= 5deg` (path)
  - `above(obj1, obj0.opening_top)` (subgoal)
  - `after(Lift) and before(Pour)`
- Linh hoạt: quỹ đạo, khoảng cách, tốc độ, góc xoay quanh trục chai.

### Thay đổi so với dự kiến

| Dự kiến | Thực tế | Lý do |
|---|---|---|
| `SCENE_SCHEMA` chứa toàn bộ scene | Tách thành `SCENE_VLM_SCHEMA` (model điền) và `MEASURED_FIELDS` (code ghép) | Model không được tự điền số đo |
| Bằng chứng dạng `objects[1].state...` | Dạng `obj1.state...`, dùng ID vật | Chỉ số mảng dễ nhầm khi model liệt kê vật theo thứ tự khác. ID thì khớp với mask GSAM2 |
| `fragile: true/false` | `fragile: yes/no/unknown`; mọi trường đều có `unknown` | Cho model nói "không biết" thay vì bị ép đoán |
| `target_id` có thể là `null` | `-1` khi không có đích | Ollama ép kiểu `integer` ổn định hơn |
| Không có | Thêm trường `contact_type` | Ghi kiểu tiếp xúc (free_space / plane / prismatic / revolute) |
| Không có | Thêm `FIELD_GUIDE` | Xem khó khăn 2 bên dưới |
| Thứ tự trường tùy ý | `reason → evidence → status → phase → rule` | Xem khó khăn 1 bên dưới |
| Idle, Contact, Retract | `SKIP_SKILLS = {"Idle"}` | Idle không có thao tác. Contact và Retract có trong `skills_lowlevel.json` nhưng **không có trong `skills.yaml`**, bước 4 phải tự viết mô tả cho chúng |

### Khó khăn gặp phải
1. **Model chốt kết luận trước rồi mới tìm lý do.**
   - Ollama sinh JSON đúng theo thứ tự trường trong schema. Khi `status` đứng đầu, `qwen3.5:9b` gắn `constraint` cho 8/11 chiều, rồi nhét chuỗi bằng chứng vào `rule`.
   - Cách xử lý: đặt `reason` và `evidence` lên trước `status`. Số chiều `flexible` tăng từ 3 lên 5.
2. **Schema chỉ ép kiểu dữ liệu, không truyền được ý nghĩa.**
   - Thiếu phần định nghĩa trường, model chép câu hỏi vào `rule`, để `phase` toàn `none`, và chọn vật đang cầm làm đích.
   - Cách xử lý: viết `FIELD_GUIDE` để bước 4 đưa vào prompt. Sau khi có nó, cả ba lỗi đều hết.
3. **Thinking mode trả về nội dung rỗng.**
   - Model nghĩ khoảng 3.800 token, chạm giới hạn ngữ cảnh mặc định của Ollama (`done_reason=length`) trước khi kịp viết JSON.
   - Cách xử lý: đặt `num_ctx=16384`. Bước 2 phải mặc định giá trị này.
4. **Thinking mode chậm.**
   - Mỗi đoạn skill mất khoảng 90 giây khi bật thinking, so với khoảng 17 giây khi tắt.
   - Chất lượng: bật thinking thì bằng chứng đúng định dạng 100%; tắt thì có 2 bằng chứng sai định dạng.
   - Video demo có 9 đoạn cần suy luận, tức khoảng 14 phút nếu bật thinking. Bước 6 sẽ so sánh kỹ hai chế độ.
5. **VLM mô tả sai trạng thái vật.**
   - Khi thử text-only, `qwen3-vl:8b` ghi chai là `empty` dù prompt nói "full of water".
   - Cần đo lại ở bước 3 với ảnh thật. Nếu lỗi kiểu này lặp lại, cân nhắc model lớn hơn cho bước A.

### Việc tiếp theo
- **Bước 2:** client Ollama (`llm_client.py`): mặc định `num_ctx=16384`, `temperature=0`, ép JSON theo schema, kiểm tra bằng `validate()`, thử lại khi lỗi, có cache.

---

## 06/10/2026 – Bước 2: Client gọi Ollama

### Đã làm
File `video_pipeline/llm_client.py`. Hàm chính là `call(model, prompt, schema, images, think, system, cache_dir)`, trả về dict gồm:
- `data`: JSON đã kiểm tra theo schema
- `raw`, `thinking`
- `duration_s`, `prompt_tokens`, `eval_tokens`
- `repairs`, `cached`

| Chức năng | Cách làm |
|---|---|
| Kiểm tra sẵn sàng | `check_ready()` khớp **đúng tên kèm tag**. Thiếu model thì dừng ngay |
| Ép JSON | Truyền schema vào `format`, sau đó kiểm tra lại bằng `constraint_schema.validate()` |
| Ổn định kết quả | `temperature=0`, `seed=0` |
| Ngữ cảnh | Mặc định `num_ctx=16384`, để thinking mode không bị cắt |
| Giữ model trong VRAM | `keep_alive=30m`, để không phải nạp lại model giữa các đoạn skill |
| Lỗi mạng hoặc 5xx | Thử lại tối đa 3 lần, giãn cách tăng dần |
| Lỗi 4xx (request sai) | Báo lỗi ngay, không thử lại |
| JSON đúng cú pháp nhưng sai schema | Gửi lại kèm đầu ra cũ và danh sách lỗi để model tự sửa, tối đa 1 lượt |
| Đầu ra rỗng hoặc bị cắt | Báo lỗi ngay, gợi ý tăng `num_ctx` |
| Cache | Băm toàn bộ request (model, prompt, nội dung ảnh, schema, options). Gọi lại cùng đầu vào thì đọc file, không chạy model |
| Đo đạc | Ghi thời gian và số token mỗi lần gọi, dùng cho bước 6 |

**Kiểm tra** (`python3 llm_client.py --selftest --image <keyframe>`), 10/10 ca đạt:
- **Với Ollama thật:**
  - Ollama sẵn sàng.
  - Chặn được model không tồn tại.
  - VLM nhận ảnh thật (khung 3,7 s của video demo) kèm schema scene: 19 giây.
  - LLM kèm schema constraint: 22 giây, chọn đúng đích là cốc.
  - Gọi lại cùng đầu vào thì lấy từ cache (0,000 giây).
  - Bật think trên `qwen3-vl:8b-instruct` bị báo lỗi HTTP 400 ngay, không thử lại.
- **Với phản hồi giả lập:**
  - Sai schema thì model tự sửa được sau 1 lượt.
  - Vẫn sai sau lượt sửa thì báo lỗi.
  - Bị cắt thì báo lỗi ngay, không gửi lại.
  - Gọi không kèm schema thì nhận text thường.

### Thay đổi so với dự kiến

| Dự kiến | Thực tế | Lý do |
|---|---|---|
| Dùng lại `check_ollama_ready` của `vlm_recognize.py` | Viết `check_ready()` mới | Hàm cũ chỉ so phần trước dấu `:`, nên `qwen3-vl:8b` cũng khớp với `qwen3-vl:30b`. Hàm cũ cũng chỉ cảnh báo chứ không dừng |
| "Lỗi parse JSON thì thử lại 2 lần" | Gửi lại **kèm danh sách lỗi** để model tự sửa | Với `temperature=0`, gửi lại y nguyên sẽ ra đúng đầu ra cũ |
| Không có | Lỗi 4xx thì không thử lại; đầu ra rỗng hoặc bị cắt thì không gửi lại | Thử lại những trường hợp này chỉ tốn thời gian |
| Không có | Thêm `keep_alive`, `seed` | Giữ VRAM và ổn định kết quả |
| Dùng `/api/generate` như `vlm_recognize.py` | Dùng `/api/chat` | Cần nhiều lượt hội thoại cho bước tự sửa, và cần tách system / user |

### Khó khăn và quan sát
1. **VLM liệt kê cả người, ghế, bàn thành vật thể.** Khi chưa có ID vẽ sẵn lên ảnh, VLM liệt kê 5 "vật": tay cầm chai, cốc, người, ghế, bàn. Đây là lý do bước 3 phải vẽ ID từ mask GSAM2 lên keyframe và chỉ cho mô tả các vật có ID. Nắp chai vẫn là `unknown`, khớp với nhận xét khi xem ảnh bằng mắt.
2. **Kết quả LLM dao động giữa các lần chạy dù `temperature=0`.**
   - Cùng một prompt, ở bước 1 (chưa đặt `seed`) model trả 7 constraint / 4 flexible; lần này (`seed=0`) trả 8 / 3.
   - Ngoài `seed`, kết quả còn phụ thuộc vào trạng thái GPU, nên "cùng đầu vào → cùng đầu ra" chỉ chắc chắn được nhờ cache.
   - Bước 6 cần đo độ ổn định thật bằng cách chạy nhiều lần với cache tắt.
3. Tin nhắn báo lỗi gửi model để tự sửa đang lấy nguyên thông báo tiếng Việt từ `validate()`. Ca giả lập chạy đúng, nhưng chưa thử model thật có hiểu không. Nếu bước 4 gặp lỗi sửa không được thì đổi sang tiếng Anh.

### Việc tiếp theo
- **Bước 3:** cắt 3 keyframe (đầu / giữa / cuối) mỗi đoạn skill, vẽ ID và nhãn từ mask GSAM2, gọi VLM và ghép khối `measured`, xuất `scene_<i>.json`.
