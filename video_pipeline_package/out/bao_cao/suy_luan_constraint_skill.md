# Suy luận yếu tố bắt buộc và yếu tố linh hoạt trong từng skill

## 1. Bài toán

Tầng nhận diện skill (xem `nguyen_ly_nhan_dien_skill.md`) đã cho biết **người làm gì, lúc nào**: chuỗi đoạn skill như Lift → MoveToTarget → Place, mỗi đoạn kèm vật đang thao tác. Nhưng để robot UR3 làm lại được, chỉ biết tên skill là chưa đủ. Robot còn cần biết trong mỗi skill:

- **Yếu tố bắt buộc (constraint):** điều phải giữ, nếu vi phạm thì task thất bại hoặc gây hại. Ví dụ khi mang chai nước đang mở nắp sang cốc, chai phải được giữ thẳng đứng trong suốt quá trình di chuyển.
- **Yếu tố linh hoạt (flexible):** điều robot được tự do chọn. Ví dụ quỹ đạo di chuyển tới cốc, tốc độ, hay góc xoay của chai quanh trục của nó.
- **Không xác định (unknown):** điều video không cho đủ thông tin để kết luận, ví dụ lực siết của tay.

Đầu vào là video cùng kết quả của các tầng trước. Đầu ra là file `constraints.json` liệt kê ba nhóm trên cho từng đoạn skill.

Phần này dùng mô hình ngôn ngữ chạy local để **phân tích môi trường và vật thể**, rồi **suy luận** ra yếu tố bắt buộc và linh hoạt. Mọi kết luận đều phải chỉ ra được bằng chứng cụ thể trong cảnh.

---

## 2. Tổng quan luồng xử lý

```
 video_depthcam/demo_20260912_172945_0.mcap  (ROS2 bag từ camera depth RealSense: ảnh màu + depth đã align)
        │  đã được các tầng trước xử lý thành:
        ▼
 out/json/skills_lowlevel.json   out/gsam2_demo/masks_gsam2_merged.json   out/mp4/demo_color.mp4
 (11 đoạn skill)                 (mask + ID vật từng khung hình)          (video màu)
 out/json/skill_params.json      out/json/demo_pose3d.json                out/npz/demo_depth.npz
 (độ cao, điểm cầm, hướng tiếp)  (vị trí 3D + trục + tilt_deg từng khung) (depth mm + nội tham số)
        │                                        │                                       │
        └────────────────────┬───────────────────┴───────────────────────────────────────┘
                             ▼
   [Bước 1] Schema + checklist     → định nghĩa khuôn dữ liệu, dùng chung cho mọi bước sau
   [Bước 2] Client Ollama          → hàm gọi model chung, ép JSON, cache kết quả
                             ▼
   [Bước 3] Bước A – Phân tích cảnh (VLM qwen3-vl:8b-instruct)
            keyframe có vẽ ID vật  ──►  scene.json   (chỉ MÔ TẢ, không suy luận)
                             ▼
   [Bước 4] Bước B – Suy luận (LLM qwen3.5:9b, bật thinking)
            scene.json + skill + checklist  ──►  constraints_raw.json
                             ▼
   [Bước 5] Kiểm tra tự động (code Python, không dùng model)
            loại kết luận không có bằng chứng + đối chiếu số đo depth  ──►  constraints.json + rejected.json
                             ▼
   [Bước 6] Chạy thử và so sánh model trên video demo_20260912 (có depth)  ──►  báo cáo so sánh
```

Ý tưởng cốt lõi là **tách "nhìn" khỏi "suy luận"**:

- **Bước A** chỉ trả lời câu hỏi "trong ảnh có gì, các vật đang ở trạng thái nào". Mô hình nhìn ảnh cỡ 8B mà phải vừa nhìn vừa suy luận nhiều tầng thì rất dễ bịa.
- **Bước B** chỉ làm việc trên text đã có cấu trúc, nên chọn được mô hình suy luận tốt hơn.
- Khi kết quả sai, xem `scene.json` là biết lỗi nằm ở khâu **nhìn** hay khâu **suy luận**.

### Vì sao không cần chạy realtime

Phần suy luận constraint chỉ chạy **một lần cho mỗi video (offline)**. Khi UR3 thực thi, robot không gọi lại mô hình ngôn ngữ mà chỉ tính các luật đã sinh ra, ví dụ `tilt(bottle) <= 15°`, trong vòng điều khiển ở tần số cao. Cách tách này giống ReKep (Huang et al., CoRL 2024). Vì vậy mô hình mất vài giây hay vài chục giây cho một đoạn skill đều chấp nhận được.

---

## 3. Công cụ sử dụng và mục đích

| Công cụ | Dùng ở bước | Mục đích |
|---|---|---|
| **Ollama** (server local, `localhost:11434`) | 2, 3, 4 | Chạy mô hình ngôn ngữ ngay trên máy, không cần API trả phí, dữ liệu không gửi ra ngoài. Tham số `format` cho phép truyền JSON Schema để ép mô hình trả đúng cấu trúc. |
| **qwen3-vl:8b-instruct** | 3 | Mô hình **nhìn ảnh** (VLM). Đọc keyframe và mô tả thuộc tính vật thể, môi trường. Chiếm khoảng 6 GB VRAM, đo trên máy đạt khoảng 74 token/s. Nhận diện chi tiết tốt hơn bản `qwen2.5vl:7b` đang dùng trong `vlm_recognize.py`. |
| **qwen3.5:9b** (chế độ thinking) | 4 | Mô hình **suy luận văn bản**. Chạy checklist phản chứng cho từng chiều của skill. Chiếm khoảng 6,6 GB VRAM, đo được khoảng 64 token/s. Chế độ thinking giúp mô hình suy nghĩ từng bước trước khi trả lời nên kết quả nhất quán hơn. |
| **Mask Grounded-SAM2** (`masks_gsam2_merged.json`) | 3 | Cung cấp vị trí và ID ổn định của từng vật ở mọi khung hình. Dùng để vẽ ID và nhãn lên keyframe, nhờ đó VLM gọi đúng vật theo `object_id` thay vì tự đặt tên. |
| **`sam_dino/common.py`** (`decode_rle`, `draw_frame`, `color_for`) | 3 | Dùng lại hàm sẵn có để giải mã mask RLE và tô màu theo ID. Không cần cài thêm `pycocotools`. |
| **OpenCV** (`cv2`, đã cài trong `venv`) | 3 | Đọc video, cắt keyframe tại thời điểm đầu, giữa, cuối của mỗi đoạn skill, vẽ khung và chữ. |
| **requests** (đã cài) | 2 | Gửi HTTP request tới Ollama, giống cách `vlm_recognize.py` đang làm. |
| **hashlib + thư mục cache** | 2 | Băm nội dung đầu vào (model + prompt + ảnh). Nếu đã từng gọi thì đọc lại kết quả cũ, khỏi chạy lại mô hình khi chỉ sửa code ở bước sau. |
| **JSON Schema** (dict Python thuần) | 1, 2, 5 | Mô tả khuôn dữ liệu của `scene.json` và `constraints.json`. Cùng một schema vừa truyền cho Ollama để ép đầu ra, vừa dùng để kiểm tra ở bước 5. Không cần cài thêm `pydantic`. |
| **`skill_ontology/skills.yaml`** | 4 | Lấy mô tả, preconditions và effects của từng skill để đưa vào prompt, giúp mô hình hiểu skill đó dùng để làm gì. |
| **Số đo từ depth** (`demo_pose3d.json`, `skill_params.json`) | 3, 5 | Kết quả đo sẵn của tầng trước: vị trí 3D, trục chính và góc nghiêng `tilt_deg` của vật ở từng khung, độ cao nhấc, điểm cầm, hướng tiếp cận. Đưa vào scene như **dữ kiện đo được**, và dùng ở bước 5 để đối chiếu constraint với chuyển động thật. |

Cả hai mô hình chạy cùng lúc chiếm khoảng 13,4 GB trên RTX 5060 Ti 16 GB, nên không phải nạp lại mô hình khi chuyển từ bước A sang bước B.

---

## 4. Chi tiết từng bước

### Bước 1: Định nghĩa schema và checklist chiều

**File tạo ra:** `video_pipeline/constraint_schema.py`

**Mục đích:** thống nhất khuôn dữ liệu cho mọi bước sau. Đây cũng là nơi quyết định mô hình **được phép nói gì**. Nếu không có khuôn cố định, mỗi lần gọi mô hình sẽ trả về một kiểu khác nhau và không so sánh được.

**Input:** không có dữ liệu. Đây là phần thiết kế.

**Nội dung:**

1. **Schema `scene.json`** mô tả cảnh tại một đoạn skill:
   - Mỗi vật gồm `id`, `label`, trạng thái (`lid`: open/closed/none, `contents`: liquid/solid/empty/unknown, `fill_level`), vật liệu, dễ vỡ hay không, có biến dạng không, hình dạng, kiểu đối xứng, các bộ phận cầm được và bộ phận chức năng (miệng chai, quai cốc).
   - Môi trường gồm mặt đỡ (bàn), các vật cản gần đường đi, mức độ bừa bộn.
   - Quan hệ không gian giữa các vật, ví dụ "chai ở bên trái cốc".

2. **Checklist 11 chiều** mà bước B bắt buộc phải xét hết cho mỗi skill:

   | Chiều | Ý nghĩa | Ví dụ khi là bắt buộc |
   |---|---|---|
   | `grasp_region` | Cầm vào phần nào của vật | Cầm quai ấm, không cầm thân nóng |
   | `grasp_orientation` | Hướng tiếp cận khi cầm | Cầm từ trên xuống |
   | `object_orientation_during_motion` | Tư thế vật khi di chuyển | Chai mở nắp, có nước thì phải giữ thẳng |
   | `path_shape` | Hình dạng quỹ đạo | Có vật cản thì phải vòng qua |
   | `path_clearance` | Khoảng cách an toàn với vật khác | Cách vật cản tối thiểu 5 cm |
   | `speed` | Tốc độ, gia tốc | Vật đầy nước hoặc dễ vỡ thì phải đi chậm |
   | `end_position` | Vị trí đích | Miệng chai nằm trên miệng cốc |
   | `end_orientation` | Tư thế ở đích | Khi rót, chai nghiêng 60–90° |
   | `contact` | Kiểu tiếp xúc với môi trường | Đẩy thì vật phải trượt trên mặt bàn |
   | `rotation_about_symmetry_axis` | Xoay quanh trục đối xứng | Vật tròn đối xứng thì chiều này linh hoạt |
   | `ordering` | Thứ tự với skill trước, sau | Phải nhấc lên trước khi di chuyển |

   Bốn kiểu `contact` (tự do, trượt trên mặt phẳng, tịnh tiến, quay quanh trục) lấy từ bài Li & Brock (2026).

3. **Schema `constraints.json`**: mỗi chiều của mỗi skill có các trường:
   - `status`: `constraint` / `flexible` / `unknown`
   - `phase`: `path` (giữ trong suốt skill) hoặc `subgoal` (phải đạt khi kết thúc skill). Cách chia này lấy từ ReKep.
   - `rule`: luật dạng gần với code, có ngưỡng số, ví dụ `tilt(obj1) <= 15deg`
   - `evidence`: danh sách đường dẫn tới thuộc tính trong `scene.json` làm căn cứ, ví dụ `objects[1].state.contents=liquid`
   - `reason`: giải thích ngắn bằng lời

**Output:** module Python `video_pipeline/constraint_schema.py` (đã làm xong), gồm:
- `SCENE_VLM_SCHEMA`: phần VLM điền
- `MEASURED_FIELDS`: các khóa số đo do code ghép vào
- `DIMENSIONS`: checklist 11 chiều
- `CONSTRAINT_SCHEMA`
- `FIELD_GUIDE`: định nghĩa từng trường để đưa vào prompt
- `EVIDENCE_PATTERN`
- hàm `validate()`

Các bước 3, 4, 5 đều import module này. Chạy `python3 constraint_schema.py` để tự kiểm tra.

Rút ra từ lần chạy thử với Ollama:
- **Thứ tự trường quan trọng.** Ollama sinh JSON đúng theo thứ tự trường trong schema. Nếu đặt `status` đầu tiên, model chốt `constraint` cho gần như mọi chiều. Vì vậy thứ tự được đặt là `reason → evidence → status → phase → rule`.
- **Schema chỉ ép kiểu dữ liệu, không truyền được ý nghĩa.** Thiếu `FIELD_GUIDE`, model chép câu hỏi vào `rule` và lấy vật đang cầm làm `target_id`.
- **Thinking mode cần `num_ctx` ≥ 16384.** Với ngữ cảnh mặc định, model nghĩ hết giới hạn trước khi viết JSON và trả về nội dung rỗng. Ghi chú này dành cho bước 2.

---

### Bước 2: Client gọi Ollama dùng chung

**File tạo ra:** `video_pipeline/llm_client.py`

**Mục đích:** gom mọi lần gọi mô hình vào một chỗ. Có ba cái lợi:

- Đổi mô hình chỉ cần đổi tham số `--vlm` / `--llm`, không phải sửa code.
- Ép đầu ra đúng schema.
- Có cache để chạy lại nhanh và cho cùng kết quả.

**Input:**
- Tên mô hình, ví dụ `qwen3-vl:8b-instruct`
- Prompt (text)
- Danh sách ảnh (tùy chọn, mã hóa base64)
- JSON Schema của đầu ra mong muốn
- Bật hoặc tắt chế độ thinking

**Xử lý:**
1. Kiểm tra Ollama đang chạy và mô hình đã được pull. Dùng lại logic `check_ollama_ready` trong `vlm_recognize.py`.
2. Băm toàn bộ đầu vào. Nếu đã có trong thư mục cache thì trả về luôn.
3. Gửi request tới `/api/chat` với `format=<schema>`, `temperature=0` để kết quả ổn định giữa các lần chạy.
4. Parse JSON trả về. Nếu lỗi parse thì thử lại tối đa 2 lần.
5. Ghi kết quả, thời gian chạy và số token vào cache.

**Output:**
- Một dict Python đúng schema
- Log thời gian và số token, dùng để so sánh tốc độ ở bước 6

**Đã làm:** file `video_pipeline/llm_client.py`, tự kiểm tra bằng `python3 llm_client.py --selftest`. Có hai điều chỉnh so với dự kiến:
- **Dùng `/api/chat` thay cho `/api/generate`.** Lý do: bước tự sửa cần gửi lại nhiều lượt hội thoại.
- **Khi đầu ra sai schema, gửi lại kèm danh sách lỗi để model tự sửa**, thay vì gửi lại y nguyên. Với `temperature=0`, gửi lại y nguyên sẽ ra đúng đầu ra cũ.

Chi tiết trong `time_line.md`.

---

### Bước 3: Bước A – Phân tích môi trường và vật thể

**File tạo ra:** `video_pipeline/infer_constraints.py` (phần `analyze_scene`)

**Mục đích:** biến ảnh thành mô tả có cấu trúc về **những thuộc tính quyết định constraint**. Chỉ biết nhãn "chai nhựa" là chưa đủ. Phải biết chai **đang mở nắp**, **có nước**, **đối xứng tròn**, và **giữa chai với cốc có vật cản không**. Bước này chỉ mô tả và tuyệt đối không suy luận ra constraint.

**Input:**
- `out/json/skills_lowlevel.json`: danh sách đoạn skill (thời gian bắt đầu và kết thúc, tên skill, `object_id` đang thao tác, `max_height_mm`), cùng `id_to_label`
- `out/gsam2_demo/masks_gsam2_merged.json`: mask và bbox của từng vật ở từng khung hình
- `out/mp4/demo_color.mp4`: video màu rút từ bag
- `out/json/skill_params.json` và `out/json/demo_pose3d.json`: số đo từ depth

**Xử lý:**
1. Với mỗi đoạn skill, lấy **3 keyframe**: đầu, giữa, cuối đoạn.
2. Trên mỗi keyframe, tô mask và ghi nhãn `#1 plastic bottle`, `#0 mug` lên từng vật bằng `draw_frame`. VLM nhờ đó trả lời theo đúng ID, và kết quả ánh xạ thẳng về mask.
3. Gửi 3 ảnh kèm prompt cho `qwen3-vl:8b-instruct`. Prompt yêu cầu:
   - Chỉ mô tả những gì **nhìn thấy được**
   - Không chắc thì ghi `unknown`, không được đoán
   - Trả về đúng `SCENE_SCHEMA`
4. Ghép thêm khối `measured` lấy từ số đo depth của đoạn đó:
   - Độ cao vật lúc đầu, lớn nhất và lúc cuối (`skill_params.json`)
   - Góc nghiêng `tilt_deg` nhỏ nhất, lớn nhất và trung bình (`demo_pose3d.json`, chỉ lấy các khung `axis_confident=true`)
   - Quãng đường vật đi được (từ `center_mm`)
   - Phần thân bị cầm, hướng tiếp cận

   Khối này đánh dấu nguồn là `measured`, tách khỏi thông tin VLM nhìn, để bước B biết đâu là số đo thật và đâu là mô tả.

**Output:** `out/constraints_demo/scene_<chỉ số đoạn>.json`, mỗi đoạn skill một file. Ví dụ cho đoạn MoveToTarget thứ nhất (3,39–4,09 s):

```json
{
  "segment": {"index": 5, "skill": "MoveToTarget", "t": [3.388, 4.093], "actor_id": 1,
              "prev": "Lift", "next": "Pour"},
  "objects": [
    {"id": 1, "label": "plastic water bottle",
     "state": {"lid": "unknown", "contents": "liquid", "fill_level": "full"},
     "material": "plastic", "fragile": false, "symmetry": "rotational_vertical",
     "functional_parts": ["cap", "opening"]},
    {"id": 0, "label": "white cup", "state": {"contents": "empty"},
     "functional_parts": ["opening_top", "handle"]}
  ],
  "environment": {"support_surface": "table", "obstacles_near_path": [], "clutter": "low"},
  "relations": ["obj1 left_of obj0"],
  "measured": {
    "height_mm": {"start": 59.4, "max": 62.4, "end": 55.4},
    "tilt_deg": {"min": 0, "max": 0, "mean": 0},
    "travel_mm": 0,
    "grasp_part": "thân", "grasp_along": 0.655
  }
}
```

Các giá trị `tilt_deg` và `travel_mm` ở trên chỉ minh họa. Số thật sẽ được tính khi chạy.

Một chi tiết quan sát được khi xem keyframe của video này: ở độ phân giải 640×480, **rất khó nhìn nắp chai đang đóng hay mở**, cũng không thấy rõ dòng nước khi rót. Đây là ca thử tốt cho bước A. VLM phải ghi `lid: unknown` thay vì đoán. Bước B khi đó phải dựa vào ngữ cảnh: skill liền sau là Pour, nên nắp nhiều khả năng đã mở. Bước B phải ghi rõ đây là suy luận, không phải quan sát.

Kèm theo là các ảnh keyframe đã vẽ ID, lưu cạnh file JSON để người kiểm tra có thể đối chiếu bằng mắt.

**Thời gian ước tính:** khoảng 5–10 giây mỗi đoạn skill.

---

### Bước 4: Bước B – Suy luận yếu tố bắt buộc và linh hoạt

**File:** `video_pipeline/infer_constraints.py` (phần `infer_constraints`)

**Mục đích:** từ mô tả cảnh, suy ra trong skill này cái gì phải giữ, cái gì được tự do. Đây là bước cần năng lực suy luận nhất nên dùng mô hình text có chế độ thinking.

**Input:**
- `scene_<i>.json` từ bước 3
- Tên skill, kèm mô tả, preconditions và effects lấy từ `skills.yaml`
- Skill liền trước và liền sau, để suy ra chiều `ordering` và vật đích. Ví dụ đoạn MoveToTarget trong `skills_lowlevel.json` chỉ ghi vật đang cầm, không ghi đích. Bước này suy ra đích là cốc, dựa vào skill liền sau (Pour) và quan hệ không gian.
- Khối `measured` trong scene: số đo thật về độ cao, độ nghiêng và quãng đường. Mô hình được yêu cầu **dựa vào số đo khi đặt ngưỡng**. Ví dụ, chai nghiêng tối đa X° khi di chuyển thì ngưỡng đặt quanh X° kèm biên an toàn, thay vì tự nghĩ ra ±15°.
- Checklist 11 chiều từ bước 1
- Mô tả robot: UR3 với gripper 2 ngón. Constraint phải áp dụng được cho robot, không phải cho bàn tay người.

**Xử lý:** gửi cho `qwen3.5:9b` (thinking) một prompt có ba quy tắc cứng:

1. **Phản chứng cho từng chiều:** *"Nếu chiều này thay đổi tùy ý, task có thất bại hoặc gây hại không?"* Có thì là `constraint`, không thì là `flexible`, không đủ thông tin thì là `unknown`.
2. **Bắt buộc dẫn chứng:** mỗi `constraint` phải trích ít nhất một thuộc tính trong `scene.json` làm căn cứ.
3. **Mặc định linh hoạt cho chuyển động tự do:** quỹ đạo và tốc độ là `flexible`, trừ khi có lý do cụ thể như vật cản hay chất lỏng. Nguyên tắc này lấy từ CP-Gen (Lin et al., CoRL 2025): đoạn di chuyển tự do có thể thay bằng motion planning.

**Output:** `out/constraints_demo/constraints_raw.json`, tức kết quả thô chưa kiểm tra. Ví dụ một đoạn:

```json
{
  "segment": 5, "skill": "MoveToTarget", "actor_id": 1, "target_id": 0,
  "dimensions": {
    "object_orientation_during_motion": {
      "status": "constraint", "phase": "path", "rule": "tilt(obj1) <= 15deg",
      "evidence": ["objects[1].state.contents=liquid", "segment.next=Pour", "measured.tilt_deg.max"],
      "reason": "Chai chứa nước và sắp được rót nên nhiều khả năng đã mở nắp; số đo cho thấy người giữ chai gần thẳng khi di chuyển"
    },
    "path_shape": {
      "status": "flexible", "evidence": ["environment.obstacles_near_path=[]"],
      "reason": "Không có vật cản, quỹ đạo tự do"
    },
    "rotation_about_symmetry_axis": {
      "status": "flexible", "evidence": ["objects[1].symmetry=rotational_vertical"]
    },
    "contact": {
      "status": "unknown", "reason": "Video không đo được lực"
    }
  }
}
```

**Thời gian ước tính:** khoảng 20–40 giây mỗi đoạn skill, vì có thinking.

---

### Bước 5: Kiểm tra tự động

**File:** `video_pipeline/infer_constraints.py` (phần `validate`)

**Mục đích:** chặn các kết luận bịa trước khi chúng đến robot. Bước này là code Python thuần, không gọi mô hình, nên luôn cho cùng một kết quả.

**Input:**
- `constraints_raw.json` từ bước 4
- Các file `scene_<i>.json` từ bước 3
- `CONSTRAINT_SCHEMA` và `DIMENSIONS` từ bước 1
- `out/json/demo_pose3d.json` và `out/json/skill_params.json`: số đo từng khung hình để đối chiếu

**Xử lý:** lần lượt kiểm tra năm điều kiện:

1. **Đúng schema:** đủ trường bắt buộc, `status` và `phase` chỉ nhận giá trị hợp lệ.
2. **Đủ checklist:** cả 11 chiều đều có câu trả lời. Chiều nào thiếu thì tự gán `unknown` và ghi chú lại.
3. **Bằng chứng có thật:** với mỗi `evidence`, tra ngược vào `scene.json` xem thuộc tính đó có tồn tại và đúng giá trị không. Không khớp thì **loại constraint** đó.
4. **Nhất quán logic:**
   - `constraint` phải có `rule`
   - Không có chiều nào vừa là `constraint` vừa có `evidence` mâu thuẫn, ví dụ ghi `contents=empty` nhưng lại nói giữ thẳng để khỏi đổ
   - `target_id` phải là một ID có trong cảnh
5. **Đối chiếu số đo (nhờ có depth):** với các constraint có thể đo được, tính giá trị thật trên mọi khung hình của đoạn skill.
   - `object_orientation_during_motion`: so ngưỡng với `tilt_deg` từng khung. Vượt ngưỡng ở hơn 10% số khung thì người làm mẫu đã không giữ constraint đó, nên gắn cờ `contradicted`.
   - `end_position` (vd: vật phải cao hơn miệng cốc): so với `center_mm` và `height_mm`.
   - Mỗi constraint được gắn `verification`:
     - `verified`: số đo thỏa mãn
     - `contradicted`: số đo vi phạm
     - `not_measurable`: chiều này không đo được, ví dụ lực

   Chỉ dùng các khung `axis_confident=true`. Khi vật quá tròn, trục chính không đáng tin.

**Output:**
- `out/constraints_demo/constraints.json`: **kết quả cuối cùng**, chỉ gồm những kết luận qua được kiểm tra, mỗi constraint kèm trạng thái `verification`
- `out/constraints_demo/rejected.json`: các kết luận bị loại kèm lý do, để người xem lại và chỉnh prompt

---

### Bước 6: Chạy thử và so sánh mô hình trên video demo_20260912 (có depth)

**Mục đích:** kiểm tra toàn bộ luồng chạy được từ đầu đến cuối, và có số liệu để chọn mô hình thay vì chọn theo cảm tính.

**Video test:** `video_depthcam/demo_20260912_172945_0.mcap`, metadata ở `video_depthcam/metadata.yaml`.
- Quay bằng camera depth RealSense (ROS 2 Jazzy), dài 11,6 s, khoảng 30 fps, độ phân giải 640×480
- 4 topic: `color/image_raw` (347 khung), `aligned_depth_to_color/image_raw` (346 khung, đã align theo pixel với ảnh màu), `color/camera_info`, `tf_static`
- Cảnh: một người cầm chai nước nhựa trên bàn, rót vào cốc trắng, rồi đặt chai lại

Video này **đã được các tầng trước xử lý**, nên bước 6 không phải chạy lại GSAM2 hay nhận diện skill:

| Dữ liệu sẵn có | Tạo bởi |
|---|---|
| `out/mp4/demo_color.mp4` | `mcap_to_mp4.py` |
| `out/npz/demo_depth.npz`: depth mm theo chỉ số khung màu + fx, fy, cx, cy | `depth_from_mcap.py` |
| `out/gsam2_demo/masks_gsam2_merged.json`: vật #0 white cup, #1 plastic water bottle, #2–4 hand | `gsam2_video.py` |
| `out/json/skills_lowlevel.json`: 11 đoạn Idle → Reach → Contact → Grasp → Lift → MoveToTarget → **Pour** → MoveToTarget → Place → Retract → Idle | `hoi_skill_inference.py` |
| `out/json/skill_params.json`, `out/json/demo_pose3d.json`: độ cao, điểm cầm, hướng tiếp cận, trục và `tilt_deg` từng khung | `skill_params.py`, `object_pose.py` |

Video này hợp để thử hơn video v2 (`video_depthcam/2.mcap`) vì hai lý do:
- **Có depth.** File `2.mcap` chỉ ghi ảnh màu, `camera_info` và `tf_static`, không có topic depth. Ngoài ra chỉ có 38 khung trong 8,9 s.
- **Có skill Pour**, là skill có nhiều constraint rõ ràng nhất: tư thế chai khi mang tới cốc, vị trí miệng chai so với miệng cốc, góc nghiêng khi rót.

**Xử lý:**
1. Chạy cấu hình chính: bước A dùng `qwen3-vl:8b-instruct`, bước B dùng `qwen3.5:9b`.
2. Chạy cấu hình so sánh: bước A thay bằng `qwen2.5vl:7b` (mô hình đang dùng hiện tại), bước B giữ nguyên. Nhờ vậy thấy được ảnh hưởng riêng của khâu "nhìn".
3. Chạy cấu hình chính thêm 2 lần nữa để đo độ ổn định: cùng đầu vào thì kết quả có giống nhau không.
4. Chạy cấu hình chính một lần **không đưa khối `measured`** vào prompt. Mục đích là xem số đo từ depth giúp mô hình đặt ngưỡng chính xác hơn bao nhiêu.
5. Tổng hợp số liệu. Các đoạn Idle được bỏ qua vì không có thao tác.

**Output:** `out/bao_cao/so_sanh_model_constraint.md` gồm:
- Bảng thời gian chạy và số token mỗi bước, theo từng mô hình
- Tỷ lệ constraint bị loại ở bước 5. Tỷ lệ cao nghĩa là mô hình hay bịa.
- Tỷ lệ constraint `verified` / `contradicted` khi đối chiếu với số đo depth, có và không có khối `measured`
- Độ ổn định giữa các lần chạy
- Vài ví dụ đúng và sai tiêu biểu, kèm keyframe
- Đề xuất mô hình nên dùng

---

## 5. Danh sách file sẽ tạo

```
video_pipeline/
├── constraint_schema.py      # Bước 1: schema + checklist 11 chiều
├── llm_client.py             # Bước 2: gọi Ollama, ép JSON, cache
└── infer_constraints.py      # Bước 3–5: phân tích cảnh, suy luận, kiểm tra
                              #   chạy: python infer_constraints.py \
                              #         --skills ../out/json/skills_lowlevel.json \
                              #         --masks ../out/gsam2_demo/masks_gsam2_merged.json \
                              #         --video ../out/mp4/demo_color.mp4 \
                              #         --pose3d ../out/json/demo_pose3d.json \
                              #         --params ../out/json/skill_params.json \
                              #         --out ../out/constraints_demo \
                              #         --vlm qwen3-vl:8b-instruct --llm qwen3.5:9b

out/constraints_demo/
├── keyframes/                # ảnh keyframe đã vẽ ID vật
├── scene_<i>.json            # Bước 3: mô tả cảnh từng đoạn
├── constraints_raw.json      # Bước 4: kết quả thô
├── constraints.json          # Bước 5: KẾT QUẢ CUỐI
└── rejected.json             # Bước 5: các kết luận bị loại

out/bao_cao/
└── so_sanh_model_constraint.md   # Bước 6
```

---

## 6. Giới hạn của giai đoạn này

- **Góc nghiêng đo trong hệ camera, chưa theo phương trọng lực.** `tilt_deg` trong `demo_pose3d.json` là góc của trục chai trong hệ tọa độ camera. Nếu camera chúc xuống bàn thì "thẳng đứng" trong ảnh không trùng với thẳng đứng thật. Cần ước lượng pháp tuyến mặt bàn từ depth để quy về phương trọng lực. Giai đoạn này mới so sánh tương đối, vd chai nghiêng *thêm* bao nhiêu độ so với lúc đứng yên trên bàn.
- **Ngưỡng là ngưỡng của người làm mẫu, chưa phải ngưỡng tối thiểu.** Depth cho biết người đã giữ chai nghiêng tối đa bao nhiêu. Nó không cho biết nghiêng thêm bao nhiêu thì nước bắt đầu đổ.
- **Một video chỉ cho ra giả thuyết.** Từ một lần làm, không phân biệt chắc được điều người làm vì bắt buộc với điều người làm tình cờ. Muốn khẳng định cần nhiều video của cùng một skill.
- **Lực và tiếp xúc không quan sát được.** Các chiều này sẽ thường là `unknown`, nhất là với Push, Pull, Insert. Đây là kết quả trung thực, không phải lỗi.
- **Mô hình 8–9B có giới hạn.** Có thể nhận sai trạng thái khó nhìn như chai trong suốt có nước hay không. Bước 5 chỉ chặn được kết luận *không có căn cứ trong scene*, không chặn được lỗi nhìn sai ở bước A.

## 7. Hướng mở rộng sau giai đoạn này

1. **Quy góc nghiêng về phương trọng lực:** ước lượng mặt phẳng bàn từ depth (RANSAC trên đám mây điểm), lấy pháp tuyến làm hướng "lên", rồi tính lại `tilt_deg` theo hướng đó.
2. **Tập nhãn tay để đánh giá:** gán nhãn đúng/sai cho 20–30 đoạn skill, rồi đo precision/recall của toàn bộ luồng.
3. **Chuyển `rule` thành hàm chi phí thực thi được** cho bộ tối ưu điều khiển UR3, theo cách ReKep làm.
4. **Nâng mô hình nếu cần:** Qwen3-VL-30B-A3B (MoE, chạy được với một phần trên RAM), hoặc Gemini API free tier làm baseline so sánh.

## 8. Tài liệu tham khảo

- Huang et al., *ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation*, CoRL 2024. https://arxiv.org/abs/2409.01652. Nguồn của cách chia `path` / `subgoal` và ý tưởng constraint là hàm trên keypoint.
- Lin et al., *Constraint-Preserving Data Generation for One-Shot Visuomotor Policy Generalization*, CoRL 2025. https://openreview.net/forum?id=KSKzA1mwKs. Nguồn của nguyên tắc "chuyển động tự do là linh hoạt" và định nghĩa bất biến theo hệ tọa độ vật.
- Li & Brock, *From a Single Demonstration to a General Policy for Contact-Rich Manipulation*, 2026. https://arxiv.org/abs/2605.17601. Nguồn của bốn kiểu ràng buộc tiếp xúc và cách tách cấu trúc chung khỏi chi tiết riêng của từng lần làm.
