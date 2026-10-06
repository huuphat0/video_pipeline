# Video → Skill Pipeline (tiếng Việt)

Nhận diện chuỗi **skill thao tác tay** từ video người thật, gán nhãn tiếng Việt,
xác định **vị trí chạm** trên vật thể, và suy luận **tên task** — dùng làm dữ
liệu tham khảo cho Task Planner của robot UR3.

Pipeline kết hợp 2 tầng để giảm hallucination:

| Tầng | Công cụ | Nhiệm vụ |
|---|---|---|
| Ngữ nghĩa | VLM qua Ollama (Qwen2.5-VL / Qwen3-VL) | phân loại skill, nhận diện vật thể, định vị bbox |
| Hình học | MediaPipe HandLandmarker | xác thực tay có **thực sự chạm** vật không, chạm vào bộ phận nào |

Nhãn vị trí chạm chỉ được tin khi tầng hình học xác nhận — nếu MediaPipe không
thấy tay, giá trị trả về là mô tả tự do của VLM và **chưa được kiểm chứng**.

## Cài đặt

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ollama + model VLM
ollama pull qwen3-vl:8b-instruct     # nhanh, 6.1 GB
ollama pull qwen2.5vl:32b            # chính xác hơn, 21 GB
```

`video_pipeline/models/hand_landmarker.task` (trọng số MediaPipe, 7.5 MB) đã kèm sẵn
trong repo nên không cần tải thêm. Font tiếng Việt lấy từ hệ thống
(`DejaVuSans-Bold.ttf`); thiếu font thì overlay tự fallback sang font mặc định
của PIL.

## Chạy

Repo **không chứa video** (`.mp4` bị gitignore vì quá nặng) — dùng video của
bạn. Bảo đảm Ollama đang chạy, rồi:

```bash
cd video_pipeline
python3 vlm/segment_and_overlay_vi_v3.py /duong/dan/video.mp4 \
    --model qwen3-vl:8b-instruct \
    --out out.mp4 --log log.json --task-log task_log.json
```

### Nếu nguồn là ROS2 bag (.mcap) của robot UR3

Bag chỉ chứa **ảnh camera thô**, KHÔNG có topic nhãn skill nào — nên vẫn phải
qua pipeline VLM như trên. Bước trung gian đổi bag -> mp4:

```bash
python3 data_io/mcap_to_mp4.py demo.mcap --list          # xem có topic ảnh nào
python3 data_io/mcap_to_mp4.py demo.mcap --out demo.mp4  # rút luồng ảnh màu
```

Rồi đưa `demo.mp4` vào pipeline. Thêm `--topic` để rút luồng khác (vd
`/camera/camera/aligned_depth_to_color/image_raw` — depth 16-bit được tô màu
sẵn cho dễ xem).

fps ghi ra được tính từ timestamp thật của bag (bag ghi không đều nhau), để
mốc thời gian `idx/fps` mà pipeline dùng khớp với timeline thật.

### Video dài hay ngắn — chọn `--interval`

Video demo ngắn (~12 s) mà để mặc định `--interval 1.5` thì chỉ được ~8 mốc.
Hạ xuống `--interval 1.0` (hoặc nhỏ hơn) cho video ngắn để không bỏ sót đoạn
chuyển skill nhanh. Đổi lại: càng nhỏ càng nhiều request VLM.

### Nhánh SUY LUẬN TỪ MASK — skill mức thấp, không cần VLM

Đường đi này **thay MediaPipe** bằng mask của Grounded-SAM, và cho ra chuỗi
skill mức thấp (Reach / Contact / Grasp / Lift / MoveToTarget / Pour / Place /
Retract) thay vì chỉ Grasp với Pour:

```bash
cd ../sam_dino
../../venv/bin/python gsam2_video.py ../out/demo_color.mp4 \
    --prompt "hand . plastic water bottle . white cup ." \
    --interval 1.0 --outdir ../out/gsam2_demo
cd ../github/video_pipeline
python3 skill_inference/hoi_skill_inference.py ../out/gsam2_demo/masks_gsam2.json \
    --out ../out/skills_lowlevel.json
python3 visualization/render_hoi_video.py ../out/demo_color.mp4 \
    --masks ../out/gsam2_demo/masks_gsam2.json \
    --skills ../out/skills_lowlevel.json --out ../out/demo_lowlevel.mp4 --scale 2
```

Không gọi Ollama lần nào, nên chạy lại cho kết quả y hệt (deterministic).

#### Vì sao thay được MediaPipe

| | MediaPipe + bbox VLM (bản v3) | Mask SAM2 (nhánh này) |
|---|---|---|
| Tay đang nắm chặt vật | Mất dấu landmark (đầu ngón bị che) | Mask tay/vật vẫn tách rời sạch |
| Vị trí vật | Bbox VLM, lệch giữa các lần gọi | Mask theo vật, ID ổn định |
| Độ phân giải thời gian | Mốc thưa (mặc định 1.5 s) | Mọi frame (30 fps) |
| Pha ngắn (Reach 0.2 s) | Không có mốc nào rơi vào | Bắt được |

#### Suy ra skill bằng hình học

| Skill | Điều kiện đo được |
|---|---|
| Reach | chưa chạm, khoảng cách TÂM tay–vật đang giảm |
| Contact | mask tay (nới 3 px) vừa chồng lấn mask vật, vật chưa di chuyển |
| Grasp | chạm liên tục, vật đứng yên hoặc đi cùng tay |
| Lift / Place | tâm vật đi lên / đi xuống |
| MoveToTarget | tâm vật đi ngang |
| Pour | trục chính của vật xoay (PCA trên mask) |
| Push | đang chạm, vật TỰ di chuyển nhưng KHÔNG cùng hướng với tay |
| Release | vừa mất chạm trong khi vật đứng yên |
| Retract | đã rời vật, khoảng cách tâm đang tăng |

`Push` phân biệt với `Grasp` bằng **tương quan chuyển động** `cos(v_tay, v_vật)`
— xem mục "Vì sao không dùng tư thế ngón tay" bên dưới.

#### Độ cao THẬT bằng depth (tuỳ chọn nhưng nên dùng)

Bag của robot UR3 có sẵn luồng depth (`aligned_depth_to_color`), nên `Lift` /
`Place` có thể đo bằng **milimét thật** thay vì pixel:

```bash
python3 data_io/depth_from_mcap.py demo.mcap --out demo_depth.npz
python3 skill_inference/hoi_skill_inference.py masks_gsam2.json --depth demo_depth.npz \
    --out skills_lowlevel.json
python3 visualization/render_hoi_video.py demo_color.mp4 --masks masks_gsam2.json \
    --skills skills_lowlevel.json --depth demo_depth.npz --out out.mp4 --scale 2
```

Mốc quy chiếu (mặt bàn) **tự hiệu chuẩn**: lấy phân vị 90 của độ sâu đo được
trên chính vật đó trong cả video — vật ở xa camera nhất chính là lúc nó nằm
trên bàn, không cần biết trước mặt bàn cao bao nhiêu. Camera nhìn xuống nên vật
nhấc lên thì depth giảm, đổi dấu để "cao hơn" là số dương.

Kết quả trên video demo — độ cao đo được khớp với hình ảnh:

| Thời điểm | Đo được | Nhìn thấy trong video |
|---|---|---|
| 0.0–2.0 s | 1.2 cm | chai nằm trên bàn, tay đang tới |
| 2.4–3.4 s | 1.5 → 6.3 cm | chai được nhấc khỏi bàn |
| 3.7–4.4 s | ~5.9 cm | đang rót, chai nghiêng |
| 5.7–6.4 s | ~0.1 cm | chai đã đặt lại xuống bàn |
| 8.1–9.7 s | 6.5 → 0 cm | hạ chai xuống lần hai |

**Depth còn sửa được cả phân loại:** trước khi có depth, đoạn 7.38–9.46 s bị
gọi là `Place` suốt; với độ cao thật thì thấy rõ chai **đang được giữ nguyên
độ cao** ở 6.7 cm và **đi ngang**, nên đoạn đó đúng ra là `MoveToTarget`, chỉ
đoạn hạ xuống thật mới là `Place`.

Lưu ý: luồng màu và luồng depth trong bag có **số frame khác nhau** (347 vs
346), nên `depth_from_mcap.py` ghép theo **timestamp gần nhất** và lưu lại theo
CHỈ SỐ FRAME MÀU — nếu ghép theo thứ tự message thì mask và depth sẽ lệch dần.

### SUB-SKILL — bước con bên trong một skill cấp cao

```bash
python3 skill_inference/subskill_inference.py masks_gsam2.json --depth demo_depth.npz \
    --out subskills.json
python3 visualization/render_hoi_video.py demo_color.mp4 --masks masks_gsam2.json \
    --skills skills_lowlevel.json --depth demo_depth.npz \
    --subskills subskills.json --out out_sub.mp4 --scale 2
```

| Sub-skill | ĐO BẰNG GÌ |
|---|---|
| `Approach` | chưa có pixel tay nào nằm sát vật trong ảnh |
| `Align` | có pixel tay sát vật, nhưng tay còn ở TRÊN CAO: `gap > 40 mm` |
| `Close_Gripper` | tay đã ngang tầm vật: `gap <= 40 mm` |
| `Orient_Tilt` | đang giữ + đang nghiêng, tốc độ xoay `vang >= +15°/s` |
| `Hold_Pour_Angle` | đang giữ + đang nghiêng, `\|vang\| < 15°/s` |
| `Return_Upright` | đang giữ + đang nghiêng, `vang <= -15°/s` |

`gap` = trung vị độ sâu của các pixel TAY nằm sát VẬT trừ trung vị độ sâu của
VẬT (xem `compute_gap`). Tay ở trên cao thì gần camera hơn nên depth nhỏ hơn →
gap âm nhiều; khi chạm thì cùng độ sâu → gap ≈ 0.

#### Ba tiêu chí KHÔNG ĐO ĐƯỢC — và vì sao

Định nghĩa yêu cầu có 3 tiêu chí dựa trên **trạng thái ngón tay**: "ngón còn
mở" (Approach), "CHƯA khép ngón" (Align), "ngón đang khép lại"
(Close_Gripper). Cả ba đều **không cài được** trên dữ liệu này — đã đo bằng
MediaPipe HandLandmarker ở mục "Vì sao không dùng tư thế ngón tay": tay đeo
găng nắm chai cho `curl` ≈ 0.05 **không đổi suốt video**, tức không phân biệt
được mở hay khép. Nên 3 sub-skill trên được tách bằng **khoảng cách hình học**
thay vì trạng thái ngón, và `Close_Gripper` gộp cả pha "đang khép" lẫn "đã
khép" vì không có cách nào đo ranh giới giữa hai pha đó.

#### Depth sửa được một lỗi THẬT của bản 2D

Cách xác định "đã chạm" cũ (mask tay nới rộng chồng lấn mask vật trong ảnh
2D) **sai khi tay ở trên cao**: bàn tay chiếu vuông góc xuống trùng vị trí
chai nên 2D báo "chạm". Đo trên video demo: t = 1.01–1.31 s, mask 2D báo CHẠM
nhưng gap = **96–121 mm** — tay vẫn lơ lửng cách chai cả 12 cm. Với depth thì
đoạn đó đúng ra là `Align`, không phải `Contact`.

#### Kết quả trên video demo (đã đối chiếu bằng mắt từng mốc)

```
Grasp  0.74–2.55 s   Approach 0.74–0.97 → Align 0.97–1.71 → Close_Gripper 1.71–2.55
Pour   4.19–5.10 s   Orient_Tilt 4.19–5.10
```

Video demo **chỉ có `Orient_Tilt`**, không có `Hold_Pour_Angle` và
`Return_Upright`. Đối chiếu khung hình: người rót nghiêng chai liên tục tới
gần nằm ngang rồi **đặt chai xuống luôn** (t≈5.3 s) — không có pha giữ nguyên
một góc nào. Chai nằm nghiêng trên bàn tới t≈6.4 s rồi được **nhấc lên dựng
thẳng lại**; đoạn dựng thẳng đó cố ý KHÔNG nhận là `Return_Upright` vì episode
rót bắt buộc phải bắt đầu từ tư thế gần thẳng (xem `find_pour_episodes`) —
nếu không thì mọi lần nhấc một vật đang nằm nghiêng lên đều bị gọi là "rót".

Vì vậy kết quả này cũng **sửa luôn một lỗi của bản trước**: đoạn 6.27–7.38 s
từng bị gán `Pour` thứ hai, thực ra là **nhấc chai lên**, không phải rót.

#### Ba cái bẫy đã gặp thật (đã xử lý trong code)

1. **Cẳng tay làm hỏng tín hiệu tiến/lùi.** SAM2 cắt `hand` thành một vùng
   liền gồm cả cẳng tay, nên khoảng cách mask tay–vật luôn nhỏ dù bàn tay còn
   xa → pha Reach biến mất. Đo lại bằng khoảng cách **tâm** tay–tâm vật.
2. **Mask rung lúc buông vật** làm tín hiệu "đang chạm" nhấp nháy → lấp khe hở
   ngắn và bỏ đoạn chạm quá ngắn.
3. **Góc trục chính nhảy bậc** khi mask đổi hình (vật bị che một phần): vài độ
   trong 1 frame chia cho 1/30 s thành >200°/s, đủ sinh Pour giả → lọc median
   trên chuỗi góc + chặn tốc độ xoay vô lý.

Ngoài ra Reach/Retract chỉ có nghĩa khi gắn với một lần tương tác, nên đoạn
đứng lẻ ở đầu/cuối video bị đổi thành Idle.

4. **Mask có thể KHÁC kích thước video.** `gsam2_video.py` có cờ `--max-side`
   thu nhỏ frame TRƯỚC khi detect, nên mask sinh ra theo hệ toạ độ đã thu nhỏ.
   Video demo 640x480 nhỏ hơn 720 nên không bị thu nhỏ (mask khớp), nhưng
   `video_test2.mp4` 1280x720 bị thu nhỏ thành 720x405 — tra mask bằng toạ độ
   frame gốc sẽ `IndexError`. `render_hoi_video.py` và `hand_pose_masked.py`
   đều đã tự kéo mask/ảnh về cùng cỡ trước khi dùng.

#### Vì sao KHÔNG dùng tư thế ngón tay (đã thử và đo, kết quả âm tính)

Ý tưởng ban đầu: mask nói tay có chạm vật, còn ngón cuộn hay duỗi sẽ phân biệt
`Grasp` (đang cầm) với `Contact`/`Push` (chỉ chạm/đẩy), và bắt được `Release`
đúng lúc ngón mở ra. Đã cài `hand_pose_masked.py` chạy MediaPipe
HandLandmarker rồi **đo trên dữ liệu thật** — và nó KHÔNG dùng được:

| | tay đeo găng (video demo) | tay trần (video_test2) |
|---|---|---|
| MediaPipe "phát hiện" tay | 336/347 frame (97%) | 156/297 frame (53%) |
| `curl` lúc ĐANG nắm vật | ~0.05 (ngón duỗi thẳng) | 0.30–0.51, nhiễu |
| `curl` lúc tay mở | ~0.06 | ~0.39 |

Với tay đeo găng, model vẽ **đúng vị trí** bàn tay nhưng dựng skeleton **ngón
duỗi thẳng** lên một bàn tay đang nắm chặt — `curl` gần như không đổi suốt
video, tức là **không phân biệt được lúc cầm và lúc buông**. Đây là bằng chứng
đo được cho thấy dùng landmark của MediaPipe để suy ra trạng thái ngón là
không đáng tin (và tỉ lệ "phát hiện được" 97% là chỉ số GÂY NHẦM — phát hiện
ra tay không có nghĩa là landmark đúng).

Cũng đã đo và bác bỏ một giả thuyết liên quan: **cắt ảnh theo bbox mask rồi
phóng to làm kết quả TỆ HƠN**, không phải tốt hơn — 295/347 (85%) so với
337/347 (97%) khi chạy trên toàn ảnh; số frame mà chỉ cách cắt làm được là 5,
so với 47 frame chỉ toàn ảnh làm được. Lý do: mask `hand` của SAM2 gồm cả cẳng
tay nên crop là vùng dài, tô đen nền xoá mất đường viền cánh tay mà MediaPipe
dùng để định vị, và phóng to đẩy bàn tay ra khỏi thang kích thước quen thuộc.
Vì vậy `hand_pose_masked.py` mặc định chạy **toàn ảnh**, chỉ dùng mask để CHỌN
đúng bàn tay và KIỂM CHỨNG landmark (chế độ `--mode crop` giữ lại để tái lập
phép đo).

**Thay thế đã dùng:** tương quan chuyển động `cos(v_tay, v_vật)`. Vật được cầm
thì đi cùng hướng với tay; vật chỉ bị chạm/đẩy thì đứng yên hoặc đi lệch
hướng. Đo trên video demo: cos trung vị **0.97** lúc có tương tác, và trên
video_test2 là **0.89** — tín hiệu rất rõ, lại hoàn toàn tất định, không phụ
thuộc model nào đoán tư thế.

#### Hạn chế đã biết

- Grounding DINO cần **tên vật có trong `--prompt`**; vật không liệt kê thì vô
  hình. Muốn tự tìm vật không cần tên thì dùng nhánh B (`sam_dinov2/`).
- `Pour` chỉ nhận ra với vật DÀI (chai, hộp) vì dựa vào trục chính của mask.
  Vật gần vuông (cốc, bát) xoay không đổi góc → không phát hiện được.
- **`Release` rất khó tách khỏi `Place`** — cả hai đều là "đặt vật xuống rồi
  buông". Trong video demo, Release chỉ dài 1 frame nên bị gộp vào Place.
  Muốn tách bạch cần thêm tín hiệu (xem phần đề xuất trong lịch sử trao đổi).
- Mask tay gồm cả cẳng tay nên mọi tín hiệu dựa trên TÂM tay đều bị lệch về
  phía cẳng tay; dùng hiệu (đạo hàm) thì sai lệch này triệt tiêu, nhưng vị trí
  tuyệt đối thì không dùng được.
- Tốc độ ~5.7 fps đầu-cuối (61 s cho video 11.6 s). Mask được cache ra JSON
  nên bước suy luận skill và render chạy lại gần như tức thì.

### Xuất dải timeline (tuỳ chọn)

`timeline_overlay.py` **hậu xử lý** video đã annotate: thêm 1 dải ngang dưới
cùng vẽ TOÀN BỘ chuỗi skill thành các khối màu, khối đang phát được tô sáng
kèm con trỏ thời gian — nhìn 1 khung là nắm được cả tiến trình thay vì phải
tua hết video. Script chỉ đọc lại file log của pipeline nên chạy rất nhanh,
**không tốn thêm request VLM nào**:

```bash
python3 visualization/timeline_overlay.py out.mp4 --log log.json \
    --task-log task_log.json --out out_timeline.mp4 --scale 2
```

`--scale 2` phóng to video gốc (640x480 -> 1280x960) cho dễ đọc rồi ghép dải
timeline xuống dưới.

Model mặc định trong script là `qwen2.5vl:32b` (nặng, 21 GB) — luôn truyền
`--model` nếu bạn muốn dùng bản 8b nhẹ hơn.

Các cờ hay dùng khác:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--interval` | `1.5` | khoảng cách giữa các mốc lấy mẫu (giây) — nhỏ hơn thì chi tiết hơn nhưng chậm hơn |
| `--skip-task-log` | tắt | bỏ bước suy luận tên task, chỉ xuất timeline |
| `--skills` / `--names-vi` | `skill_ontology/*.yaml` của repo | đường dẫn ontology, tính theo vị trí file script nên chạy từ thư mục nào cũng được |

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
  data_io/                           # CHUYỂN DỮ LIỆU VÀO
    mcap_to_mp4.py                   #   ROS2 bag (.mcap) -> mp4
    depth_from_mcap.py               #   ROS2 bag (.mcap) -> depth .npz (căn frame màu)
    dexycb_to_video.py               #   sequence DexYCB -> mp4
  perception/                        # ĐO ĐẠC TAY / VẬT
    contact_point.py                 #   điểm gắp + hướng tiếp cận + mô tả vị trí
    object_pose.py                   #   vị trí 3D + trục + độ nghiêng vật từ depth
    hand_pose_masked.py              #   đo tư thế ngón (ĐÃ ĐO: không dùng được)
  skill_inference/                   # SUY LUẬN SKILL TỪ MASK (không VLM)
    hoi_skill_inference.py           #   mask SAM2 -> skill mức thấp
    subskill_inference.py            #   skill cấp cao -> sub-skill (Approach/Align/...)
    skill_params.py                  #   tham số cho robot + gom điểm gắp
    skills_lowlevel.json             #   kết quả mẫu
  vlm/                               # PIPELINE VLM
    segment_and_overlay_vi_v3.py     #   pipeline chính (video -> video có nhãn skill)
    infer_task_name.py               #   suy luận tên task từ chuỗi skill
    SEGMENT_OVERLAY_V3_CACH_HOAT_DONG.md   # kiến trúc & luồng dữ liệu
    SEGMENT_OVERLAY_V3_IMPROVEMENTS.md     # lịch sử lỗi đã gặp và cách sửa
  visualization/                     # XUẤT VIDEO / ẢNH
    render_hoi_video.py              #   mask + skill + sub-skill -> video có nhãn
    timeline_overlay.py              #   hậu xử lý: thêm dải timeline skill
    grasp_card.py                    #   thẻ điểm gắp cho từng lần cầm
  models/
    hand_landmarker.task             # trọng số MediaPipe
skill_ontology/
  skills.yaml                        # 16 skill + precondition/effect
  skill_names_vi.yaml                # tên tiếng Việt
```

Các lệnh đều chạy từ thư mục `video_pipeline/` (vd `python3 skill_inference/hoi_skill_inference.py ...`).
Script import lẫn nhau theo dạng gói (`from skill_inference import hoi_skill_inference as H`),
mỗi script tự thêm `video_pipeline/` vào `sys.path` nên chạy trực tiếp từ thư mục nào cũng được.

Tất cả đều cần để chạy, trừ 2 file `.md` là tài liệu thuần.
`mcap_to_mp4.py` chỉ cần khi nguồn là bag; `timeline_overlay.py` là tuỳ chọn.
Hai file này **không** sửa `segment_and_overlay_vi_v3.py` — chúng nằm trước và
sau pipeline chính.

### Hai kiểu video xuất ra

| Kiểu | File | Nội dung |
|---|---|---|
| Gốc | `out.mp4` | ảnh camera + `Skill: <tên tiếng Việt> - <vật thể>` ở góc trên, vị trí chạm hiện chớp vàng |
| Timeline | `out_timeline.mp4` | như trên + dải timeline toàn bộ chuỗi skill + tên task, phóng to 2x |

Video ghi bằng codec `mp4v` của OpenCV, **không phát được trên nhiều trình
duyệt**. Convert sang H.264 trước khi chia sẻ:

```bash
ffmpeg -i out.mp4 -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p \
    -movflags +faststart out_h264.mp4
```

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

## Lưu ý sử dụng

Dữ liệu sinh ra là **tham khảo từ video quan sát** — không tự động đưa vào Task
Planner hay `skills.yaml`. Con người phải xem xét thủ công trước khi bổ sung luật.
