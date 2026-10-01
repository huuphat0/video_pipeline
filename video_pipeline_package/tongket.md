# BÀN GIAO — Nhận diện skill thao tác từ video người

Cập nhật: 2026-09-15
Repo: `video_pipeline_ai/github`
Tài liệu chi tiết: [`SKILL_VERIFICATION.md`](SKILL_VERIFICATION.md) · [`README.md`](README.md)

> **Đọc mục 6 (đã thử và bác bỏ) và mục 8 (các loại lỗi hay gặp) TRƯỚC KHI VIẾT CODE.**
> Đó là phần tốn thời gian nhất để rút ra và dễ mất nhất khi chuyển phiên.

---

## 1. Mục tiêu

Từ video quay người thật thao tác vật thể → nhận diện **skill cấp thấp** và
**sub-skill**, cùng **điểm gắp trên vật**, để làm dữ liệu cho robot UR3 thực
hiện lại.

**Học SKILL, không học CHUYỂN ĐỘNG** — nên mọi thứ xuất ra đều là **tương đối
với vật**, không dùng toạ độ tuyệt đối của tay trong ảnh (camera người quay
khác camera robot nên con số đó không chuyển được).

---

## 2. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Skill cấp thấp | **9/12 đã kiểm chứng** — `Push`, `Pull`, `Release` thiếu dữ liệu, không phải lỗi code |
| Sub-skill | **6/6 đã cài và kiểm chứng** — `Approach/Align/Close_Gripper`, `Orient_Tilt/Hold_Pour_Angle/Return_Upright` |
| Điểm gắp trên vật | ✅ có, trôi **0.038** theo chiều dài vật |
| Hướng tiếp cận | ✅ có |
| Nhiều điểm gắp từ 1 video | ✅ `grasp_episodes()` — test3 cho 5 lần cầm, chai gắp ở 2 độ cao khác nhau |
| Gộp ID vật bị phân mảnh (P1) | ✅ **xong** — `sam_dino/merge_object_ids.py` + sửa `_assign_ids`, xem mục 7 P1 |
| Điểm gắp **NHIỀU VẬT/frame** cùng lúc | ✅ `skill_params.py --mode grasp_points` — bảng toạ độ cho mọi vật trong khung hình, không chỉ vật đang thao tác |
| Xuất tham số cho robot | ✅ `skill_params.py` |
| Toạ độ 3D thật (X,Y,Z mm, hệ camera) | ✅ `depth_from_mcap.py` trích nội tham số camera + `contact_point_xyz_mm()` — CHỈ đúng khi camera lúc dùng đứng y nguyên vị trí lúc quay |
| Áp điểm gắp đã học sang VẬT MỚI (ảnh/video khác) | ✅ `contact_point.py::apply_grasp_point` + `sam_dino/gsam2_frame.py` (bản 1 frame, không cần SAM2 video session) — đã chạy thật bằng GPU, không đoán API |
| Gộp ID vật bị phân mảnh (P1) | ✅ **xong** — `sam_dino/merge_object_ids.py` + sửa `_assign_ids`, xem mục 7 P1 |
| Điểm gắp **NHIỀU VẬT/frame** cùng lúc | ✅ `skill_params.py --mode grasp_points` — bảng toạ độ cho mọi vật trong khung hình, không chỉ vật đang thao tác |
| Kiểm chứng trên nhiều bộ dữ liệu | ✅ demo (bag UR3) · DexYCB (RGB-D) · video_test3 (RGB) |
| Commit | ❌ **~20 file chưa commit** — 2 file `sam_dino/` nằm ở repo git khác với `github/` |

### ⚠️ Lỗi mới phát hiện phiên này (đã sửa cả hai)

1. **`describe_location` đảo top/bottom cho VẬT DÀI** (chai/kéo/thìa, không ảnh
   hưởng vật tròn) — điều kiện neo chiều dùng dấu `<` thay vì `>`. Làm SAI mọi
   `along`/`part` của vật dài trong các JSON đã xuất (KHÔNG ảnh hưởng điểm vẽ
   trên video, vì điểm vẽ là phép chiếu độc lập với cách đặt tên). Đã sửa +
   chạy lại toàn bộ số liệu — xem mục 5.
2. **`dict.get(key, d["other_key"])` crash khi thiếu `"other_key"`** trong
   `contact_point_xyz_mm` — `.get()` của Python LUÔN tính giá trị mặc định dù
   key chính đã có, gặp ngay khi `apply_grasp_point` gọi hàm này với dict chỉ
   có `axis_x/axis_y`. Đã sửa sang kiểm tra `in` tường minh.

Xem video: `out/best_skills.mp4` (ghép demo + DexYCB) · `out/demo_subskill_h264.mp4` (đầy đủ nhất)
· `out/t3_contact_h264.mp4` · `out/skill_taxonomy.html` (bản đồ 3 tầng).

---

## 3. Pipeline và file

```
.mcap ──mcap_to_mp4.py──→ .mp4 ──┐
      └─depth_from_mcap.py─→ .npz ┤
                                   ├─→ sam_dino/gsam2_video.py ─→ masks_gsam2.json
(DexYCB: dexycb_to_video.py ───────┘                                │
                                                                    ▼
                              sam_dino/merge_object_ids.py (hậu xử lý, gộp ID vỡ)
                                                                    │
                                                                    ▼
                              hoi_skill_inference.py --depth → skills_lowlevel.json
                              subskill_inference.py  --depth → subskills.json
                              contact_point.py               → điểm gắp (module)
                              skill_params.py --mode segments        → skill_params.json (theo đoạn skill)
                              skill_params.py --mode grasp_points    → toạ độ gắp MỌI vật trong video
                              skill_params.py --mode grasp_episodes  → toạ độ gắp từng LẦN CẦM
                                                                    │
                                                                    ▼
                              render_hoi_video.py ─→ video có nhãn + mũi tên + vòng ngắm
```

`sam_dino/gsam2_video.py` đã tự nối ID tốt hơn (theo vai trò tay/vật thay vì
chữ nhãn — xem P1), nhưng `merge_object_ids.py` vẫn nên chạy vì nó gộp được
cả những phân mảnh KHÔNG liền chunk mà bước tracking không tự thấy.

**Không bước nào gọi VLM** → chạy lại cho kết quả y hệt.

| File | Dòng | Việc |
|---|---:|---|
| `hoi_skill_inference.py` | 916 | mask → 12 skill cấp thấp, đo mọi tín hiệu |
| `render_hoi_video.py` | 550 | render video có nhãn + timeline + điểm gắp + hướng |
| `subskill_inference.py` | 511 | skill → sub-skill |
| `skill_params.py` | 392 | xuất tham số cho robot + gom nhiều điểm gắp |
| `hand_pose_masked.py` | 309 | đo tư thế ngón — **ĐÃ BÁC BỎ**, giữ làm bằng chứng |
| `sam_dino/merge_object_ids.py` | ~150 | hậu xử lý gộp ID vật bị phân mảnh trên `masks_gsam2.json` có sẵn (P1) |
| `contact_point.py` | 280 | điểm gắp + hướng tiếp cận + mô tả vị trí |
| `timeline_overlay.py` | 235 | hậu xử lý thêm dải timeline |
| `mcap_to_mp4.py` / `depth_from_mcap.py` / `dexycb_to_video.py` | 163/142/155 | bộ chuyển dữ liệu vào |
| `segment_and_overlay_vi_v3.py` | 1059 | pipeline CŨ (VLM + MediaPipe) — **không dùng nữa**, giữ tham chiếu |

---

## 4. Dữ liệu

| Đường dẫn | Gì | Dung lượng |
|---|---|---|
| `video_pipeline_ai/demo_20260912_172945_0.mcap` | bag UR3 + RealSense, 11.6s, **có depth** | 533 MB |
| `video_pipeline_ai/video_test3.mp4` | 8.9s, **KHÔNG depth** | — |
| `datasets/dexycb/raw/` | 4 sequence DexYCB, **có depth** | 994 MB |
| `video_pipeline_ai/out/` | mọi kết quả + mask + video | — |

DexYCB tải được **994 MB thay vì 12.8 GB** bằng cách tải tiền tố file `.tar.gz`
rồi giải nén phần đã tải (tar ghi tuần tự).

---

## 5. Kết quả kiểm chứng (đã đối chiếu từng mốc với khung hình)

```
demo (11 đoạn)
  Idle → Reach → Contact → Grasp → Lift → MoveToTarget
       → Pour [Orient_Tilt → Hold_Pour_Angle → Return_Upright]
       → MoveToTarget → Place → Retract → Idle

Grasp → Approach 0.74-0.97 → Align 0.97-1.71 → Close_Gripper 1.71-2.55
Pour  → Orient_Tilt 4.19-5.80 → Hold_Pour_Angle 5.80-6.27 → Return_Upright 6.27-7.31
```

**Điểm gắp chai (demo):** (202, 215) · vùng `thân` · `along=0.351` · 241 frame đã xác thực
(số `along` đã SỬA — xem lỗi mới phát hiện ngay dưới đây, giá trị cũ 0.649 SAI chiều)

**test3 — 5 lần cầm, chai gắp ở 2 độ cao:**
```
ly thuỷ tinh  0.17-0.37s  along=0.066  phần dưới · bên phải
ly thuỷ tinh  0.47-2.03s  along=0.183  phần dưới · bên phải
chai nước     3.36-5.36s  along=0.144  phần dưới     (hầu hết frame dùng nhánh vật TRÒN, không bị lỗi)
chai nước     6.23-8.43s  along=0.615  thân          (đã sửa từ 0.382 — nhánh vật DÀI, bị lỗi)
ly thuỷ tinh  8.30-8.60s  along=0.089  phần dưới · bên phải
```

### ⚠️ LỖI MỚI PHÁT HIỆN VÀ ĐÃ SỬA: `along` của VẬT DÀI bị đảo top/bottom

`contact_point.py::describe_location` (nhánh `trục chính`, dùng cho vật DÀI như
chai/kéo/thìa — KHÔNG ảnh hưởng vật TRÒN như ly/cốc, nhánh đó tính riêng và
đúng) có điều kiện "neo chiều" bị **đảo dấu**: `if (c+hi*axis)[1] < (c+lo*axis)[1]`
đáng lẽ phải là `>`. Hậu quả: `along=1` (đáng lẽ là ĐẦU TRÊN) lại rơi vào ĐÁY,
và ngược lại — mọi giá trị `along`/`part` của vật dài bị **đảo ngược hoàn toàn**.

Đo trực tiếp trên mask chai thật (frame 10, demo, chai đứng thẳng chưa bị cầm):
bản lỗi cho NẮP CHAI `along=0.007 "phần dưới"` và ĐÁY CHAI `along=0.995 "phần
trên"` — ngược hẳn thực tế. Đã sửa dấu `<` thành `>`, kiểm chứng lại: nắp chai
`along=0.993 "phần trên"`, đáy chai `along=0.005 "phần dưới"` — đúng.

**KHÔNG ảnh hưởng đến điểm vẽ trên video** (`axis_x`/`axis_y` trong
`draw_contact_marker`) — điểm đó là phép chiếu hình học độc lập với cách đặt
tên `along`, nên chấm hồng trên mọi video đã render từ trước vẫn đúng vị trí.
Chỉ SỐ và CHỮ (`along`, `part`) trong các file JSON (`grasp_points_*.json`,
`grasp_episodes_*.json`, `skill_params*.json`) bị sai đối với VẬT DÀI — đã
chạy lại toàn bộ và cập nhật số liệu ở trên.

---

## 6. ⛔ ĐÃ THỬ VÀ BÁC BỎ — đừng thử lại

| Hướng | Số đo | Kết luận |
|---|---|---|
| **MediaPipe đo tư thế ngón** | Tay đeo găng nắm chai: `curl ≈ 0.05` **không đổi suốt video**; tay trần chỉ phát hiện 53% frame | Không dùng được. Tỉ lệ "phát hiện 97%" là **chỉ số gây nhầm** — bắt được tay ≠ landmark đúng |
| **Cắt ảnh theo mask cho MediaPipe** | 295/347 (85%) so với **337/347 (97%)** toàn ảnh | Cắt làm TỆ HƠN |
| **EPIC-KITCHENS làm nguồn** | Bộ dữ liệu **không có depth** | Chỉ kiểm được nửa taxonomy |
| **Tâm tay suy ra điểm gắp** | Đo trong **hệ vật**: dọc trục ổn (0.020) nhưng **ngang kém** (0.085) và lệch +0.42…+0.50 so với trục | Không dùng làm vị trí. Dùng **trọng tâm vùng chạm** |
| **Medoid làm giá trị ĐO** | trôi 0.267 | Nhảy quanh vành tiếp xúc. Chỉ dùng để VẼ |
| **Trọng tâm vùng chạm để VẼ** | rơi ra ngoài vật (2/4 mốc) | Chỉ dùng để ĐO. Vẽ phải neo vào trục vật |

---

## 7. Việc PHẢI LÀM TIẾP (xếp theo ưu tiên)

### P1 — Gộp ID vật bị phân mảnh `(rẻ, sửa được ngay)` — ✅ ĐÃ XONG

Nguyên nhân xác nhận: Grounding DINO là open-vocab, cùng MỘT vật thật đổi
chữ nhãn giữa các keyframe (`glass cup`→`glass`→`cup`), và `_assign_ids` cũ
đòi khớp CHỮ nhãn tuyệt đối nên tạo ID mới dù bbox gần như không đổi (IoU đo
được 0.96–0.99 tại điểm gãy). Test3: **4 ID cho 2 vật thật**, chỉ 1/4 nhận
tiếp xúc — đúng như mô tả cũ.

Đã làm cả 2 hướng gợi ý:
1. **Sửa gốc** `sam_dino/common.py::Sam2ChunkTracker._assign_ids` (+ `_role`
   mới): nối ID theo **IoU + VAI TRÒ tay/vật** (heuristic giống `pick_roles`
   bên `hoi_skill_inference.py`) thay vì so khớp chữ nhãn tuyệt đối. Áp dụng
   cho các lần chạy `gsam2_video.py` MỚI.
2. **Hậu xử lý** `sam_dino/merge_object_ids.py` (mới): gộp ID trên
   `masks_gsam2.json` ĐÃ CÓ SẴN, không cần chạy lại DINO+SAM2. Thuật toán:
   tách mỗi ID thành các RUN liên tục (một ID có thể "biến mất rồi quay lại"
   — id2 ở test3 sống 0-209 rồi 255-263, đoạn giữa 210-254 là ID khác), rồi
   nối 2 RUN cùng vai trò, cách nhau ≤10 frame, IoU bbox tại điểm nối ≥0.5,
   và KHÔNG từng cùng xuất hiện chung frame (điều kiện chặn gộp nhầm 2 vật
   thật đang cùng trong khung hình).

**Đã kiểm chứng bằng mắt** (đối chiếu bbox từng frame) trên 4 file:

| File | Trước | Sau | Gộp | Đúng không |
|---|---|---|---|---|
| `gsam2_t3` (test3) | 6 ID | 4 ID | `id2↔id4`, `id3↔id5` (2 ly) | ✅ bbox trùng khít, IoU 0.96–0.99 |
| `gsam2_demo` | 5 ID | 5 ID | không có gì để gộp | ✅ (không có phân mảnh) |
| DexYCB `seq_144839` | 4 ID | 4 ID | không có gì để gộp | ✅ |
| DexYCB `seq_154850` | 6 ID | 5 ID | `power drill`↔`bottle` (IoU 0.78) | ✅ cùng 1 khối, chỉ đổi nhãn giữa video |
| DexYCB `seq_155735` | 6 ID | 5 ID | `wood block box`↔`box` (IoU 0.99) | ✅ hộp đứng yên nguyên video |

Sau gộp, `skill_params.py grasp_points_per_object` trên test3 ra đúng
**3 vật thật** (chai, ly có chạm, ly không chạm) thay vì 6 dòng lẫn lộn với
3 dòng `contact_frames: 0` giả — số liệu episode (5 lần cầm) khớp y hệt bảng
ở mục 5, xác nhận không có hồi quy.

**Việc còn lại (không chặn P1, nhưng nên làm)**: nối `merge_object_ids.py`
thành bước tự động ngay sau `gsam2_video.py` trong tài liệu chạy (mục 9),
thay vì gọi tay; và converting `sam_dino/common.py` sang git riêng nên nhớ
commit ở **2 repo khác nhau**.

### P2 — Detector phải tìm ĐÚNG vật đang được cầm `(chặn lớn nhất)`

**3/5 sequence DexYCB thất bại** vì Grounding DINO bỏ sót chính vật đang cầm
(ví dụ `144839`: người cầm hộp can, detector chỉ ra kéo/khối gỗ/máy khoan).

→ Ba hướng: (a) luôn đưa đúng tên vật vào `--prompt`; (b) hạ `--box-threshold`;
(c) dùng **nhánh B** `sam_dinov2/` để tự tìm vật **không cần tên** (đã có sẵn
trong repo, chưa nối vào pipeline).

### P3 — Pose 6DoF của vật `(để điểm gắp chính xác hơn)`

Điểm gắp còn trôi **0.038** vì trục chính ước lượng bằng PCA trên mask, không
phải tư thế thật của vật. Khi chai xoay 94°→177° thì trục trôi theo.

→ Cần pose estimator (FoundationPose…) hoặc dùng `pose.npz` có sẵn của DexYCB.
Vật cần có texture rõ.

### P4 — Quay thêm dữ liệu có depth `(cách rẻ nhất để mở rộng)`

Cần quay bằng chính rig UR3 + RealSense, xuất `.mcap` như demo:

| Cần kiểm chứng | Video cần có |
|---|---|
| `Push`, `Pull` | đẩy/kéo vật trượt trên bàn **không nhấc lên** |
| `Release` | pha buông dài hơn 1 frame |
| Nhiều loại vật | 4–5 vật khác nhau (chai, cốc sứ, **ly thuỷ tinh**, hộp, thìa) — mỗi thứ gắp riêng |
| Vật trong suốt | ly thuỷ tinh **rỗng** — để xác nhận RealSense có đo được depth không |

### P5 — Sub-skill chưa cài

| Skill | Sub-skill | Thước đo dự kiến |
|---|---|---|
| `Pour` | `Pre_Pour` | đang giữ + thẳng + ở trên vật chứa + không di chuyển — **cần suy luận LIÊN VẬT THỂ** |
| `Lift` | `Settle_Grip`, `Raise`, `Hold_At_Height` | `vheight`, `height_mm` — **không cần gì mới, làm được ngay** |
| `Place` | `Align_Over_Target`, `Lower`, `Touch_Surface` | `vheight`, `height_mm` |
| `MoveToTarget` | `Transport`, `Decelerate_Settle` | `vcx`/`vcy`, `hold` |
| `Push` | `Contact_Surface`, `Apply_Force`, `Slide` | `overlap`, `obj_speed`, `hold` |
| `Retract` | `Settle_At_Rest` | `vcdist` |

**Ưu tiên cụm `Lift` + `Place`** — dùng tín hiệu đã có, video nào cũng có 2 pha này.

### P6 — Vật TRÒN khi mô tả vị trí

Đã có xử lý (`ELONG_MIN = 2.0`: dài → trục chính, tròn → phương đứng) nhưng cần
thêm: **trong/ngoài** cho vật chứa (cốc, bát), và **tâm/rìa** cho vật tròn.

### P7 — Unit test

13 lỗi đã tìm ra đều bằng chạy tay. Nên có test cho `pick_active_object`,
`split_grasp_subskills`, `deglitch_contact`, `find_contact`, `_object_references`.

---

## 8. Các LOẠI LỖI hay gặp — đọc trước khi viết code

Trong phiên này tìm ra **13 lỗi thật** bằng cách chạy trên dữ liệu mới. Chúng
xếp thành **5 nhóm**, và nhóm nào cũng sẽ tái diễn:

| Nhóm | Mô tả | Ví dụ đã gặp |
|---|---|---|
| **1. Dùng một luật cho hai ngữ cảnh khác nhau** | Điều kiện để **VÀO** một pha không được dùng làm điều kiện để **Ở TRONG** pha đó | `held` đúng để loại "bình nằm nghiêng trên bàn" nhưng sai cho "đang rót"; `not_rising` đúng để loại "nhấc vật nằm nghiêng" nhưng sai cho "dựng thẳng lại" |
| **2. Trộn nhiều vật vào một chuỗi** | Chuỗi theo "vật đang thao tác" trộn nhiều vật, nhưng **mọi giá trị THAM CHIẾU phải tính theo từng vật** | `d_ref` (mốc mặt bàn) và `upright_ref` (tư thế nghỉ) tính chung → cả hai vật đều sai |
| **3. Đạo hàm/trung bình qua điểm gián đoạn** | Đạo hàm qua chỗ đổi vật là rác (đo được **±600 mm/s**) | Sinh `Lift`/`Place` giả |
| **4. Trung bình mà không kiểm tra kết quả còn trong tập hợp** | Trọng tâm của hình **vành khuyên** nằm ở giữa — tức trên bàn tay | Điểm gắp rơi ra ngoài vật ở 2/4 mốc |
| **5. ĐO SAI HỆ QUY CHIẾU** | Đo độ ổn định của điểm gắn trên vật trong hệ ảnh, khi chính vật đang chuyển động | Con số "44 px" của tâm tay là SAI — lẫn chuyển động của vật |

**Lỗi nhóm 5 là lỗi tôi mắc và phải tự đính chính** — đã ghi cả hai lần đo vào
`SKILL_VERIFICATION.md` mục 4f.

### Cạm bẫy kỹ thuật cụ thể

- **Mask `hand` của SAM2 gồm cả CẲNG TAY** → mọi tín hiệu dựng trên TÂM tay đều
  lệch; đạo hàm thì sai lệch triệt tiêu nên vẫn dùng được
- **Mask có thể KHÁC kích thước video**: `gsam2_video.py --max-side` thu nhỏ frame
  TRƯỚC khi detect. Video 1280×720 → mask 720×405. Phải kéo về cùng cỡ.
- **Luồng màu và depth trong bag KHÁC số frame** (347 vs 346) → ghép theo
  **timestamp**, lưu theo chỉ số frame màu
- **Vật tròn: trục chính PCA nhảy LOẠN**, không phải "không đổi" → sinh dương
  tính giả, không chỉ bỏ sót. Ngưỡng độ dài: cốc/ly 1.27–1.54, chai 2.71–3.14

---

## 9. Chạy lại

```bash
cd video_pipeline_ai/github/video_pipeline

# --- Demo (bag ROS2, có depth) ---
python3 mcap_to_mp4.py ../../demo_20260912_172945_0.mcap --out ../../out/demo_color.mp4
python3 depth_from_mcap.py ../../demo_20260912_172945_0.mcap --out ../../out/demo_depth.npz
cd ../sam_dino && ../../venv/bin/python gsam2_video.py ../../out/demo_color.mp4 \
    --prompt "hand . plastic water bottle . white cup ." --interval 1.0 \
    --outdir ../../out/gsam2_demo

# --- Gộp ID vật bị phân mảnh (P1) — luôn chạy bước này trước khi suy luận ---
python3 merge_object_ids.py ../../out/gsam2_demo/masks_gsam2.json \
    --out ../../out/gsam2_demo/masks_gsam2_merged.json
cd ../github/video_pipeline
python3 hoi_skill_inference.py ../../out/gsam2_demo/masks_gsam2_merged.json \
    --depth ../../out/demo_depth.npz --out ../../out/skills_lowlevel.json
python3 subskill_inference.py ../../out/gsam2_demo/masks_gsam2_merged.json \
    --depth ../../out/demo_depth.npz --out ../../out/subskills.json
python3 render_hoi_video.py ../../out/demo_color.mp4 \
    --masks ../../out/gsam2_demo/masks_gsam2_merged.json --skills ../../out/skills_lowlevel.json \
    --depth ../../out/demo_depth.npz --subskills ../../out/subskills.json \
    --out ../../out/demo_subskill.mp4 --scale 2

# --- Tham số skill theo đoạn (1 vật/đoạn) ---
python3 skill_params.py ../../out/gsam2_demo/masks_gsam2_merged.json --mode segments \
    --skills ../../out/skills_lowlevel.json --depth ../../out/demo_depth.npz \
    --out ../../out/skill_params.json

# --- Toạ độ điểm gắp cho MỌI vật trong video (không chỉ vật đang thao tác) ---
python3 skill_params.py ../../out/gsam2_demo/masks_gsam2_merged.json --mode grasp_points \
    --depth ../../out/demo_depth.npz --out ../../out/grasp_points.json

# --- Toạ độ điểm gắp cho TỪNG LẦN CẦM (nếu một vật được cầm nhiều lần) ---
python3 skill_params.py ../../out/gsam2_demo/masks_gsam2_merged.json --mode grasp_episodes \
    --depth ../../out/demo_depth.npz --out ../../out/grasp_episodes.json
```

Môi trường: `/home/phuoc/ur3_skill_learning_host/venv/bin/python` (có cv2,
mediapipe, PIL, yaml, mcap, mcap-ros2-support).

---

## 10. Việc dọn dẹp còn nợ

- **13 file chưa commit** — nên tách thành các commit theo chủ đề: (1) bộ chuyển
  dữ liệu vào, (2) suy luận skill + sub-skill, (3) điểm gắp + tham số, (4) tài liệu
- `requirements.txt` đã thêm `mcap`, `mcap-ros2-support`
- `out/skill_taxonomy.html` viết để publish Artifact nhưng phiên này xác thực bằng
  `ANTHROPIC_AUTH_TOKEN` nên **không publish được** — nếu phiên mới đăng nhập bằng
  tài khoản claude.ai thì publish được
