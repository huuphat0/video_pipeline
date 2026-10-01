# Kiểm chứng skill — trạng thái

Cập nhật: 2026-09-13

Phạm vi: tầng **skill cấp thấp** và **sub-skill** của pipeline suy luận từ mask
(`hoi_skill_inference.py` + `subskill_inference.py`). Toàn bộ nhãn đều là phép đo
hình học trên mask SAM2 + độ sâu — **không gọi VLM**, nên chạy lại cho kết quả
y hệt.

Video xem kết quả: `out/best_skills.mp4` (ghép 2 phần, có thẻ tiêu đề).

---

## 1. Bộ dữ liệu đã dùng để kiểm chứng

| Bộ | Nền tảng | Depth | Độ phân giải | Vai trò |
|---|---|---|---|---|
| `demo_20260912_172945_0.mcap` | ROS2 bag, RealSense (UR3) | ✅ uint16 mm | 640×480 | Kiểm chứng chính — chuỗi skill đầy đủ nhất |
| DexYCB `20200928_155212` cam `932122060861` | RealSense D415 | ✅ uint16 mm | 640×480 | Kiểm chứng **độc lập** — dữ liệu khác hoàn toàn, không chỉnh tham số |
| `video_test2.mp4` | RGB (không depth) | ❌ | 720×405 | Kiểm chứng chuỗi trên cảnh khác |
| DexYCB `144839`, `154850`, `155735` | RealSense D415 | ✅ | 640×480 | **Thất bại** — xem mục 5 |

---

## 2. Skill cấp thấp — đã kiểm chứng chạy tốt

| Skill | demo | DexYCB | test2 | Ghi chú |
|---|:--:|:--:|:--:|---|
| `Idle` | ✅ | ✅ | ✅ | |
| `Reach` | ✅ | ✅ | ✅ | |
| `Contact` | ✅ | ✅ | ✅ | |
| `Grasp` | ✅ | ✅ | ✅ | |
| `Lift` | ✅ | ✅ | ✅ | đo bằng mm thật |
| `MoveToTarget` | ✅ | — | — | |
| `Pour` | ✅ | — | ✅ | |
| `Place` | ✅ | — | ✅ | |
| `Retract` | ✅ | — | — | |
| `Push` | — | — | — | **chưa video nào có pha đẩy thật** |
| `Pull` | — | — | — | chưa gặp |
| `Release` | ⚠️ | — | — | chỉ dài **1 frame** trong demo nên bị gộp vào `Place` |

**9/12 skill đã kiểm chứng chạy tốt.** 3 skill còn lại chưa gặp pha tương ứng
trong bất kỳ video nào (không phải lỗi code — code đã cài, chỉ thiếu dữ liệu).

### Chuỗi thực tế nhận được

```
demo (13 đoạn)
  Idle → Reach → Contact → Grasp → Lift → MoveToTarget → Pour → Grasp
       → Lift → MoveToTarget → Place → MoveToTarget → Retract → Idle

DexYCB kéo (5 đoạn)
  Idle → Reach → Contact → Grasp → Lift

video_test2 (10 đoạn)
  Idle → Reach → Contact → Lift → Pour → Grasp → Lift → Pour → Place → Idle
```

---

## 3. Sub-skill (tầng 3) — đã kiểm chứng chạy tốt

### Grasp → { Approach, Align, Close_Gripper }

| Sub-skill | demo | DexYCB | Đo bằng |
|---|:--:|:--:|---|
| `Approach` | ✅ 0.77–0.97s | ✅ 1.03–1.57s | chưa có pixel tay nào nằm sát vật (`gap = NaN`) |
| `Align` | ✅ 0.97–1.71s | ✅ 1.57–2.17s | có pixel tay sát vật nhưng tay còn trên cao (`gap > 40 mm`) |
| `Close_Gripper` | ✅ 1.71–2.55s | ✅ 2.17–2.47s | tay đã ngang tầm vật (`gap ≤ 40 mm`) |

**Đây là kết quả quan trọng nhất**: bộ ba sub-skill này đo bằng **khoảng cách
theo chiều sâu**, thứ mà MediaPipe không làm được (xem mục 6). Chúng chạy đúng
trên **hai bộ dữ liệu khác nhau, cùng một bộ tham số**.

### Pour → { Pre_Pour, Orient_Tilt, Hold_Pour_Angle, Return_Upright }

| Sub-skill | Trạng thái | Đo bằng |
|---|---|---|
| `Pre_Pour` | ○ **chưa cài** | cần suy luận liên vật thể (bình ↔ vật chứa) |
| `Orient_Tilt` | ✅ demo 4.19–5.80s | `vang ≥ +15°/s` |
| `Hold_Pour_Angle` | ✅ demo 5.80–6.27s | `\|vang\| < 15°/s`, đang giữ + nghiêng |
| `Return_Upright` | ✅ demo 6.27–7.31s | `vang ≤ −15°/s` |

Ban đầu tôi kết luận sai rằng video demo "rót liên tục một chiều rồi đặt xuống
luôn, không có pha giữ góc". Xem lại khung hình dày ở 5.0–7.25s thì **sai**:
chai giữ nguyên tư thế nằm ngang suốt 5.0–6.75s (tay vẫn trên chai), rồi
6.75–7.25 mới dựng thẳng lên. Người quay có làm đúng động tác giữ góc và dựng
thẳng — lỗi nằm ở logic nhận diện, không phải ở video. Xem mục 4b.

---

### Điểm chạm trên vật (`contact_point.py`)

Bản suy luận từ mask trước đó **đã bỏ mất** thông tin "chạm vào đâu trên vật" —
bản v3 dùng VLM mới có (`touch_location`: "nắp", "thân", "quai"). Với việc học
từ video người thật thì đây là dữ liệu cần: robot phải biết nắm **chỗ nào**,
không chỉ biết là "đang nắm".

Cách tính, hình học thuần:

1. **Vùng chạm** = pixel của vật nằm trong bán kính 6 px quanh bàn tay
2. **Điểm chạm** = trọng tâm vùng đó (không lấy 1 pixel vì rìa mask rung)
3. **Vị trí trên vật** = chiếu điểm chạm lên trục chính (PCA), chuẩn hoá 0–1
   theo chiều dài vật, neo lại để `t=1` là đầu ở **trên** trong ảnh → chai dựng
   đứng thì `t=1` là phía nắp. Kèm vị trí ngang → "bên trái/phải"

Video vẽ vòng ngắm tại điểm chạm + nhãn. **Chỉ vẽ khi `touched` đã xác nhận** —
`find_contact` dựa trên mask 2D nới rộng nên vẫn ra "điểm chạm" khi tay ở trên
cao, đúng cái bẫy đã gặp ở mục 4.

| Video | Nhãn hiện ra | Đúng? |
|---|---|---|
| demo, nắm chai | `thân · chai nước` suốt grasp/lift/pour/place | ✅ tay nắm giữa thân chai |
| test3, chạm ly | `phần trên · bên trái · ly thuỷ tinh` | ✅ tay ở phía trên-bên trái ly |
| test3, nắm chai | `phần trên` rồi `thân` | ✅ khớp khung hình |

---

### Tham số skill để robot thực hiện lại (`skill_params.py`)

Nhãn "Grasp" chưa đủ để robot làm. `skill_params.py` kết xuất bản ghi mà Task
Planner dùng được:

```json
{ "skill": "Grasp", "object": "chai nước", "start": 1.21, "end": 2.52,
  "contact_on_object": { "along": 0.486, "part": "thân", "n_frames": 39 },
  "approach": { "deg": 46.5, "label": "từ trên-phải" },
  "height_mm": { "at_start": 5.4, "max": 15.4, "at_end": 15.4 } }
```

**Nguyên tắc: học SKILL, không học CHUYỂN ĐỘNG.** Toạ độ **tuyệt đối** của tay
trong ảnh KHÔNG có trong bản ghi — camera người quay khác camera robot nên con
số đó không chuyển được. Mọi thứ đều **tương đối với vật**: vị trí trên vật
(0–1 dọc trục), hướng tới (góc so với vật), độ cao mm, góc vật.

#### Tâm tay hay tâm lòng bàn tay?

Đã đo cả hai, và câu trả lời là **không dùng cái nào làm vị trí tương tác**:

- **Tâm mask tay** — mask `hand` của SAM2 gồm cả cẳng tay nên tâm bị kéo về phía
  cổ tay. Trên video demo, tâm mask và "điểm dày nhất" lệch nhau 20–48 px.
- **Tâm lòng bàn tay** — **không có cách hình học nào đáng tin để tìm**. Cách
  "điểm dày nhất của distance transform" **hỏng hẳn ở t=1.4s**, rơi vào mép trên
  ảnh `(408, 0)`.

Cách đúng là **chia theo giai đoạn**:

| Giai đoạn | Dùng gì | Vì sao |
|---|---|---|
| Trước khi chạm | **tâm mask tay** → chỉ để suy ra HƯỚNG tới | cẳng tay kéo tâm lệch nhưng **không đổi hướng** tay-tới-vật |
| Sau khi chạm | **điểm chạm trên VẬT** | đây mới là thứ robot cần: "nắm chỗ này", tính theo hệ của vật |

Video vẽ đúng theo hai giai đoạn: **trước khi chạm hiện mũi tên vàng** "tới: từ
trên-phải"; **sau khi chạm hiện vòng ngắm hồng** "thân · chai nước". Hai thứ
không bao giờ hiện cùng lúc.

---

## 4. Lỗi THẬT tìm ra được NHỜ kiểm chứng

Đây là giá trị chính của việc chạy trên dữ liệu mới: **7 lỗi** chỉ lộ ra khi gặp
cảnh khác với video demo, không cái nào phát hiện được nếu chỉ chạy mãi trên demo.

| # | Lỗi | Phát hiện ở đâu | Cách sửa |
|---|---|---|---|
| 1 | **Rót giả khi NHẤC vật dài** — nhặt kéo lên làm trục chính xoay, đủ ngưỡng sinh `Pour` | DexYCB 155212 | thêm điều kiện `vheight < 30 mm/s`: không vừa nhấc vừa rót |
| 2 | **Vật "đang thao tác" nhảy qua lại** giữa 2 vật nằm sát nhau → tín hiệu 2 vật trộn vào nhau | DexYCB 154850 | chọn theo **diện tích tiếp xúc** + giữ nguyên vật cũ |
| 3 | **Đạo hàm vượt ±600 mm/s ở frame đổi vật** → `Lift`/`Place` giả | DexYCB 144839 | bỏ tín hiệu đạo hàm quanh frame đổi vật |
| 4 | **`subskill_inference.py` có logic rót RIÊNG**, sửa `hoi_skill_inference.py` không đủ | chạy lại sau khi sửa #1 | áp cùng điều kiện ở cả 2 nơi |
| 5 | **`Align` xuất hiện SAU `Close_Gripper`** — bàn tay không chỉnh hướng lại sau khi đã nắm | DexYCB 155212 | ép thứ tự tăng dần Approach→Align→Close_Gripper |
| 6 | **`IndexError` ở đoạn cuối video** — `end = ce` (mốc đã loại trừ) cộng thêm 1 | DexYCB | kẹp về `min(end+1, n)` |
| 7 | **HUD tự mâu thuẫn** — ghi "Khép ngón" kèm "○ chưa chạm"; và ở pha `Approach` lại ghi "● đang chạm" (rơi về mask 2D khi `gap` là NaN) | DexYCB | HUD lấy cùng nguồn với nhãn sub-skill |

Lỗi #4 và #7 đặc biệt đáng chú ý: cả hai đều là **sửa một nửa** — sửa ở tầng suy
luận mà quên tầng render/tầng sub-skill. Chỉ chạy end-to-end trên dữ liệu mới mới
lộ ra.

### 4b. Bốn MÂU THUẪN giữa các sub-skill (do người dùng phát hiện)

Người dùng xem video và chỉ ra: ở `Orient_Tilt` HUD ghi "chưa chạm"; giữa vùng
rót có đoạn bị gán `Grasp`; và hai pha `Hold_Pour_Angle` + `Return_Upright` có
thật trong video nhưng không được nhận ra. Đúng cả ba. Nguyên nhân là 4 mâu
thuẫn trong chính logic tôi viết:

| # | Mâu thuẫn | Vì sao vô lý | Cách sửa |
|---|---|---|---|
| a | HUD ghi "chưa chạm" lúc rót | Rót **bắt buộc** phải đang cầm. Tôi đã áp luật `touched = (sub == Close_Gripper)` cho **mọi** episode, kể cả Pour | Pour → `touched = True`; Grasp → `Close_Gripper` mới chạm |
| b | Pha `Hold_Pour_Angle` biến mất | Episode rót đòi `held` (độ cao > 15 mm) cho **mọi** frame, nhưng người rót có thể giữ nguyên góc rót khi bình còn nằm sát bàn (đo được: độ cao ≈ 0 suốt 5.2–6.7s). Điều kiện đó cắt mất pha giữ góc | Bỏ `held`, dùng `contact` — tay vẫn trên bình suốt |
| c | Pha `Return_Upright` biến mất | Điều kiện "không nhấc lên" (`vheight < 30 mm/s`) bị áp cho **mọi** frame, nhưng dựng thẳng lại thường đi kèm nhấc bình lên (đo được: `vheight` dương từ 6.4–7.4s) | Chỉ xét "không nhấc" ở **frame bắt đầu** nghiêng — nó dùng để loại ca *nhấc vật đang nằm nghiêng*, không để mô tả toàn bộ diễn biến |
| d | `Grasp` xuất hiện giữa vùng rót (5.77–6.27s) | Lúc giữ nguyên góc rót, tốc độ xoay tụt dưới ngưỡng nên bộ phân loại rơi về nhánh chung và gán `Cầm nắm` — trong khi `dev` lúc đó là 54–61°, rõ ràng đang nghiêng | Thêm **chốt rót**: đã vào rót thì giữ nhãn `Pour` cho tới khi vật dựng thẳng lại (`dev < 20°`) |

Sửa xong, chuỗi rót trong demo thành một đoạn liền `Pour 4.09–7.45s` với đủ 3
sub-skill. Chốt rót (d) còn sửa luôn một lỗi cũ trên `video_test2`: đoạn
4.60–6.33s trước đây gán `Grasp` nhưng `dev` ở đó là 54–61° — tức 2.3 giây bị
gán sai.

**Bài học lặp lại lần thứ ba**: cả 4 mâu thuẫn đều thuộc dạng *áp một luật đúng
cho chỗ này sang chỗ khác mà không kiểm tra*. `held` đúng để loại "bình nằm
nghiêng trên bàn" nhưng sai cho "đang rót"; `not_rising` đúng để loại "nhấc vật
nằm nghiêng" nhưng sai cho "dựng thẳng lại". **Điều kiện để VÀO một pha không
được dùng làm điều kiện để Ở TRONG pha đó.**

### 4c. Hai lỗi về VẬT TRÒN và THAM CHIẾU TRỘN (lộ ra trên video_test3)

| # | Lỗi | Triệu chứng đo được | Cách sửa |
|---|---|---|---|
| 8 | **Tin trục chính của vật TRÒN** | `video_test3`: cốc/ly là vật tròn nên trục chính (PCA) nhảy loạn → sinh `Rót` giả ở 0.70–2.17s trong khi video chỉ có tay với tới cốc | Thêm **độ dài** (trục lớn/nhỏ) vào feature; dưới `ELONG_MIN = 2.0` thì **bỏ góc**. Đo thật: cốc/ly 1.27–1.54, chai 2.71–3.14 |
| 9 | **Mốc tham chiếu tính trên chuỗi TRỘN nhiều vật** | `video_test3`: góc tư thế nghỉ của cốc và của chai lệch ~50°, lấy trung vị chung làm **cả hai đều sai** — chai bị gán "nghiêng 40–60°" suốt video dù đang dựng đứng | Tính `upright_ref` và `d_ref` **riêng cho từng vật** (`_object_references`), rồi tra theo vật đang thao tác |

Lỗi #8 là mặt trái của một giới hạn tôi đã ghi nhận từ trước ("vật gần vuông
xoay không đổi góc → không phát hiện được `Rotate`") nhưng chưa lường hết: không
phải chỉ *bỏ sót*, mà còn **sinh dương tính giả**, vì trục chính của vật tròn
không phải là "không đổi" mà là **đổi loạn**.

Lỗi #9 cùng họ với vấn đề `d_ref` nêu ở mục "độ cao" — cùng một gốc: **chuỗi
theo "vật đang thao tác" trộn nhiều vật, nhưng mọi giá trị THAM CHIẾU phải tính
theo từng vật.**

### 4d. Ba lỗi nữa khi làm tham số skill

| # | Lỗi | Triệu chứng | Cách sửa |
|---|---|---|---|
| 10 | **Bảng tên hướng bị ĐẢO** | Tay hạ xuống chai **từ trên-phải** nhưng hệ ghi "từ dưới-phải". Góc đo ra 46° nhưng bảng tra lại ánh xạ 45° → "dưới-phải" | Sửa bảng theo quy ước toán (y hướng lên): 45° = trên-phải, 90° = phía trên |
| 11 | **Hướng tiếp cận gán cho MỌI đoạn** | `Pour`, `Place` đều có dòng "tới từ ..." — vô nghĩa vì tay đã ở trên vật từ trước, con số chỉ là vị trí hiện tại | Chỉ gán cho đoạn **chứa frame chạm ĐẦU TIÊN** của vật |
| 12 | **O(n²) làm treo script** | `skill_params.py` gọi `frame_features` cho **cả video** bên trong vòng lặp từng frame → treo ở 120 s | Tính feature và mốc tham chiếu **một lần** ở ngoài vòng lặp |

Lỗi #11 lặp lại lần thứ tư cùng một dạng đã gặp ở mục 4b: **một đại lượng chỉ
có nghĩa ở một thời điểm cụ thể lại bị gán cho mọi thời điểm**. Hướng tiếp cận
chỉ tồn tại ở khoảnh khắc bắt đầu chạm; ở các đoạn sau nó không phải là "hướng
đi tới" mà là "đang ở đâu".

### 4e. Điểm chạm rơi RA NGOÀI vật (do người dùng hỏi mới lộ ra)

| # | Lỗi | Triệu chứng đo được | Cách sửa |
|---|---|---|---|
| 13 | **Trọng tâm vùng chạm rơi ra ngoài vật** | Ở t=5.0s và t=8.5s, điểm trả về nằm **trong mask TAY và ngoài mask CHAI** | Lấy **medoid** — pixel THẬT của vùng chạm gần trọng tâm nhất |

Nguyên nhân: khi bàn tay **ôm quanh** vật, vùng chạm là một hình **vành khuyên**
theo rìa vật. Trọng tâm của một vành khuyên nằm ở **giữa** — tức rơi vào bàn tay
chứ không phải trên vật. Đo được: 2/4 mốc kiểm tra bị sai (t=2.0 và 3.0 thì
đúng, t=5.0 và 8.5 thì sai), và đúng ở những mốc tay nắm CHẶT — tức là lỗi chỉ
lộ ra ở chính tình huống quan trọng nhất.

Sau khi sửa, cả 4 mốc đều nằm trên chai. Đây là lỗi thứ năm thuộc cùng một họ:
**lấy trung bình/trọng tâm của một tập hợp mà không kiểm tra kết quả có còn nằm
trong tập hợp đó không.**

### 4f. "Lấy tâm tay suy ra điểm gắp" — đo, kết luận SAI, rồi đo lại

⚠️ **Mục này ghi lại cả một kết luận sai của tôi, để không ai lặp lại.**

Lần đầu tôi đo "trôi" bằng **toạ độ ảnh (px)** và kết luận tâm tay vô dụng:

| Cách đo (SAI — đo trong hệ ẢNH) | Trôi |
|---|---|
| Tâm mask tay | 44.4 px |
| Medoid chiếu lên trục vật | 0.309 |
| Trọng tâm vùng chạm | 0.097 |

**Sai ở chỗ:** trong pha rót, **bản thân cái chai đang di chuyển và xoay**. Đo
trôi trong hệ ảnh thì mọi điểm trên chai đều "trôi" — kể cả một vết sơn dán
chết trên chai. Con số 44 px **lẫn chuyển động của vật với sai số của phép đo**.

Đo lại ĐÚNG — quy về **hệ của vật** trước rồi mới đo, và đo **cả hai chiều**:

| Cách đo (ĐÚNG — hệ VẬT) | Trôi dọc trục | Trôi ngang |
|---|---|---|
| **Tâm mask tay** | **0.020** | 0.085 |
| Trọng tâm vùng chạm | 0.038 | **0.039** |
| Medoid | 0.267 | — |

Kết luận đúng **ngược lại một nửa** so với lần đầu:
- **Tâm tay ổn định hơn theo chiều DỌC trục** (0.020 so với 0.038) — đề xuất của
  người dùng có cơ sở thật, không phải vô dụng như tôi nói.
- Nhưng **kém ổn định hơn theo chiều NGANG** (0.085 so với 0.039), và giá trị
  ngang của nó vô nghĩa: tâm tay nằm lệch **+0.42…+0.50** so với trục vật (nó ở
  trên bàn tay, bên cạnh chai), trong khi trọng tâm vùng chạm nằm ở **+0.03…+0.09**
  — tức đúng chỗ tay chạm vào chai.

→ Chọn **trọng tâm vùng chạm**: tổng sai số hai chiều nhỏ hơn
(√(0.038²+0.039²)=0.055 so với √(0.020²+0.085²)=0.087), và chiều ngang của nó
mới là đại lượng có nghĩa.

**Medoid không dùng để đo** (dù dùng để vẽ): khi tay ôm quanh vật, vùng chạm là
một **vành khuyên**, medoid nhảy lung tung quanh vành — kém ổn định nhất.

#### Điểm VẼ phải neo vào trục vật, không vẽ ở medoid

Sửa phép ĐO chưa đủ: video vẫn thấy điểm gắp trôi, vì **điểm vẽ** vẫn là medoid.
Đã sửa lần nữa — điểm vẽ = **giao của toạ độ `along` đã đo với TRỤC CHÍNH của
vật**. Nhờ vậy điểm vẽ ổn định theo `along` và **dán chặt vào vật** khi vật xoay
hoặc di chuyển (đã kiểm bằng mắt qua 4 mốc 4.2→6.6s).

### 4g. Gom nhiều vị trí gắp từ một video (`grasp_episodes`)

Mỗi **lần cầm** (đoạn tiếp xúc liên tục đã xác thực) cho một điểm gắp, quy về
**hệ của vật**. Một video người cầm vật nhiều lần sẽ cho một TẬP điểm gắp:

```
video_test3 — 5 lần cầm:
  ly thuỷ tinh  0.17-0.37s  along=0.066  phần dưới · bên phải
  ly thuỷ tinh  0.47-2.03s  along=0.183  phần dưới · bên phải
  chai nước     3.36-5.36s  along=0.149  phần dưới
  chai nước     6.23-8.43s  along=0.382  thân          <- GẮP Ở ĐỘ CAO KHÁC
  ly thuỷ tinh  8.30-8.60s  along=0.089  phần dưới · bên phải
```

Ba lần cầm ly cho `along` 0.066–0.183 (nhất quán), hai lần cầm chai cho
0.149 và 0.382 — **cùng một video, hai vị trí gắp khác nhau trên cùng vật**.
Đây chính là cách mở rộng ra nhiều vị trí gắp.

---

## 5. Trường hợp THẤT BẠI — và vì sao

3/5 sequence DexYCB cho kết quả sai. **Nguyên nhân không nằm ở tầng skill** mà ở
tầng phát hiện vật:

Grounding DINO **không tìm ra vật thật sự được cầm**. Ví dụ sequence `144839`:
người cầm **hộp can**, nhưng detector chỉ ra `scissors` / `wood block` /
`power drill` — hộp can không được phát hiện. Pipeline bám theo các vật sai, và
bàn tay quét qua chúng trên đường tới hộp can sinh ra `Push`/`Lift` giả.

Đây là **giới hạn của tầng nhận diện vật, không phải của tầng suy luận skill**.
Cách xử lý: đưa đúng tên vật vào `--prompt` của `gsam2_video.py`, hoặc hạ
`--box-threshold` để không sót vật.

Sequence `155212` chạy tốt vì detector bắt đúng cây kéo — vật duy nhất được cầm.

---

## 6. Điều KHÔNG thể kiểm chứng bằng bất kỳ video nào

Các tiêu chí cần **trạng thái ngón tay** đều không đo được. Đã đo bằng MediaPipe
HandLandmarker trên chính dữ liệu này:

| Phép đo | Tay đeo găng (demo) | Tay trần (video_test2) |
|---|---|---|
| MediaPipe "phát hiện" tay | 336/347 frame (97%) | 156/297 frame (53%) |
| Độ cuộn ngón khi **đang nắm** vật | **~0.05** — ngón duỗi thẳng | 0.30–0.51, nhiễu |
| Độ cuộn ngón khi tay **mở** | ~0.06 | ~0.39 |

Tay đeo găng: model vẽ **đúng vị trí** bàn tay nhưng dựng skeleton ngón duỗi
thẳng lên một bàn tay đang nắm chặt — độ cuộn **không đổi suốt video**. Tỉ lệ
"phát hiện 97%" là chỉ số gây nhầm: bắt được tay không có nghĩa landmark đúng.

Hệ quả: `Close_Gripper` phải gộp cả pha "đang khép" lẫn "đã khép", và
`Open_Gripper` (trong `Place`, `Release`) không cài được. **Đây là lỗ hổng không
lấp được bằng thêm ngưỡng** — cần đổi nguồn dữ liệu (cảm biến găng, hoặc camera
thứ hai).

---

## 7. Chạy lại để tái lập

```bash
cd video_pipeline_ai/github/video_pipeline

# --- Demo (bag ROS2) ---
python3 mcap_to_mp4.py ../../demo_20260912_172945_0.mcap --out ../../out/demo_color.mp4
python3 depth_from_mcap.py ../../demo_20260912_172945_0.mcap --out ../../out/demo_depth.npz
cd ../sam_dino && ../../venv/bin/python gsam2_video.py ../../out/demo_color.mp4 \
    --prompt "hand . plastic water bottle . white cup ." --interval 1.0 \
    --outdir ../../out/gsam2_demo
cd ../github/video_pipeline
python3 hoi_skill_inference.py ../../out/gsam2_demo/masks_gsam2.json \
    --depth ../../out/demo_depth.npz --out ../../out/skills_lowlevel.json
python3 subskill_inference.py ../../out/gsam2_demo/masks_gsam2.json \
    --depth ../../out/demo_depth.npz --out ../../out/subskills.json
python3 render_hoi_video.py ../../out/demo_color.mp4 \
    --masks ../../out/gsam2_demo/masks_gsam2.json --skills ../../out/skills_lowlevel.json \
    --depth ../../out/demo_depth.npz --subskills ../../out/subskills.json \
    --out ../../out/demo_subskill.mp4 --scale 2

# --- DexYCB (RGB-D rời) ---
python3 dexycb_to_video.py <thư_mục_sequence> --camera 932122060861 \
    --out ../../out/dexycb/seq.mp4
# rồi chạy gsam2_video.py → hoi_skill_inference.py → subskill_inference.py → render
```

---

## 8. Tóm tắt một dòng

**9/12 skill cấp thấp và 6/6 sub-skill đã cài đều đã kiểm chứng chạy tốt**, trên
2 bộ dữ liệu RGB-D độc lập cùng một bộ tham số. 3 skill chưa kiểm chứng
(`Push`, `Pull`, `Release`) là do thiếu dữ liệu, không phải lỗi code. Giới hạn
thật còn lại: mọi thứ cần **trạng thái ngón tay**, và mọi thứ cần **detector
tìm đúng vật đang được cầm**.

Chuỗi rót đầy đủ trong demo — 11 đoạn, có đủ 3 pha:

```
Idle → Reach → Contact → Grasp → Lift → MoveToTarget
     → Pour [Orient_Tilt → Hold_Pour_Angle → Return_Upright]
     → MoveToTarget → Place → Retract → Idle
```
