# segment_and_overlay_vi_v3.py — Nhật ký nâng cấp

Bản v3 bắt đầu từ v2 (chỉ in tên skill chung chung, vd "Grasp") và được nâng
cấp qua nhiều vòng lặp thực tế: chạy trên video thật → xem kết quả sai ở
đâu → sửa → chạy lại kiểm chứng. Dưới đây là toàn bộ các nâng cấp theo thứ
tự, kèm lý do (vấn đề gặp phải) cho từng cái.

## 1. Overlay chi tiết: Skill + vật thể + vị trí chạm

**Vấn đề:** v2 chỉ hiện "Skill: Cầm nắm", không biết đang cầm gì, ở đâu.

**Giải pháp:** Mỗi cửa sổ 3 frame, hỏi VLM đồng thời skill + tên vật thể
trong 1 request JSON `{"skill": ..., "object": ...}`. Sau đó bổ sung thêm
trường `touch_location` (vị trí chạm cụ thể, vd "quai cốc") — hiện thành
2 dòng riêng biệt trên overlay, không dính chung 1 câu.

## 2. Sửa nhầm lẫn Lift ↔ Place ở frame cuối

**Vấn đề:** Cửa sổ trượt 3 frame cần khung "trước-giữa-sau"; ở mốc cuối
cùng không có frame "sau" (video hết đúng lúc) nên VLM chỉ thấy 2 khung,
dễ đoán nhầm "đang nhấc lên" thay vì "đang đặt xuống".

**Giải pháp:** Đọc thêm 1 frame thật ở ~0.5s sau mốc cuối làm "khung sau"
dự phòng (`tail_peek_frame`). Đồng thời đưa `description` của từng skill
từ `skills.yaml` vào prompt để VLM phân biệt rõ hơn dựa trên định nghĩa
gốc, không chỉ tên gọi.

## 3. Hiệu ứng hiển thị: từ "luôn hiện" → "chớp đúng lúc" → "chỉ chớp khi có sự kiện mới"

**Vấn đề (3 vòng lặp):**
- Ban đầu chữ vị trí chạm hiện liên tục suốt cả đoạn skill (vd suốt 5s Pour) — rối mắt.
- Sửa thành chớp trong khoảng cố định (~0.5s) quanh mỗi mốc sample — nhưng bật/tắt đột ngột như nhấp nháy.
- Sửa fade in/out mượt bằng alpha blending — nhưng vẫn chớp lặp lại ở MỌI mốc sample có cùng giá trị (vd giữ nguyên "thân" suốt 5 mốc liên tiếp thì nháy 5 lần).

**Giải pháp cuối cùng:** Chỉ chớp ở mốc ĐẦU TIÊN của mỗi lần chạm mới —
so sánh giá trị `touch_location` với mốc liền trước, giá trị không đổi
thì không hiện lại. Đồng thời giới hạn chỉ hiện cho các skill có thao tác
chạm tay trực tiếp (`TOUCH_DISPLAY_SKILLS`), không hiện cho Reach/Pour/Lift.

## 4. Từ "hỏi VLM mô tả vị trí chạm bằng chữ" → "hybrid MediaPipe + VLM grounding"

**Vấn đề cốt lõi:** Hỏi thẳng VLM "tay đang chạm vào đâu" bị hallucination
nặng — model 7B có thiên kiến mạnh (vd "cầm chai" → mặc định đoán "nắp
chai" dù ảnh cho thấy tay còn cách xa chai, hoặc đang cầm thân/nhãn).

**Giải pháp (nhiều lớp, xây dần):**
1. Dùng **MediaPipe Hands** (chạy local, CPU, model `hand_landmarker.task`
   ~7.6MB) để lấy toạ độ pixel THẬT của bàn tay — không hỏi VLM đoán điểm.
2. Dùng VLM (grounding format `bbox_2d` gốc của Qwen2.5-VL) để định vị
   bbox của vật thể.
3. Đo khoảng cách THẬT bằng code (không qua VLM) giữa tay và bbox vật thể
   — nếu quá xa thì kết luận CHƯA chạm, trả rỗng, không tốn thêm request
   mô tả nào (chặn hallucination tận gốc thay vì lọc sau).
4. Ngưỡng khoảng cách "coi là chạm" **co giãn theo kích thước vật thể**
   (không dùng số pixel cố định) — vật to hơn/video độ phân giải cao hơn
   thì ngưỡng cũng phải lớn theo tỉ lệ.

## 5. MediaPipe không ổn định 100% — vá bằng nhiều lớp dự phòng

**Vấn đề:** Ngay cả khi tay đang chạm thật, MediaPipe đôi khi vẫn không
phát hiện được (đặc biệt khi ngón tay cuộn/che khuất quanh vật nhỏ như
nắp chai) — cùng 1 tư thế, có lúc ra 21 điểm, có lúc ra 0.

**Giải pháp 3 lớp:**
1. Hạ `min_hand_detection_confidence` từ mặc định 0.5 xuống 0.1.
2. Thử thêm vài frame liền kề ngay sau mốc sample (~150ms, tư thế gần
   như không đổi) thay vì chỉ thử đúng 1 frame (`detect_fingertips_px_multi`).
3. Khi vẫn thất bại hoàn toàn ở mọi frame thử: dùng lại vị trí đã được
   xác thực gần nhất trong CÙNG chuỗi Grasp/thao tác liên tục trên cùng
   vật thể (giả định hợp lý: tay ít khi đổi vị trí nắm giữa lúc giữ yên).

## 6. Phân loại "bộ phận đang chạm" — từ suy luận hình học đơn giản đến định vị từng bộ phận

**Vấn đề (nhiều vòng lặp):**
- Suy luận theo TỈ LỆ VỊ TRÍ trên 1 bbox tổng của vật thể (gần đỉnh =
  nắp, còn lại = thân) sai hệ thống: khi tay cầm giữa thân, bbox vật thể
  do VLM trả về luôn bị "cắt cụt" ngay tại chỗ tay bắt đầu che khuất
  (không phải đáy/nắp thật) → suy luận tỉ lệ luôn sai lệch đúng lúc quan
  trọng nhất.
- Hỏi thẳng VLM "đây là nắp/thân/đáy nào" trên ảnh cắt cận cảnh: model 7B
  vẫn hallucination ra "nắp" bất kể ảnh thật, kể cả khi ép chọn trong
  danh sách đóng hoặc cho xem cả vật thể kèm định nghĩa từng phần.

**Giải pháp cuối cùng — định vị TRỰC TIẾP từng bộ phận đặc trưng:**
- Mỗi LOẠI vật thể có tập bộ phận đặc trưng riêng (đúng yêu cầu người
  dùng): cốc → `["quai", "miệng"]`, chai → `["nắp"]` (bỏ "đáy" — luôn bị
  che khi cầm thân, không đáng tin).
- Hỏi VLM định vị bbox riêng cho TỪNG bộ phận (vd "nắp của chai nước"),
  rồi so khoảng cách điểm chạm tới bbox đó — không suy luận tỉ lệ gián
  tiếp qua bbox tổng nữa.
- Lọc bỏ bbox "bộ phận" nào chiếm >60% diện tích bbox tổng (dấu hiệu VLM
  không định vị được riêng phần đó, trả nhầm cả vật thể).
- "thân" KHÔNG nằm trong danh sách so khoảng cách — nó luôn là bộ phận
  MẶC ĐỊNH khi không khớp bộ phận đặc trưng nào (nếu để "thân" thi đấu
  cùng, nó luôn thắng do bbox thân chiếm gần hết diện tích, at dist≈0).
- Khi MediaPipe phát hiện được tay nhưng ĐẶT SAI vị trí từng khớp riêng
  lẻ (do ngón tay bị che khuất, occlusion) khiến "điểm gần nhất" bị lệch:
  dùng cả **vùng bao quanh toàn bộ bàn tay** (không chỉ 1 điểm) để so
  chồng lấn (overlap) với bbox từng bộ phận — đáng tin hơn nhiều so với
  chỉ 1 điểm có thể bị lệch.

## 7. Thứ tự xử lý: tính vị trí chạm SAU khi đã khử nhiễu skill

**Vấn đề:** Ban đầu tính vị trí chạm ngay trong lúc phân loại skill thô,
rồi mới khử nhiễu skill (vd 1 mốc bị đổi thành "Grasp" do khử nhiễu theo
2 mốc lân cận) — mốc đó đã bị bỏ lỡ hoàn toàn bước xác thực vị trí chạm
từ trước, giữ nguyên câu mô tả thô chưa kiểm chứng dù skill hiển thị đã
đúng là Grasp.

**Giải pháp:** Tách quy trình thành 3 giai đoạn rõ ràng: (1) phân loại
skill thô cho mọi mốc, (2) khử nhiễu skill xong mới đến khử nhiễu tên vật
thể, (3) DỰA TRÊN skill đã khử nhiễu mới tính vị trí chạm.

## 8. Khử nhiễu tên vật thể (object hallucination)

**Vấn đề 1:** VLM đôi khi bịa tên vật thể hoàn toàn không có trong khung
hình (vd gọi "bóng" ở khung đầu tiên dù ảnh chỉ có tay và chai nước).

**Vấn đề 2:** Trong 1 chuỗi thao tác liên tục cầm-nắm 1 vật (Grasp → Lift
→ Pour → Place), 1 mốc giữa chừng đột nhiên đổi tên vật thể khác hẳn (vd
đang rót "chai nước" suốt, lúc Place lại gọi thành "nắp chai" — vô lý về
mặt vật lý vì nắp chai không có trong tay lúc đó).

**Giải pháp 2 lớp:**
- `smooth_isolated_object_noise`: khử nhiễu tổng quát không phụ thuộc
  skill — mốc đầu tiên nếu khác hẳn mốc kế tiếp thì lấy theo mốc kế tiếp;
  mốc giữa nếu khác cả 2 mốc lân cận trong khi 2 mốc đó trùng nhau thì
  sửa theo. (Từng thử áp dụng đối xứng cho cả mốc CUỐI nhưng gây lỗi lan
  truyền ngược khi chính mốc liền trước mới là mốc sai — đã bỏ quy tắc đó.)
- `smooth_object_labels`: ràng buộc vật lý riêng cho chuỗi thao tác giữ
  vật liên tục (`HOLDING_SKILLS`) — vật thể không thể tự đổi tên giữa
  chừng trừ khi mốc kế tiếp cũng xác nhận cùng tên mới (nghĩa là đổi vật
  thật, không phải nhiễu 1 mốc).

## 9. Tự nâng cấp "Reach" → "Grasp" khi có bằng chứng tiếp xúc thật

**Vấn đề:** VLM (kể cả bản 32B) đôi khi phân loại SAI LIÊN TỤC nhiều mốc
liền nhau thành "Reach" dù ảnh cho thấy rõ tay đang nắm chặt vật thể (vd
cầm quai cốc) — vì sai liên tục nên cơ chế khử nhiễu 1-mốc-lẻ không phát
hiện được (cần ít nhất 1 mốc khác biệt xen giữa 2 mốc giống nhau để so
sánh).

**Giải pháp:** Không chỉ tin nhãn skill của VLM — chủ động chạy quy trình
kiểm chứng tiếp xúc (MediaPipe + bbox) ngay cả với mốc bị gắn nhãn
"Reach". Nếu phát hiện tay THỰC SỰ chạm vật (verified=True, có vị trí cụ
thể), tự động nâng cấp nhãn skill từ "Reach" lên "Grasp" — vì theo định
nghĩa "Reach" là tiến lại gần nhưng CHƯA chạm, có chạm thật thì không còn
là Reach nữa.

## 10. Mở rộng skill được phép hiển thị vị trí chạm

**Vấn đề:** Model 32B phân loại kỹ hơn 7B — lúc tay đang vặn nắp chai lại
gọi đúng là "Close" (đóng) thay vì "Grasp", nhưng code cũ chỉ hiện vị trí
chạm khi skill == "Grasp" nên bỏ sót đúng khoảnh khắc quan trọng nhất.

**Giải pháp:** Đổi điều kiện hiển thị từ so sánh đúng 1 skill sang tập
hợp `TOUCH_DISPLAY_SKILLS = {Grasp, Close, Open, Press, Release, Insert,
Remove, Rotate}` — mọi skill có thao tác chạm tay trực tiếp.

## 11. Nâng cấp model: qwen2.5vl:7b → qwen2.5vl:32b

**Lý do:** Người dùng có 16GB VRAM, muốn model chính xác hơn cho các suy
luận không gian tinh vi (vị trí chạm, phân biệt vật thể).

**Cân nhắc:** Qwen2.5-VL trên Ollama chỉ có 4 size chính thức: 3B/7B/32B/
72B (không có size trung gian 13-14B). 32B mặc định ~21GB > 16GB VRAM —
Ollama tự động chia lớp GPU/CPU (không vượt quá VRAM thật có), đổi lại
mỗi lần gọi chậm hơn nhiều (~29s thay vì 1-3s, do 42% lớp phải chạy trên
CPU). Đã cân nhắc dùng `llama3.2-vision:11b` (vừa VRAM, nhanh hơn) nhưng
loại bỏ vì KHÔNG hỗ trợ định dạng grounding gốc (`bbox_2d`/`point_2d`) mà
toàn bộ pipeline định vị vật thể/bộ phận đang phụ thuộc vào — đổi model
họ khác sẽ làm hỏng phần định vị bbox đã xây dựng.

## Giới hạn còn tồn tại (đã xác nhận, không phải bug)

- MediaPipe có thể hoàn toàn không phát hiện được tay ở 1 số góc quay/tư
  thế cụ thể dù đã hạ ngưỡng confidence tối thiểu — nếu đó là mốc chạm
  DUY NHẤT trong cả video (không có chuỗi liền kề để mượn dữ liệu), hệ
  thống phải fallback về mô tả tự do chưa kiểm chứng của VLM, có thể sai.
- 32B chạy chậm (~29s/request) do phải chia sẻ tài nguyên CPU/GPU trong
  16GB VRAM — xử lý 1 video ~10-15 giây có thể mất 10-20 phút.
- Cơ chế "thử nhiều frame liền kề trong ~0.15s" để chống MediaPipe không
  ổn định đôi khi bắt trúng 1 khoảnh khắc chạm rất thoáng qua rồi tay di
  chuyển tiếp ngay sau đó — chữ vẫn hiện đủ hết cửa sổ fade (0.6s) nên có
  thể lệch nhẹ so với đúng lúc tay còn chạm thật trên màn hình.

---

## Phụ lục: cơ chế hoạt động chi tiết

### A. VLM đoán "skill" như thế nào?

Skill KHÔNG được đoán từ 1 ảnh tĩnh duy nhất, mà từ **cửa sổ trượt 3
khung hình liên tiếp** (`classify_window`): khung trước — khung hiện tại
— khung sau. Lý do: nhiều skill trông giống hệt nhau nếu chỉ nhìn 1 ảnh
tĩnh (vd tay đang "giữ" một cái cốc lơ lửng giữa không trung trông giống
hệt nhau dù đó có thể là đang Lift, đang Place, hay đang MoveToTarget) —
chỉ có SỰ THAY ĐỔI giữa 3 khung (vật thể đi lên hay đi xuống, tay đang
khép lại hay mở ra) mới phân biệt được.

Quy trình 1 request cho mỗi mốc thời gian (`build_window_prompt` +
`classify_window`):

1. Gộp 3 ảnh (base64) vào 1 request duy nhất gửi cho Qwen2.5-VL.
2. Prompt liệt kê TOÀN BỘ skill hợp lệ kèm **mô tả gốc từ `skills.yaml`**
   (không chỉ tên) — vd không chỉ nói "Place" mà nói "Place: Set a held
   object down at the current position, releasing it". Việc đưa định
   nghĩa vào giúp VLM có tiêu chí rõ ràng hơn tên gọi trần trụi, đặc biệt
   với các cặp dễ nhầm (Lift ↔ Place, Grasp ↔ Release).
3. Prompt yêu cầu trả về JSON gồm 3 trường trong 1 lần hỏi (tiết kiệm,
   không tốn thêm request riêng): `skill`, `object` (tên vật thể), và
   `touch_location` (mô tả tự do vị trí chạm — trường này CHỈ dùng làm
   phương án dự phòng, vì đây chính là chỗ VLM hay hallucination, đã thay
   bằng cơ chế hybrid ở phần B).
4. Nếu ảnh không rõ hành động (tay chưa vào khung, hoặc đứng yên), VLM
   được phép trả `"Unknown"` thay vì bị ép chọn bừa 1 skill.
5. Kết quả JSON được parse (`parse_skill_object`), đối chiếu tên skill
   trả về với danh sách hợp lệ trong `skills.yaml` (không phân biệt hoa
   thường) — trả về `"Unknown"` nếu VLM bịa ra tên không có trong danh sách.

Vì đây là 1 lần suy luận độc lập cho MỖI mốc thời gian (không có bộ nhớ
giữa các mốc), kết quả có thể nhiễu ở từng mốc đơn lẻ — đây là lý do cần
thêm bước khử nhiễu hậu kỳ (`smooth_skill_labels`, `smooth_object_labels`,
`smooth_isolated_object_noise`, và bước tự nâng cấp Reach→Grasp) chạy
SAU khi đã có toàn bộ chuỗi skill thô của cả video.

### B. Hybrid MediaPipe + VLM grounding hoạt động thế nào?

Mục tiêu: biết CHÍNH XÁC bàn tay có đang chạm vật thể hay không, và nếu
có thì chạm vào bộ phận nào — mà không để VLM tự "tưởng tượng" ra câu trả
lời (lỗi hallucination gặp xuyên suốt phần trước của file này). Ý tưởng
cốt lõi: **mỗi model làm đúng việc nó giỏi nhất** — MediaPipe giỏi định vị
hình học chính xác trên bàn tay (nhưng không hiểu ngữ nghĩa "đây là cái
gì"), còn VLM giỏi hiểu ngữ nghĩa/ngôn ngữ (nhưng định vị hình học tinh
vi thì hay đoán bừa). Quy trình gồm 4 bước, chạy trong
`describe_touch_location_grounded` — CHỈ được gọi khi skill thuộc
`TOUCH_DISPLAY_SKILLS` (Grasp/Close/Open/Press/Release/Insert/Remove/
Rotate), để không tốn tài nguyên cho các skill không liên quan đến chạm:

**Bước 1 — Định vị bàn tay bằng MediaPipe (không qua VLM):**
`detect_fingertips_px_multi` chạy model `HandLandmarker` của MediaPipe
(local, CPU, ~7.6MB) trên khung hình hiện tại VÀ vài khung liền kề ngay
sau đó (~150ms, tư thế gần như không đổi) — thử lần lượt cho tới khi
thành công, vì đã kiểm chứng MediaPipe không ổn định 100% (cùng 1 tư thế
nắm chặt, có lúc phát hiện được cả 21 điểm landmark, có lúc trả về 0).
`min_hand_detection_confidence` hạ xuống 0.1 (thay vì mặc định 0.5) để
ưu tiên không bỏ sót tay đang nắm chặt, chấp nhận đánh đổi vài false
positive. Nếu KHÔNG frame nào trong toàn bộ chuỗi thử phát hiện được tay
→ không có tín hiệu gì để kiểm chứng → dùng tạm câu trả lời tự do của
VLM ở bước classify_window (chưa kiểm chứng, có thể sai) làm phương án
cuối cùng.

**Bước 2 — Định vị vật thể bằng VLM (grounding format gốc của
Qwen2.5-VL):** `locate_object_bbox_px` hỏi VLM trả về bounding box pixel
của vật thể, theo ĐÚNG định dạng `bbox_2d` mà Qwen2.5-VL được huấn luyện
riêng để xuất ra (`[{"bbox_2d": [x1,y1,x2,y2], "label": ...}]`) — đây là
lý do bắt buộc phải dùng model họ Qwen2.5-VL, không thể đổi sang VLM khác
(vd Llama-3.2-Vision) vì các model đó không hỗ trợ định dạng grounding
này, sẽ mất khả năng định vị bbox chính xác.

**Bước 3 — Đo khoảng cách THẬT bằng code (không qua VLM):**
`closest_point_to_bbox` tính khoảng cách Euclidean từ landmark bàn tay
GẦN NHẤT tới bbox vật thể (0 nếu điểm nằm trong bbox). Ngưỡng "coi là
chạm" (`contact_margin_px`) KHÔNG phải số cố định mà co giãn theo kích
thước vật thể (`max(25px, 12% kích thước bbox)`) — vật to hơn / video độ
phân giải cao hơn thì ngưỡng cũng lớn theo tỉ lệ. Nếu khoảng cách vượt
ngưỡng → kết luận CHƯA chạm, trả về rỗng ngay, KHÔNG tốn thêm request VLM
nào để mô tả (vừa tiết kiệm vừa loại bỏ hoàn toàn cơ hội hallucination ở
bước tiếp theo).

**Bước 4 — Chỉ khi xác nhận có chạm thật, định vị bộ phận cụ thể**
(`classify_touch_region_grounded`): thay vì hỏi VLM "bạn đang chạm vào
đâu" (dễ hallucination, đã kiểm chứng thất bại nhiều lần dù ép chọn
trong danh sách đóng), pipeline hỏi VLM định vị bbox TRỰC TIẾP cho TỪNG
bộ phận đặc trưng của loại vật thể đó (`touch_parts_for_object`: cốc →
"quai"/"miệng", chai → "nắp"), rồi:
- Ưu tiên so **chồng lấn (overlap)** giữa bbox bao trọn TOÀN BỘ landmark
  bàn tay (`hand_region_bbox`, không phải 1 điểm) với bbox từng bộ phận
  — đáng tin hơn nhiều so với 1 điểm gần nhất khi ngón tay bị che khuất
  khiến MediaPipe đặt sai vị trí khớp riêng lẻ.
- Nếu không có overlap đáng kể, fallback sang khoảng cách từ điểm chạm
  gần nhất tới bbox bộ phận (`closest_point_to_bbox`).
- Loại bỏ bbox "bộ phận" nào chiếm >60% diện tích bbox toàn vật thể (dấu
  hiệu VLM không định vị riêng được, trả nhầm cả vật thể).
- Nếu không bộ phận đặc trưng nào đủ gần/chồng lấn, mặc định trả "thân"
  (bộ phận trung tính, không tham gia so khớp để tránh luôn thắng do
  bbox thân chiếm gần hết diện tích).

Tóm lại, chuỗi suy luận là: **MediaPipe trả lời "tay ở đâu"** (hình học
chính xác) → **VLM trả lời "vật thể/bộ phận nào ở đâu"** (ngữ nghĩa +
grounding) → **code so sánh khoảng cách/chồng lấn giữa 2 kết quả đó**
(quyết định cuối cùng, không qua VLM) — không bước nào để VLM tự đoán
trực tiếp câu trả lời cuối cùng bằng lời văn.

## 12. Tích hợp suy luận tên task + ghi vào video_task_log.json

**Lý do:** `infer_task_name.py` đã có sẵn (đọc chuỗi skill, hỏi LLM đặt 1
tên task tổng quát, tích luỹ vào `video_task_log.json`) nhưng phải chạy
tay thành 1 lệnh riêng sau khi có `segments_log`. Muốn mỗi lần chạy nhận
diện video mới thì tự động có luôn cả bước này.

**Giải pháp:**
- Thêm `model_name` (mặc định vẫn `qwen2.5vl:7b`) làm tham số cho
  `infer_task_name()` thay vì hard-code — để gọi bằng đúng model đang
  cấu hình trong `segment_and_overlay_vi_v3.py` (hiện là 32B) thay vì
  luôn dùng 7B của riêng `infer_task_name.py`.
- Tách phần ghi file thành `append_task_log()` dùng chung được bởi cả 2
  script, tránh trùng code.
- `segment_and_overlay_vi_v3.py` sau khi ghi xong `segments_log`, tự động
  gọi `infer_task_name` + `append_task_log` — tắt được bằng
  `--skip-task-log` nếu không cần. Vẫn giữ nguyên hành vi **tích luỹ**
  (append) chứ không ghi đè, để giữ lịch sử qua nhiều video.

## 13. Chuẩn hoá tên vật thể theo nhóm từ đồng nghĩa (toàn cục, không chỉ trong 1 chuỗi)

**Vấn đề:** Video dài, nhiều thao tác xen kẽ (video_test5, 29 mốc) làm lộ
ra 1 dạng nhiễu tên vật thể mới: CÙNG 1 vật lý thể (1 cái cốc/ly thuỷ
tinh) bị gọi luân phiên "cốc" → "cốc thuỷ tinh" → "ly" qua nhiều mốc —
không phải nhiễu 1 khung hình lẻ (mục 8) vì mỗi tên đều được 2-4 mốc liên
tiếp "xác nhận" (đúng điều kiện coi là đổi vật thật của `smooth_object_labels`),
mà là do VLM tự thiếu nhất quán cách gọi tên giữa các lần suy luận độc
lập. Thử fix bằng cách thêm "Reach" vào `HOLDING_SKILLS` (không ngắt
chuỗi theo dõi ở bước tiếp cận) chỉ giải quyết được 1 phần: khi có 1 vật
KHÁC (vd chai nước) xen vào giữa rồi quay lại cầm đúng cái cốc lúc nãy,
chuỗi theo dõi đã bị "chai nước" ghi đè, nên lúc quay lại cốc lại bị coi
là "vật mới" và chấp nhận tên khác ("ly") thay vì nhớ lại tên cũ.

**Giải pháp — chuẩn hoá ĐỘC LẬP theo mốc, không cần ngữ cảnh chuỗi:**
Định nghĩa `OBJECT_SYNONYM_GROUPS` — danh sách các nhóm từ đồng nghĩa cho
từng LOẠI vật thể, phần tử ĐẦU TIÊN của mỗi nhóm là tên chuẩn. Hàm
`normalize_object_synonyms` áp dụng tra cứu này cho MỌI mốc ngay sau bước
khử nhiễu 1-khung-lẻ, TRƯỚC bước khử nhiễu theo chuỗi
(`smooth_object_labels`) — vì đây là ánh xạ độc lập theo từng mốc (không
cần biết mốc trước/sau), nó đúng cả khi 1 vật thể bị "bỏ dở giữa chừng"
(chuyển sang thao tác vật khác ở giữa) rồi cầm lại sau đó.

**BÀI HỌC QUAN TRỌNG (đã sửa sai lầm ban đầu):** Lần đầu gộp `"cốc"` và
`"ly"/"cốc thuỷ tinh"` vào CHUNG 1 nhóm — SAI. Kiểm tra lại bằng ảnh thật
phát hiện video_test5 có **2 vật thể HOÀN TOÀN KHÁC NHAU cùng xuất hiện**:
1 cái cốc sứ đục (luôn được VLM gọi nhất quán là "cốc") và 1 cái bình
thuỷ tinh trong suốt có quai (luôn được gọi "ly" hoặc "cốc thuỷ tinh" —
2 tên NHƯNG cùng 1 vật). Gộp chung nhóm đã xoá mất phân biệt THẬT giữa 2
vật thể. Sửa lại: tách thành 2 nhóm riêng — `["cốc", "cốc nước", "ca",
...]` và `["ly", "cốc thuỷ tinh", "ly thuỷ tinh", "bình thuỷ tinh", ...]`
— chỉ gộp đồng nghĩa NỘI BỘ trong từng nhóm, không gộp giữa 2 nhóm.

**Thử và bỏ — theo dõi vị trí bbox thay vì đếm số lần xác nhận tên:**
Từng thử nâng ngưỡng "số mốc liên tiếp xác nhận đổi vật thật" từ 1 lên 2
(nghĩ rằng hallucination tên gọi hiếm khi bền quá 1 mốc) — kết quả TỆ HƠN
(4/29 mốc sai so với 2/29), vì hallucination có thể bền ngẫu nhiên NHIỀU
mốc hơn tuỳ lần chạy, không có ngưỡng cố định nào an toàn. Sau đó thử
định vị bbox thật cho MỌI mốc (không chỉ lúc Grasp) rồi so vị trí không
gian giữa các mốc liên tiếp để quyết định "cùng vật thể" thay vì tin tên
gọi (`resolve_object_identity_by_position`) — cũng THẤT BẠI (≥7/29 mốc
sai): khi 2 vật thể khác nhau được dùng NỐI TIẾP ở gần cùng 1 vị trí trên
bàn (vd đặt bình thuỷ tinh xuống rồi cầm chai nước rót ngay tại chỗ đó),
thuật toán nhầm chúng là "cùng vật thể vì vị trí gần nhau" — vị trí gần
không đảm bảo cùng vật thể khi có nhiều vật tương tự dùng luân phiên.

**Kết luận:** Quay lại cấu hình đơn giản nhất (ngưỡng xác nhận 1 mốc +
`normalize_object_synonyms` với nhóm đồng nghĩa ĐÃ TÁCH ĐÚNG) cho kết quả
thực nghiệm tốt nhất trong tất cả các cách đã thử. Kiểm chứng trên
video_test5 (29 mốc): 1 lần chạy đạt 0 lỗi hoàn toàn so với ảnh thật,
phân biệt đúng cả 3 vật thể (cốc / ly-bình thuỷ tinh / chai nước) xuyên
suốt kể cả khi bị ngắt giữa chừng bởi vật khác. Việc phân biệt vật thể
100% ổn định qua MỌI lần chạy vẫn không đảm bảo tuyệt đối do bản chất suy
luận độc lập, không có bộ nhớ giữa các lần gọi của VLM — đây là giới hạn
thật, không phải bug có thể vá hết bằng heuristic văn bản/hình học đơn giản.


## 14. Ba lỗi lộ ra ở video_test5 (chuỗi thao tác dài, nhiều vật thể)

**Bối cảnh:** Cùng 1 bình nước thuỷ tinh bị đổi tên qua lại giữa "chai
nước" ↔ "cốc/ly", và vị trí cầm/chạm ("thân", "quai", "nắp") không hiện ở
nhiều mốc dù tay đang cầm rõ ràng. Cả 3 lỗi dưới đây đều đã được kiểm
chứng bằng cách chạy lại pipeline đầy đủ trên model 32B và xem video thật.

### 14.1 Nhầm tên vật thể giữa chừng 1 chuỗi cầm liên tục

**Vấn đề:** VLM hallucination tên vật thể BỀN qua nhiều mốc liên tiếp (4
mốc) rồi tự quay lại đúng tên cũ. Vì kéo dài hơn 1 mốc nên vượt qua được
cả ngưỡng khử nhiễu 1-mốc của mục 8 lẫn điều kiện "mốc kế tiếp xác nhận
= đổi vật thật" của `smooth_object_labels` — hệ thống hiểu nhầm là người
dùng đã đổi sang vật khác rồi đổi lại.

**Giải pháp:** Thêm `_bridge_chain_reversions` — trong 1 chuỗi giữ-vật
liên tục, nếu tên đổi sang giá trị khác rồi QUAY LẠI đúng tên ban đầu,
toàn bộ đoạn ở giữa được vá lại theo tên ban đầu. Việc "quay lại đúng tên
cũ" chính là bằng chứng vật thể chưa từng rời tay, mạnh hơn bất kỳ ngưỡng
đếm số mốc nào (đã chứng minh ở mục 13 là không có ngưỡng cố định an toàn).

### 14.2 Mất vị trí chạm khi VLM không định vị được bbox vật thể

**Vấn đề:** `describe_touch_location_grounded` trả về `""` khi không định
vị được bbox vật thể (ảnh mờ do chuyển động) — bỏ luôn cả mô tả tự do
(fallback) mà VLM đã trả ở bước `classify_window`, nên overlay trống trơn
dù có sẵn thông tin dự phòng.

**Giải pháp:** Trả `fallback_text` thay vì chuỗi rỗng ở nhánh này, nhất
quán với nhánh "MediaPipe không thấy tay" vốn đã làm đúng như vậy từ
trước (`verified=False` vẫn giữ nguyên để caller biết là chưa kiểm chứng).

### 14.3 Kết luận sai "chưa chạm" giữa chừng khi đang cầm

**Vấn đề:** bbox vật thể do VLM định vị hơi lệch giữa các lần gọi độc lập
khiến phép đo khoảng cách (bước 3, phụ lục B) vượt ngưỡng và kết luận
"chưa chạm" ở 1 mốc giữa chừng, dù tay vẫn đang giữ nguyên vật đó suốt cả
chuỗi.

**Giải pháp:** Mở rộng cơ chế "mượn vị trí đã xác thực gần nhất trong
CÙNG chuỗi trên CÙNG vật thể" (mục 5, lớp 3) sang cả trường hợp này —
trước đó chỉ áp dụng khi MediaPipe hoàn toàn không phát hiện được tay.

### Kết quả sau 3 sửa đổi

Toàn bộ mốc Grasp/Lift/Pour/Rotate/MoveToTarget/Place/Release trên cả
"ly" và "chai nước" hiện đúng vị trí chạm, tên vật thể nhất quán xuyên
suốt. Còn 2 mốc trống (t=16.1s, 17.6s — ngay lúc vừa chạm chai nước mới,
chưa có lịch sử trong chuỗi để mượn) do bản chất ngẫu nhiên của VLM khi
định vị vật nhỏ/mờ — giới hạn đã biết, không phải bug.

Kết quả kiểm chứng: `log_test5_fix4.json`, `annot_test5_fix4_h264.mp4`.


source ../../venv/bin/activate   # nếu chưa activate

python3 segment_and_overlay_vi_v3.py <đường_dẫn_video_mới>.mp4 \
  --skills ../../skill_ontology/skills.yaml \
  --names-vi ../../skill_ontology/skill_names_vi.yaml \
  --out annot_<tên>.mp4 \
  --log log_<tên>.json

Ví dụ với video mới tên video_test6.mp4 đặt ở video_pipeline_ai/:

python3 segment_and_overlay_vi_v3.py ../video_test6.mp4 \
  --skills ../../skill_ontology/skills.yaml \
  --names-vi ../../skill_ontology/skill_names_vi.yaml \
  --out annot_test6.mp4 \
  --log log_test6.json

Ghi chú:
- Bỏ --log/--out thì dùng mặc định annotated_output_vi_v3.mp4 / segments_log_vi_v3.json.
- Bỏ --skip-task-log (mặc định không có) thì sẽ tự suy luận tên task và ghi tích lũy vào video_task_log.json — thêm --skip-task-log nếu không muốn việc đó.
- Model 32B chậm (~29s/lần gọi) nên video ~25-30s có thể mất 15-20 phút.
- Nếu output .mp4 không phát được (do codec mpeg4), convert thêm:
ffmpeg -i annot_<tên>.mp4 -c:v libx264 -pix_fmt yuv420p annot_<tên>_h264.mp4