# Thẻ vật và vùng nắm: nguyên lý, so sánh và hạn chế

## 1. Mục tiêu

Với mỗi lần cầm vật trong video, hệ thống cần xác định **vùng nắm** nằm ở đâu trên vật, tính bằng mm và tỉ lệ. Robot cần biết điều này để kẹp đúng chỗ. Cách cũ đo vị trí nắm trên mask 2D, nên kết quả phụ thuộc góc camera và phần vật nhìn thấy. Cách mới gắn vùng nắm vào **bề mặt vật** để kết quả không đổi khi vật bị che.

## 2. Bảng so sánh chính

Thử nghiệm giả lập che đáy chai dần từ 0 đến 60 px. Mỗi hàng là một mức che.

| Che đáy | Độ dài chai nhìn thấy | Độ dài B–T đo được | along cũ (0,391 × phần nhìn thấy) | Neo bàn: B + 62,5 mm | Neo nắp: T − 134,8 mm | along mới: B + 0,317 × B–T |
|---|---|---|---|---|---|---|
| 0 px | 197,7 mm | 197,3 mm | 76,4 mm (+14,3) | 62,1 mm (0,0) | 62,1 mm (0,0) | 62,1 mm (0,0) |
| 15 px | 190,4 mm | 197,7 mm | 81,0 mm (+18,9) | 61,9 mm (−0,2) | 62,3 mm (+0,2) | 62,1 mm (0,0) |
| 30 px | 157,1 mm | 198,2 mm | 101,2 mm (+39,1) | 61,7 mm (−0,4) | 62,6 mm (+0,5) | 62,0 mm (−0,1) |
| 45 px | 139,0 mm | 197,9 mm | 112,1 mm (+50,0) | 62,1 mm (0,0) | 62,7 mm (+0,6) | 62,3 mm (+0,2) |
| 60 px | 122,4 mm | 198,2 mm | 122,1 mm (+60,0) | 62,2 mm (+0,1) | 63,1 mm (+1,0) | 62,5 mm (+0,4) |

Giá trị trong ngoặc là chênh lệch so với hàng 0 px. Cột "Neo bàn" và "Neo nắp" dùng cùng một vị trí nắm, đo từ hai đầu của vật.

**Đọc bảng:** khi che đáy 60 px, độ dài nhìn thấy giảm từ 197,7 xuống 122,4 mm, nhưng B–T gần như không đổi (197–198 mm). Cách cũ trôi +60 mm, cách mới chỉ lệch trong khoảng −0,4 đến +1,0 mm.

## 3. Nguyên lý: along cũ, along mới và neo

Để so sánh, cần hiểu từng đại lượng đo gì và lấy gốc ở đâu.

**Neo B (đáy thật).** Mặt bàn được khớp bằng RANSAC quanh chân vật, pháp tuyến bàn làm chiều "lên". Trục vật (lấy bằng PCA trên các điểm 3D) cắt mặt bàn tại một điểm; đó là B. Điểm này xác định theo mặt bàn và trục vật, nên vẫn tính được khi đáy bị tay che.

**Neo T (nắp).** Lấy phân vị 99,5% của các điểm trên trục vật. Phân vị thay cho điểm cao nhất để bỏ nhiễu.

**along mới.** Vị trí vùng nắm tính từ B, chia cho độ dài B–T. Ví dụ: đỉnh vùng nắm cách B 62,5 mm, B–T là 197,3 mm, nên along = 62,5 / 197,3 ≈ 0,317. Vì gốc là đáy thật, hệ số này không đổi khi đáy bị che.

**along cũ.** Lấy điểm thấp nhất nhìn thấy làm gốc, rồi đo vùng nắm và chuẩn hoá theo phần nhìn thấy. Khi đáy bị che, điểm thấp nhất nhìn thấy đi lên, gốc thay đổi, và vị trí đo được cũng đổi theo. Trong bảng, cột along cũ tăng đúng bằng độ dài bị che (+60 mm khi che 60 px).

**Ví dụ cụ thể.** Từ 0 đến 60 px che đáy, đoạn chai nhìn thấy giảm từ 197,7 xuống 122,4 mm, còn B–T vẫn khoảng 198 mm. Đáy nhìn thấy đã đi lên, nên cách cũ đọc vị trí nắm từ một gốc khác và sai thêm đúng phần đáy đã mất (khoảng +60 mm trong bảng). Cách mới vẫn đo từ B thật nên giữ được khoảng 62 mm.

**Neo nắp.** Đo khoảng cách từ T xuống vùng nắm. Đây là kiểm tra chéo: nếu hai neo cho cùng một vị trí thì kết quả đáng tin.

Cách cũ và cách mới đối chiếu theo các tiêu chí sau:

| | Cách cũ | Cách mới |
|---|---|---|
| Điểm đại diện vùng nắm | trung vị của mọi điểm đỏ (cả hai mảng) | đỉnh của mảng chính (vùng tay che dày nhất) |
| Mốc dưới | điểm thấp nhất nhìn thấy | B: trục cắt mặt bàn |
| Mốc trên | điểm cao nhất nhìn thấy | T: phân vị 99,5% |
| Độ dài chia | 202,9 mm (theo phần nhìn thấy) | 197,3 mm (B–T) |
| Chiều "lên" | trục y của ảnh | pháp tuyến bàn |
| Kết quả chai demo | 0,391 (≈ 79 mm từ đáy nhìn thấy) | 0,317 (62,5 mm từ đáy thật) |

## 4. Các chức năng đã dùng

**a. Chọn frame tham chiếu và cửa sổ giữ.** Frame tham chiếu là frame gần nhất trước lúc chạm mà tay chưa chồng lên vật, nên bề mặt vật chưa bị che. Cửa sổ giữ là các frame từ lúc chạm mà vật còn đứng yên. Nhờ vậy so được từng pixel mà không cần căn ảnh.

**b. Phát hiện vùng bị che, bỏ phiếu ba tín hiệu.** Một pixel được coi là bị che nếu có ít nhất hai trong ba tín hiệu: mask tay phủ lên, độ sâu cho thấy có vật gần camera hơn bề mặt 8–80 mm (ngón tay), hoặc ngoại hình ô 8×8 đổi so với ảnh tham chiếu (đã bù độ sáng).

**c. Vùng nắm theo tần suất.** Chỉ tính các frame đã nắm chắc (vùng che ≥ 60% mức lớn nhất, để bỏ các frame đầu khi tay còn khép). Pixel bị che ở ≥ 50% các frame này thuộc vùng nắm.

**d. Tách mảng dọc trục.** Vùng nắm được chiếu lên trục vật, chia histogram thành các bin 5 mm. Cắt ở bin rỗng và khe sâu. Mỗi mảng có mép dưới, mép trên, đỉnh, along và đường kính tại đỉnh. Trên chai demo có hai mảng: mảng chính (eo chai, 58% điểm, along 0,317, đường kính 50,9 mm) và mảng phụ (thân trên, 36% điểm, along 0,596, đường kính 55,8 mm).

**e. Đặc trưng DINOv2.** Mỗi thẻ lưu 32×21 patch × 768 chiều để so khớp ngoại hình giữa các góc nhìn. Bước so khớp và chuyển nhãn sang camera khác chưa có trong repo.

## 5. Kết quả đạt được

- Vị trí nắm không còn phụ thuộc phần vật nhìn thấy: với che đáy 60 px, sai số giảm từ +60 mm xuống trong ±1 mm.
- Tách được hai vùng nắm trên chai và đo đường kính tại mỗi vùng, phục vụ chọn độ mở kẹp.
- Mọi kết quả kiểm chứng được bằng ảnh xem trước (tham chiếu, vùng nắm, tần suất che, lúc đang cầm).

## 6. Hạn chế

- **Chỉ một lần cầm có độ sâu đầy đủ** (chai demo). Ba thẻ còn lại (hai lần cầm chai và một ly trong video test) không có độ sâu, nên không có neo mm và không so được đầy đủ giữa hai cách.
- **Hai số "cách cũ" của chai demo không khớp:** 0,391 trong bảng trên (theo phần nhìn thấy) và 0,298 trong thẻ (tâm khoảng 2D, lấy giữa khoảng 0,10–0,54). Cần thống nhất định nghĩa trước khi báo cáo số cuối.
- **Trường `axis_tilt_deg` trong thẻ chai demo bằng null**, trong khi bảng ghi 5,7° ở mục bàn. Nên đổi tên thành "axis tilt" vì đây là góc giữa trục chai và pháp tuyến bàn, không phải độ nghiêng của bàn.
- **Phía sau vật không đo được:** ngón tay ở mặt khuất camera không nằm trong vùng nắm.
- **Cần frame tham chiếu chưa bị che:** nếu vật bị tay che từ đầu video thì không có ảnh tham chiếu sạch.
- **Neo đáy phụ thuộc mặt bàn:** nếu vật đặt trên khay hoặc hộp, hoặc không khớp được mặt bàn, hệ thống dùng điểm thấp nhất nhìn thấy làm dự phòng, và neo này không bền khi bị che. Trường hợp này được ghi cảnh báo trong thẻ.
- **Sai số nhỏ còn lại:** cột along mới dao động ±0,4 mm dù B–T gần như cố định, do làm tròn và do đo lại mỗi hàng.
- **Chưa có bước chuyển vùng nắm sang camera khác.** Đặc trưng DINOv2 đã lưu nhưng chưa dùng để so khớp.
- **Mẫu còn nhỏ:** một loại chai, một lần cầm có độ sâu đầy đủ. Các ngưỡng (60% đã nắm chắc, 50% tần suất, 8–80 mm, RANSAC 4 mm) chọn theo kinh nghiệm.

## 7. Kết luận ngắn

Cách mới cho kết quả ổn định hơn rõ rệt khi vật bị che, vì neo vào mặt bàn và đỉnh thay vì phần nhìn thấy. Kết quả hiện chỉ được xác nhận trên một lần cầm có độ sâu. Trước khi dùng cho robot cần thống nhất số "cách cũ", chạy thêm các lần cầm có độ sâu, và hoàn thiện bước chuyển vùng nắm sang camera khác.
