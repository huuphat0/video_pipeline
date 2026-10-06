# Nguyên lý nhận diện skill từ video

## 1. Bài toán

Đầu vào là một video quay bàn tay người thao tác với vật thể, ví dụ cầm chai nước, rót sang cốc, rồi đặt xuống bàn. Đầu ra là chuỗi các **skill** theo thời gian, ví dụ:

> Reach → Contact → Grasp → Lift → Pour → Place → Release → Retract

Mục đích là để robot biết người đang làm gì ở từng thời điểm và theo trình tự nào, từ đó học lại được thao tác.

---

## 2. Nguyên lý nhận diện skill

### 2.1 Ý tưởng tổng quát

Hệ thống không đoán skill từ một khung hình đơn lẻ, cũng không dùng mô hình ngôn ngữ để mô tả hình ảnh. Mọi kết luận đều là **phép đo hình học trên pixel và độ sâu**, và skill được suy ra từ cách các đại lượng đó thay đổi theo thời gian: vật đứng yên hay đang đi lên, tay đang tiến tới hay đang rời đi, vật có đang nghiêng không.

Nhờ vậy kết quả có thể kiểm chứng bằng mắt trên từng khung hình, và chạy lại trên cùng một video luôn cho cùng một kết quả.

### 2.2 Các bước xử lý

Nhận diện skill gồm năm tầng, mỗi tầng dùng đầu ra của tầng trước:

1. **Phân đoạn:** tách vùng bàn tay và từng vật thể ở mọi khung hình.
2. **Chọn vật đang thao tác:** tại mỗi thời điểm, xác định vật nào là đối tượng của bàn tay.
3. **Đo tín hiệu:** từ vùng phân đoạn và độ sâu, tính khoảng cách tay–vật, tốc độ vật, độ nghiêng, độ cao thật.
4. **Phân loại theo luật:** gán nhãn skill cho từng khung hình theo một thứ tự ưu tiên xác định.
5. **Hậu xử lý theo thời gian:** làm mịn nhãn, gộp đoạn quá ngắn, ghép thành các đoạn skill liên tục, rồi suy ra sub-skill bên trong.

### 2.3 Phân đoạn bằng mask

Hệ thống dùng Grounded-SAM2 để tạo mask cho bàn tay và các vật thể ở từng khung hình, với ID ổn định xuyên suốt video.

Cách này được chọn thay cho các phương pháp trước đó vì:

- **Bộ nhận diện khớp ngón (MediaPipe)** thường mất dấu tay khi bàn tay nắm chặt vật, vì các ngón bị che khuất. Khi mất dấu, toàn bộ phần kiểm tra hình học không chạy được.
- **Khung bao do mô hình ngôn ngữ đoán** không ổn định giữa các lần gọi: cùng một vật nhưng mỗi lần một khung hơi lệch.
- **Lấy mẫu thưa** (một mốc mỗi 1,5 giây) chỉ nhận ra skill kéo dài, không thấy các pha ngắn như lúc tay vừa chạm hay vừa nhấc.

Mask khắc phục cả ba điểm: tách được tay và vật ngay cả khi đang nắm chặt, giữ ID ổn định, và có ở mọi khung hình (30 fps) nên đo được cả các pha chỉ vài chục mili giây.

### 2.4 Xác định vật đang thao tác

Khi cảnh có nhiều vật, tại mỗi thời điểm hệ thống chọn một vật làm "vật đang thao tác":

- **Khi tay đang chạm vật:** chọn vật có diện tích tiếp xúc lớn nhất với bàn tay, vì tay cầm vật nào thì vùng của hai bên chồng lấn nhiều nhất với vật đó.
- **Giữ lựa chọn cũ:** chỉ đổi sang vật khác khi vật mới vượt trội rõ rệt, để tránh lựa chọn nhảy qua lại giữa hai vật nằm sát nhau.
- **Khi tay chưa chạm vật nào:** chọn vật gần tay nhất về khoảng cách tâm, để biết tay đang tiến tới vật nào.

Nếu chỉ chọn theo khoảng cách tâm, các tín hiệu của hai vật sẽ bị trộn lẫn và chuỗi nhãn trở nên vô nghĩa, ví dụ "Nhấc lên" xuất hiện trước "Cầm nắm".

### 2.5 Các tín hiệu đo được

**Chạm hay không chạm.** Vùng bàn tay được nới rộng vài pixel rồi đếm số pixel chồng lấn với vật. Vượt một ngưỡng nhỏ thì coi là đang chạm. Nếu chỉ có khe hở nhỏ do phân đoạn chưa khít, hệ thống xét khoảng cách nhỏ nhất giữa hai vùng.

**Khoảng cách tay–vật.** Đo giữa tâm bàn tay và tâm vật, không đo khoảng cách nhỏ nhất giữa hai vùng. Lý do: vùng bàn tay thường bao gồm cả cẳng tay, nên khoảng cách nhỏ nhất luôn bị cẳng tay chi phối, khiến pha "tay tiến lại gần" gần như biến mất.

**Tốc độ vật.** Tâm của vùng vật được làm mịn theo thời gian rồi lấy đạo hàm, cho vận tốc theo phương ngang và phương đứng.

**Độ cao thật.** Từ luồng độ sâu của camera, lấy độ sâu trung vị trong vùng vật và so với mốc mặt bàn (lấy từ chính video lúc vật nằm yên). Kết quả tính bằng milimét. Dùng milimét thay pixel vì pixel phụ thuộc khoảng cách tới camera, còn milimét trả lời trực tiếp câu hỏi vật đã rời khỏi bàn chưa.

**Độ nghiêng.** Với vật thon dài như chai, thìa, kéo, trục chính của vùng vật (tính bằng PCA) cho biết vật đang đứng thẳng hay nghiêng. Vật tròn như cốc hoặc ly có hai trục gần bằng nhau, trục chính nhảy loạn giữa các khung hình, nên không dùng để xét nghiêng. Tiêu chí phân biệt là tỉ lệ độ dài trên độ rộng, dưới 2 coi là vật tròn. Từ góc trục chính hệ thống tính tốc độ xoay (độ/giây) và độ lệch khỏi tư thế thẳng (lấy từ lúc vật chưa được cầm).

**Tương quan chuyển động tay–vật.** Đây là tín hiệu trực tiếp định nghĩa "đang cầm": vật được cầm thì đi cùng hướng và cùng tốc độ với bàn tay. Hệ thống tính cosin góc giữa vector vận tốc của tay và của vật. Cosin gần 1 là đang cầm; vật tự trượt trong khi tay chỉ chạm nhẹ thì cosin thấp. Tín hiệu chỉ được tính khi cả hai đều chuyển động đủ nhanh, vì khi đứng yên hướng vận tốc chỉ là nhiễu.

Hệ thống không dùng tư thế ngón tay để xác định cầm, vì khi tay đeo găng hoặc nắm chặt, bộ nhận diện khớp ngón không phân biệt được lúc cầm và lúc buông.

### 2.6 Làm sạch tín hiệu

Mask trong thực tế rung và đôi khi giật bậc thang, nên tín hiệu thô không dùng trực tiếp được. Hệ thống:

- lọc median trên góc trục chính để loại giá trị nhiễu đột ngột;
- làm mịn các chuỗi tâm, khoảng cách và độ cao bằng cửa sổ khoảng 0,23 giây;
- kẹp tốc độ xoay: rót thật hiếm khi vượt khoảng 100 độ/giây, nên các cú giật của mask bị giới hạn;
- lấp khe hở mất chạm rất ngắn (dưới 6 khung hình) và bỏ đoạn chạm quá ngắn (dưới 4 khung hình);
- bỏ đạo hàm tại thời điểm đổi vật, vì mọi tín hiệu nhảy bậc khi hai vật nằm ở hai chỗ khác nhau.

### 2.7 Luật phân loại theo khung hình

Mỗi khung hình được gán một nhãn theo luật hình học. **Thứ tự xét quan trọng**: dấu hiệu đặc trưng được xét trước dấu hiệu chung. Nếu không, mọi khung hình đang cầm đều bị gán "Cầm nắm" và không bao giờ có "Rót".

**Khi tay đang chạm vật**, xét theo thứ tự ưu tiên:

1. **Rót (Pour):** vật đang xoay đủ nhanh, vật đã từng gần thẳng đứng trong 1,5 giây trước đó, và không đang bay lên nhanh. Điều kiện "đã từng gần thẳng" phân biệt rót với việc nhấc một vật đang nằm nghiêng. Điều kiện "không bay lên nhanh" phân biệt rót với việc nhấc một vật dài khỏi bàn. Khi đã vào trạng thái rót, nhãn được giữ cho tới khi vật dựng thẳng lại, để không rơi về "Cầm nắm" khi đang giữ góc rót.
2. **Nhấc lên (Lift):** độ cao thật tăng đủ nhanh.
3. **Đặt xuống (Place):** độ cao thật giảm đủ nhanh. Khi không có độ sâu, dùng tốc độ theo chiều dọc của ảnh thay thế.
4. **Di chuyển ngang (MoveToTarget):** vật đi ngang đủ nhanh.
5. **Đẩy (Push):** vật tự di chuyển trong khi tay vẫn chạm nhưng không đi cùng tay, tức là bị đẩy trượt trên bàn chứ không được cầm.
6. **Chạm (Contact):** khoảng 0,25 giây ngay sau khi vừa chạm, khi vật chưa kịp di chuyển.
7. **Cầm nắm (Grasp):** các trường hợp còn lại, tức là đang giữ vật ổn định.

**Khi tay không chạm vật:**

1. **Buông (Release):** vừa hết chạm mà vật vẫn đứng yên, tức là vật được thả xuống chứ không phải tay rút đi.
2. **Tiến tới (Reach):** khoảng cách tâm tay–vật đang giảm đủ nhanh.
3. **Rời đi (Retract):** khoảng cách đang tăng đủ nhanh, hoặc vừa rời một pha thao tác nhưng chưa đủ tín hiệu để xác định hướng.
4. **Nhàn rỗi (Idle):** không có tương tác đáng kể.

### 2.8 Hậu xử lý theo thời gian

- **Lọc median trên chuỗi nhãn** (7 khung hình) để khử nhấp nháy giữa hai nhãn.
- **Gộp đoạn ngắn:** đoạn ngắn hơn 0,30 giây được gộp vào đoạn liền kề, khi gộp đoạn dài hơn thắng. Riêng các pha chuyển tiếp (Reach, Retract, Contact, Release) có ngưỡng riêng 0,12 giây, để không bị xoá mất.
- **Loại Reach và Retract đứng lẻ:** Reach chỉ có ý nghĩa khi dẫn tới một lần chạm, Retract khi đi sau một lần chạm. Các đoạn đứng lẻ ở đầu hoặc cuối video bị loại.
- **Ghép thành đoạn skill:** các khung hình liền nhau cùng nhãn được gộp thành một đoạn có thời điểm bắt đầu và kết thúc.

### 2.9 Sub-skill

Mỗi skill cấp cao có thể chia thành bước con:

- **Cầm nắm (Grasp)** gồm: Tiếp cận (Approach), Canh vị trí (Align), Khép ngón (Close_Gripper).
- **Rót (Pour)** gồm: Nghiêng để rót (Orient_Tilt), Giữ góc rót (Hold_Pour_Angle), Dựng thẳng lại (Return_Upright).

Tiêu chí gốc của sub-skill dựa trên trạng thái ngón tay, nhưng ngón tay không đo được đáng tin trong dữ liệu này. Hệ thống thay bằng khoảng cách thật theo độ sâu giữa các điểm tay sát vật và vật:

- không có điểm tay nào sát vật: Tiếp cận;
- khoảng cách lớn, tay còn ở trên cao: Canh vị trí;
- khoảng cách nhỏ, tay ngang tầm vật: Khép ngón.

Dùng độ sâu vì cách dựa trên chồng lấn 2D bị sai khi bàn tay nằm ngay phía trên vật: trong ảnh, bàn tay trùng với vật nên bị báo là "đã chạm", trong khi thực tế còn cách khoảng 10 cm.

### 2.10 Từ nhãn đến hình ảnh trên video

Nhãn skill không được vẽ trực tiếp từ hình ảnh. Quy trình gồm bốn bước:

1. **Phân đoạn:** tạo mask của tay và vật ở mọi khung hình.
2. **Suy ra skill:** chạy các phép đo và luật ở mục 2.4–2.8, xuất danh sách đoạn skill với thời điểm bắt đầu/kết thúc, tên skill tiếng Anh và tiếng Việt, vật thể đang thao tác và độ cao lớn nhất.
3. **Suy ra sub-skill:** xuất các khoảng thời gian sub-skill tương ứng.
4. **Vẽ:** đọc lại video từng khung hình. Với khung hình ở thời điểm t, tìm đoạn skill có thời điểm bắt đầu ≤ t < kết thúc, rồi vẽ tên skill, sub-skill, trạng thái chạm, mask, điểm chạm và dải timeline phía dưới.

Vì vậy chữ "Skill: Rót" trên video là kết quả tra bảng theo thời gian, không phải mô hình đọc hình. Nếu thay ngưỡng ở bước 2, nhãn trên video thay đổi theo, còn bước 4 không cần sửa.

---

## 3. Hạn chế đã biết

- **Phụ thuộc chất lượng mask:** mask rung hoặc bị che một phần làm tín hiệu nhiễu. Hệ thống đã lọc và kẹp để giảm ảnh hưởng nhưng không loại bỏ hoàn toàn.
- **Vật tròn không đo được độ nghiêng:** cốc và ly chỉ dùng tín hiệu chuyển động.
- **Phụ thuộc độ sâu cho một số pha:** không có độ sâu thì nhấc và đặt chỉ dựa vào tốc độ ảnh, kém chính xác hơn, và sub-skill theo khoảng cách không xác định được.
- **Ngưỡng là kinh nghiệm:** các ngưỡng được chọn từ video đã kiểm chứng. Video điều kiện khác có thể cần hiệu chỉnh.
- **Chỉ mô tả hành vi:** hệ thống cho biết người đang làm gì và theo trình tự nào, không đánh giá làm đúng hay sai.

---

## 4. Tóm tắt

Nhận diện skill là chuỗi phép đo hình học trên mask và độ sâu:

1. Tách tay và vật ở mọi khung hình.
2. Chọn vật đang thao tác theo diện tích tiếp xúc, có giữ lựa chọn để tránh nhảy.
3. Đo chạm, khoảng cách tay–vật, tốc độ, độ cao thật, độ nghiêng và tương quan chuyển động tay–vật.
4. Làm sạch tín hiệu để loại nhiễu từ mask.
5. Gán nhãn từng khung hình theo luật có thứ tự ưu tiên.
6. Làm mịn nhãn theo thời gian và ghép thành đoạn skill.
7. Suy ra sub-skill bằng khoảng cách độ sâu giữa tay và vật.
8. Vẽ nhãn lên video bằng cách tra bảng skill theo thời điểm của từng khung hình.

Vì mọi bước là công thức xác định, cùng một video luôn cho cùng một kết quả và có thể kiểm chứng bằng mắt trên các khung hình cụ thể.
