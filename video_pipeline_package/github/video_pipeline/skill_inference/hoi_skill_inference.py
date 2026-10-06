"""
hoi_skill_inference.py

Suy luận SKILL MỨC THẤP (low-level) từ tương tác tay - vật thể, dùng MASK
của Grounded-SAM (nhánh A: sam_dino/gsam2_video.py) thay cho MediaPipe.

VÌ SAO THAY MEDIAPIPE:
  segment_and_overlay_vi_v3.py xác thực "tay có chạm vật không" bằng
  MediaPipe HandLandmarker (21 landmark) + bbox vật thể do VLM đoán. Cách đó
  có 2 điểm gãy đã kiểm chứng thực tế:
    1. MediaPipe mất dấu tay khi tay NẮM CHẶT vật — đầu ngón cong vào sau
       vật, bị che khuất. Chính README của repo ghi nhận có video 0/6 mốc
       phát hiện được tay, khi đó toàn bộ tầng xác thực hình học không chạy.
    2. Bbox vật thể do VLM trả về KHÔNG ổn định giữa các lần gọi (cùng 1 vật,
       mỗi mốc một bbox hơi lệch) — repo phải thêm rất nhiều bước làm mịn
       (smooth_object_labels, _bridge_chain_reversions...) để bù lại.
  Ngoài ra cả 2 tầng đều chạy theo MỐC LẤY MẪU THƯA (mặc định 1.5 s/mốc),
  nên chỉ ra được skill mức ngữ nghĩa (Grasp, Pour) chứ không thấy được các
  pha ngắn như lúc tay vừa chạm hay vừa nhấc lên.

MASK SAM2 KHẮC PHỤC CẢ 3:
  - Mask tách được tay và vật NGAY CẢ KHI ĐANG NẮM CHẶT (đã kiểm chứng bằng
    mắt trên chính video test của repo) — không phụ thuộc landmark bị che.
  - Mask theo vật xuyên suốt video với ID ổn định, không "trôi" như bbox VLM.
  - Có mask ở MỌI frame (30 fps), nên đo được cả pha ngắn vài chục ms.

TỪ HÌNH HỌC SUY RA SKILL: mọi kết luận đều là phép đo trên pixel, không có
model nào "đoán bằng chữ", nên không hallucination và chạy lại cho kết quả
y hệt (deterministic).

    Reach     tay chưa chạm, tay ĐANG TIẾN LẠI gần vật
    Contact   vừa chạm (mask tay phình ra chạm mask vật), vật CHƯA di chuyển
    Grasp     đang giữ: chạm liên tục, vật đứng yên hoặc đi cùng tay
    Lift      đang giữ + tâm vật đi LÊN
    Place     đang giữ + tâm vật đi XUỐNG (hạ xuống mặt bàn)
    MoveToTarget  đang giữ + tâm vật đi NGANG
    Pour      đang giữ + trục chính của vật XOAY (nghiêng đi)
    Release   vừa BUÔNG: mất chạm trong khi vật đứng yên
    Retract   tay rời ra, ĐANG ĐI XA vật
    Idle      không có tương tác nào đáng kể

BA CÁI BẪY ĐÃ GẶP THẬT VÀ CÁCH XỬ LÝ (xem chi tiết ở từng hằng số):
  1. Khoảng cách mask tay-vật bị CHI PHỐI BỞI CẲNG TAY. SAM2 cắt "hand" thành
     một vùng liền gồm cả cẳng tay, nên cẳng tay ở gần vật làm khoảng cách
     min luôn nhỏ dù bàn tay còn xa -> pha Reach gần như biến mất. Khắc phục:
     đo tiến/lùi bằng khoảng cách TÂM tay - TÂM vật (CDIST_*).
  2. Mask RUNG vài frame ở lúc buông vật, làm tín hiệu "đang chạm" nhấp nháy.
     Khắc phục: lấp khe hở ngắn và bỏ đoạn chạm quá ngắn (deglitch_contact).
  3. Hình dạng mask ĐỔI BẬC (vật bị che một phần) làm góc trục chính nhảy vài
     độ trong 1 frame — chia cho 1/30 s thành >200°/s, đủ để sinh Pour giả.
     Khắc phục: lọc median trên chuỗi góc + chặn tốc độ xoay vô lý.

Cách dùng:
    python3 hoi_skill_inference.py masks_gsam2.json --out skills_lowlevel.json
    python3 hoi_skill_inference.py masks_gsam2.json --dump-signals signals.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from common import decode_rle          # dùng khi chạy trong sam_dino/
except ImportError:                        # pragma: no cover
    decode_rle = None

# --------------------------------------------------------------- tham số
CONTACT_DILATE_PX = 3        # nới mask tay ra bao nhiêu px trước khi xét chạm
CONTACT_MIN_OVERLAP_PX = 12  # số pixel chồng lấn tối thiểu để coi là đã chạm
CONTACT_FILL_GAP_FRAMES = 6  # khe hở mất chạm ngắn hơn -> lấp lại (bẫy 2)
CONTACT_MIN_RUN_FRAMES = 4   # đoạn chạm ngắn hơn -> bỏ (bẫy 2)

SMOOTH_WINDOW = 7            # cửa sổ làm mịn tâm vật (frame), ~0.23 s ở 30 fps
VEL_WINDOW = 9               # cửa sổ tính vận tốc tâm vật
ANG_MEDIAN_WINDOW = 7        # lọc median trên góc trục chính (bẫy 3)
ANG_MAX_DEG_PER_S = 150.0    # tốc độ xoay tối đa hợp lý; vượt -> nhiễu mask (bẫy 3)

# Ngưỡng BẬT cho từng skill. Vật đứng yên vẫn rung vài px/s nên phải có
# ngưỡng, nếu không mọi frame đang giữ đều bị gán Lift/Place/MoveToTarget.
LIFT_ON = 14.0      # px/s tâm vật đi lên
PLACE_ON = 14.0     # px/s tâm vật đi xuống
MOVE_ON = 20.0      # px/s tâm vật đi ngang
TILT_ON = 22.0      # deg/s trục chính xoay
APPROACH_ON = 30.0  # px/s khoảng cách tâm tay-vật GIẢM
RETRACT_ON = 30.0   # px/s khoảng cách tâm tay-vật TĂNG
RELEASE_OBJ_SPEED = 8.0   # vật đứng yên (px/s) tại lúc buông -> Release
# Ngưỡng cho tín hiệu TƯƠNG QUAN CHUYỂN ĐỘNG tay-vật (xem build_signals):
# chỉ tính cos khi CẢ HAI đang chuyển động đủ nhanh — nếu tay/vật gần như
# đứng yên thì hướng vận tốc chỉ là nhiễu, cos vô nghĩa.
HOLD_MIN_SPEED = 8.0      # px/s tối thiểu của mỗi bên để tính cos
HOLD_COS = 0.70           # cos >= ngưỡng này -> vật đang đi CÙNG tay (đang cầm)
PUSH_SPEED = 25.0         # px/s: vật tự di chuyển trong lúc đang chạm tay

# Ngưỡng cho tín hiệu ĐỘ CAO THẬT (mm) từ luồng depth — xem --depth.
# Dùng đơn vị milimét thật thay vì pixel: "nhấc lên 3 cm" có nghĩa với robot,
# còn "tâm vật đi lên 40 px" thì không, vì px phụ thuộc khoảng cách tới camera.
# Đo bằng mm cũng hết mơ hồ: vật nằm trên bàn hay đã nhấc khỏi bàn là câu hỏi
# nhị phân rõ ràng, không phải đoán qua vận tốc ảnh.
LIFT_MM_PER_S = 25.0      # mm/s đi lên -> Lift
PLACE_MM_PER_S = 25.0     # mm/s đi xuống -> Place
ON_TABLE_MM = 15.0        # cao độ dưới ngưỡng này coi như đang đặt trên bàn
DEPTH_MIN_VALID_PX = 30   # số pixel depth hợp lệ tối thiểu trong mask vật
CONTACT_PHASE_SEC = 0.25  # độ dài pha "Contact" ngay sau khi vừa chạm
# ĐỘ DÀI TỐI THIỂU để tin trục chính của vật (trục lớn / trục nhỏ của PCA).
# Vật tròn như cốc/ly/bát có hai trục gần bằng nhau nên trục chính nhảy loạn
# giữa các frame — tin nó sẽ sinh "xoay"/"rót" giả. Đo thật: cốc/ly 1.27–1.54,
# chai 2.71–3.14, nên 2.0 tách sạch hai nhóm.
ELONG_MIN = 2.0
UPRIGHT_START_DEG = 15.0     # "gần thẳng": lệch dưới ngưỡng này
POUR_FROM_UPRIGHT_SEC = 1.5  # phải gần thẳng trong khoảng này mới được gọi là rót
# KHÔNG vừa NHẤC vừa RÓT. Người ta nhấc vật lên trước, rồi mới nghiêng để rót —
# nên nếu vật đang bay LÊN nhanh mà trục chính cũng xoay thì đó là NHẤC một vật
# dài (trục chính xoay theo), không phải rót. Đo được thật:
#   rót thật (demo)          : vheight <= +5 mm/s   (đã ở trên cao, đang hạ/rót)
#   nhấc kéo lên (DexYCB)    : vheight +62..+290 mm/s (đang bay lên)
# Thiếu điều kiện này thì mọi lần nhặt một vật dài (kéo, thìa, dao) lên đều bị
# gán "Rót" — đã kiểm chứng bằng mắt trên sequence DexYCB.
POUR_MAX_RISE_MM_PER_S = 30.0
# Một khi ĐÃ rót thì GIỮ nhãn Pour cho tới khi vật dựng thẳng lại, không để nó
# rơi về Grasp ở giữa. Lý do: lúc giữ nguyên góc rót, tốc độ xoay tụt xuống dưới
# ngưỡng nên bộ phân loại rơi về nhánh chung và gán "Cầm nắm" — đã thấy thật
# trên video demo (đoạn 5.77-6.27 s nằm giữa vùng rót 3.86-7.45 s). Đây là pha
# giữ góc rót, không phải cầm nắm.
POUR_STAY_DEG = 20.0

MEDIAN_LABEL_WINDOW = 7   # lọc median trên chuỗi NHÃN để khử nhấp nháy
MIN_SEGMENT_SEC = 0.30    # đoạn ngắn hơn bị gộp vào đoạn liền kề
# Reach/Retract/Contact/Release là pha CHUYỂN TIẾP, bản chất chỉ kéo dài vài
# frame — nếu áp cùng ngưỡng tối thiểu với Grasp/Lift/Pour thì chúng bị gộp
# mất sạch. Đã gặp thật cả hai chiều: quên "Reach" thì pha Reach 0.20 s bị nuốt
# vào Idle; quên "Retract" thì pha Retract 8 frame bị nuốt vào Idle ở cuối
# video (Retract ngắn hơn ngưỡng chung 9 frame đúng 1 frame).
TRANSIENT_SKILLS = ("Reach", "Retract", "Contact", "Release")
TRANSIENT_MIN_SEGMENT_SEC = 0.12

VI_NAMES = {
    "Reach": "Tiếp cận", "Contact": "Chạm", "Grasp": "Cầm nắm",
    "Lift": "Nhấc lên", "Place": "Đặt xuống", "MoveToTarget": "Di chuyển",
    "Pour": "Rót", "Release": "Thả ra", "Retract": "Rút tay",
    "Push": "Đẩy", "Pull": "Kéo",
    "Idle": "Không tương tác",
}


# ------------------------------------------------------------- đọc mask
def decode_mask(rle):
    """Giải RLE. Dùng decode_rle của common.py nếu import được, không thì tự
    giải — để module này chạy được cả khi đứng ngoài sam_dino/."""
    if decode_rle is not None:
        return decode_rle(rle)
    flat = np.zeros(int(np.prod(rle["size"])), dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(rle["size"])


def load_mask_series(path):
    """Đọc masks_gsam2.json -> (data, per_frame) với per_frame[i] = dict
    {obj_id: mask bool} tại frame i. Frame không có object nào = dict rỗng."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = int(round(data["frames"][-1]["frame"])) + 1 if data["frames"] else 0
    per_frame = [dict() for _ in range(n)]
    for fr in data["frames"]:
        i = int(fr["frame"])
        for o in fr["objects"]:
            per_frame[i][o["id"]] = decode_mask(o["rle"])
    return data, per_frame


def pick_roles(data):
    """Chọn id nào là TAY, id nào là VẬT THỂ. Grounding DINO nhận prompt dạng
    danh từ nên nhãn là chuỗi người dùng đưa vào; ở đây nhận diện theo từ khoá
    trong nhãn để không phụ thuộc đúng tên prompt.

    Một bàn tay có thể bị tách thành NHIỀU id khi bị che khuất giữa 2 keyframe
    (đã gặp thật: `hand` ra 3 id #2/#3/#4) — nên trả về LIST id, và mọi chỗ
    dùng đều hợp nhất chúng lại thành 1 mask."""
    hands, objs = [], []
    for o in data["objects"]:
        lab = (o["label"] or "").lower()
        (hands if "hand" in lab else objs).append(o["id"])
    return hands, objs


# --------------------------------------------------------- hình học/frame
def _overlap_px(hand_mask, obj_mask, k):
    hand_dil = cv2.dilate(hand_mask.astype(np.uint8), k) > 0
    return int(np.count_nonzero(hand_dil & obj_mask))


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _principal_angle_deg(mask):
    """(góc trục chính, độ dài) của mask. Góc tính theo độ trong [0,180).

    Vật dài như chai: trục chính gần như trùng trục thân chai, nên nghiêng chai
    để rót làm góc này đổi rõ rệt. Dùng PCA thay cv2.minAreaRect vì góc của
    minAreaRect phụ thuộc thứ tự đỉnh và hay nhảy ±90° giữa 2 frame liền nhau.

    ĐỘ DÀI = căn bậc hai của tỉ lệ 2 trị riêng (trục lớn / trục nhỏ). Với vật
    TRÒN (cốc, ly, bát) hai trị riêng gần bằng nhau nên tỉ lệ ~1 và TRỤC CHÍNH
    VÔ NGHĨA — nó nhảy loạn giữa các frame. Đo được thật: cốc/ly trong
    video_test3 cho tỉ lệ 1.27–1.54, chai cho 2.71–3.14. Vì vậy độ dài phải
    được trả về cùng góc để bên gọi biết khi nào ĐỪNG tin góc (xem ELONG_MIN)."""
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return None, None
    x = xs - xs.mean()
    y = ys - ys.mean()
    cov = np.cov(np.stack([x, y]))
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    lo, hi = float(min(evals)), float(max(evals))
    elong = float(np.sqrt(hi / lo)) if lo > 1e-9 else np.inf
    return ang, elong


def load_depth(path):
    """Đọc file .npz do depth_from_mcap.py xuất. Trả (stack uint16 mm, fps) —
    stack[i] ứng với ĐÚNG frame i của video màu (đã ghép theo timestamp)."""
    z = np.load(path)
    return z["depth"], float(z["fps"])


def load_intrinsics(path):
    """Đọc nội tham số camera (fx, fy, cx, cy) từ .npz do depth_from_mcap.py
    xuất, hoặc None nếu file cũ chưa có (chạy trước khi tính năng này được
    thêm) hoặc bag không có topic camera_info. Không có intrinsics thì KHÔNG
    quy đổi được toạ độ pixel sang milimét thật (X, Y) — chỉ có Z (độ sâu)."""
    z = np.load(path)
    if "fx" not in z:
        return None
    return {"fx": float(z["fx"]), "fy": float(z["fy"]),
           "cx": float(z["cx"]), "cy": float(z["cy"])}


def pixel_to_xyz_mm(u, v, depth_mm, intr):
    """(pixel u,v + độ sâu mm) -> toạ độ 3D THẬT (mm) trong HỆ CAMERA, bằng
    mô hình pinhole chuẩn. Z dương là hướng camera nhìn tới (RealSense).

    LƯU Ý HỆ QUY CHIẾU: đây là toạ độ trong hệ CAMERA, không phải hệ ROBOT hay
    hệ VẬT — camera quay demo và camera robot là MỘT (rig UR3 + RealSense) nên
    số này CÓ dùng được cho robot NẾU biết phép biến đổi camera -> gốc robot
    (đọc từ `/tf_static` trong bag, script này chưa làm). Với video quay bằng
    camera khác (video người, điện thoại...) thì toạ độ này VÔ NGHĨA cho robot
    — xem nguyên tắc "học SKILL không học CHUYỂN ĐỘNG" ở skill_params.py."""
    if depth_mm is None or np.isnan(depth_mm) or depth_mm <= 0 or intr is None:
        return None
    x = (u - intr["cx"]) * depth_mm / intr["fx"]
    y = (v - intr["cy"]) * depth_mm / intr["fy"]
    return float(x), float(y), float(depth_mm)


def project_xyz_to_pixel(xyz_mm, intr):
    """NGHỊCH ĐẢO của `pixel_to_xyz_mm`: toạ độ 3D (mm, hệ camera) -> pixel
    (u, v) bằng mô hình pinhole chuẩn. Dùng để VẼ LẠI lên ảnh 1 điểm/đoạn
    thẳng đã tính trong không gian 3D (vd 2 đầu mút trục chính từ
    `principal_axis_3d`). Trả None nếu Z<=0 (điểm ở sau camera, vô nghĩa)."""
    x, y, z = xyz_mm
    if z <= 0:
        return None
    u = x * intr["fx"] / z + intr["cx"]
    v = y * intr["fy"] / z + intr["cy"]
    return float(u), float(v)


MIN_POINTS_3D = 30    # đám mây ít hơn -> PCA 3D không đáng tin (giống MIN_OBJECT_PIXELS)


def backproject_mask_xyz(mask, depth_frame, intr):
    """mask + depth_frame (mm) + intrinsics -> mảng Nx3 (X,Y,Z mm, hệ CAMERA)
    của pixel trong mask có depth HỢP LỆ. Tự co giãn mask về đúng cỡ
    depth_frame nếu khác (gsam2_video.py --max-side có thể thu nhỏ mask).
    Trả None nếu không đủ điểm hợp lệ.

    LỌC NGOẠI LAI BẰNG HÀNG RÀO TUKEY (IQR) — BẮT BUỘC, không phải tuỳ chọn:
    lọc `z > 0` không đủ. Đo được thật trên video demo: mask chai có vài trăm
    pixel rìa (mép mask lẹm ra nền/bàn, hoặc giá trị sentinel hỏng của cảm
    biến) đọc ra Z từ 700 tới hơn 2000 mm trong khi chai thật chỉ cách camera
    400-550 mm — làm PCA 3D CHẠY TRÊN RÁC: trục chính bị kéo lệch hẳn về phía
    các điểm ngoại lai, tilt đo ra không đổi ~75-85° suốt cả đoạn rót thay vì
    tăng dần theo chuyển động thật.

    Hàng rào Tukey: `Q1, Q3` = tứ phân vị 25%/75% của Z, `IQR = Q3 - Q1`, giữ
    điểm trong `[Q1 - k·IQR, Q3 + k·IQR]`. Cùng họ thống kê bền vững với
    median+MAD (bản trước dùng) — đã ĐO SO SÁNH TRỰC TIẾP trên video demo: 2/3
    mốc kiểm tra cho kết quả giống hệt MAD, mốc còn lại lệch tâm ~2.6mm giữa
    các biến thể. Chọn `k=3.0` (không phải 1.5 sách vở) vì k=1.5 cắt tới vào
    đúng 2 ĐẦU MÚT thật của vật khi elongation cao — đo được: Z sau lọc bị thu
    hẹp còn 450-560mm (k=1.5) so với 450-613mm (k=3.0), tức cắt mất phần đuôi
    thật của chai, sẽ làm đoạn vẽ trục (dùng min/max chiếu lên trục) NGẮN HƠN
    thật."""
    if mask is None or depth_frame is None or intr is None:
        return None
    dh, dw = depth_frame.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) != (dh, dw):
        mask = cv2.resize(mask.astype(np.uint8), (dw, dh),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    z = depth_frame[ys, xs].astype(np.float64)
    valid = z > 0
    if int(valid.sum()) < MIN_POINTS_3D:
        return None
    xs, ys, z = xs[valid].astype(np.float64), ys[valid].astype(np.float64), z[valid]

    TUKEY_K = 3.0     # 1.5 kinh điển cắt cả đầu mút thật — xem docstring
    q1, q3 = np.percentile(z, [25.0, 75.0])
    iqr = q3 - q1
    band = max(15.0, TUKEY_K * iqr)   # >=15mm: vật mỏng (IQR~0) vẫn giữ đủ điểm
    lo, hi = q1 - band, q3 + band
    keep = (z >= lo) & (z <= hi)
    if int(keep.sum()) < MIN_POINTS_3D:
        return None
    xs, ys, z = xs[keep], ys[keep], z[keep]

    x = (xs - intr["cx"]) * z / intr["fx"]
    y = (ys - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=1)


def principal_axis_3d(points_xyz):
    """PCA 3D trên đám mây điểm -> dict {center_mm, axis, elongation,
    endpoint_a_mm, endpoint_b_mm}, hoặc None nếu quá ít điểm.

    THAY THẾ `_principal_angle_deg` (PCA 2D trên mask) khi có depth: PCA 2D đo
    góc CHIẾU LÊN ẢNH, bị RÚT NGẮN PHỐI CẢNH khi vật nghiêng ra xa/lại gần
    camera — đây chính là nguồn khiến điểm gắp trôi 0.038 đã ghi trong
    tongket.md (P3): "trục chính ước lượng bằng PCA trên mask, không phải tư
    thế thật của vật". PCA 3D đo trực tiếp trên đám mây điểm THẬT trong không
    gian (đã dựng bằng `backproject_mask_xyz`) nên không mắc lỗi phối cảnh này.

    `axis` là 1 ĐƯỜNG THẲNG (không phân biệt 2 chiều `axis` với `-axis` — khác
    PCA 2D trong `contact_point.py`, nơi còn neo được chiều theo y ảnh; trong
    3D không có "trên/dưới" tự nhiên để neo). Dùng `axis_tilt_deg` (arccos trị
    tuyệt đối) để so góc, khỏi phải xử lý dấu trục.

    `endpoint_a_mm`/`endpoint_b_mm`: 2 ĐẦU MÚT thật của vật dọc theo trục (tâm
    ± độ dài chiếu lên trục) — dùng để VẼ 1 đoạn thẳng đúng chiều dài vật lên
    ảnh (xem `render_hoi_video.py::draw_pose_axis`), không phải để đo.

    `elongation` dùng CHUNG ngưỡng `ELONG_MIN` với PCA 2D: vật tròn (ly, cốc)
    vẫn nhảy loạn trong 3D vì cùng lý do (2 trị riêng lớn nhất gần bằng nhau)."""
    if points_xyz is None or len(points_xyz) < MIN_POINTS_3D:
        return None
    c = points_xyz.mean(axis=0)
    q = points_xyz - c
    cov = q.T @ q / len(q)
    w, v = np.linalg.eigh(cov)
    axis = v[:, -1]
    var_lo, var_hi = float(w[0]), float(w[-1])
    elong = float(np.sqrt(var_hi / var_lo)) if var_lo > 1e-9 else np.inf
    proj = q @ axis
    return {"center_mm": c, "axis": axis, "elongation": elong,
           "endpoint_a_mm": c + float(proj.min()) * axis,
           "endpoint_b_mm": c + float(proj.max()) * axis}


def axis_tilt_deg(axis, reference=(0.0, -1.0, 0.0)):
    """Góc nghiêng (0-90°) giữa TRỤC 3D của vật và 1 hướng tham chiếu — mặc
    định trục Y của CAMERA hướng LÊN trong ảnh (dấu trừ vì trục Y ảnh/camera
    quy ước hướng XUỐNG). Coi đây là xấp xỉ PHƯƠNG THẲNG ĐỨNG THẬT — chỉ đúng
    khi camera không nghiêng nhiều so với mặt bàn (đúng với rig gắn ngang kiểu
    UR3+RealSense). 0° = vật đứng thẳng song song phương thẳng đứng, 90° = vật
    nằm ngang.

    Dùng arccos(TRỊ TUYỆT ĐỐI của tích vô hướng) vì `axis` không có chiều —
    không phân biệt được "chúc xuống 30°" với "hướng lên 150°", mà với mục
    đích đo ĐỘ NGHIÊNG (không phải đo HƯỚNG) thì hai trường hợp đó là MỘT."""
    ref = np.asarray(reference, dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    ref = ref / np.linalg.norm(ref)
    cos_a = float(np.clip(abs(np.dot(a, ref)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _median_depth_in_mask(depth_frame, mask):
    """Trung vị độ sâu (mm) của các pixel HỢP LỆ nằm trong mask. Dùng trung vị
    chứ không phải trung bình: rìa mask vật thường lẫn pixel nền hoặc pixel
    nhiễu (0 = không đo được), trung bình bị chúng kéo lệch mạnh."""
    if mask is None or not mask.any():
        return np.nan
    vals = depth_frame[mask]
    vals = vals[vals > 0]
    if vals.size < DEPTH_MIN_VALID_PX:
        return np.nan
    return float(np.median(vals))


def frame_features(per_frame, hand_ids, obj_ids, depth=None):
    """Trả {obj_id: [feature hoặc None theo từng frame]}. Mỗi feature gồm tâm,
    diện tích, góc trục chính, độ chồng lấn với tay, và khoảng cách TÂM tay -
    TÂM vật (không dùng khoảng cách mask, xem bẫy 1 ở docstring).

    depth (tuỳ chọn): stack uint16 mm. Khi có, mỗi feature thêm `depth_mm` =
    độ sâu trung vị của vật, để build_signals quy ra ĐỘ CAO THẬT."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * CONTACT_DILATE_PX + 1,) * 2)
    per_obj = {oid: [] for oid in obj_ids}

    for fi, masks in enumerate(per_frame):
        hand_mask = None
        for hid in hand_ids:
            if hid in masks:
                hand_mask = masks[hid] if hand_mask is None else (hand_mask | masks[hid])
        hand_c = _centroid(hand_mask) if hand_mask is not None else None
        for oid in obj_ids:
            m = masks.get(oid)
            if m is None or not m.any():
                per_obj[oid].append(None)
                continue
            cx, cy = _centroid(m)
            ang_i, elong_i = _principal_angle_deg(m)
            # Vật TRÒN thì trục chính vô nghĩa -> bỏ góc, đừng để nó sinh
            # "xoay"/"rót" giả. Xem ELONG_MIN.
            if elong_i is None or elong_i < ELONG_MIN:
                ang_i = None
            cdist = (float(np.hypot(hand_c[0] - cx, hand_c[1] - cy))
                     if hand_c is not None else None)
            depth_mm = np.nan
            if depth is not None and fi < len(depth):
                df = depth[fi]
                # Mask và depth có thể khác cỡ (mask sinh trên frame đã thu nhỏ
                # bởi --max-side của gsam2_video.py) -> kéo mask về cỡ depth.
                m_d = m
                if m.shape[:2] != df.shape[:2]:
                    m_d = cv2.resize(m.astype(np.uint8), (df.shape[1], df.shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                depth_mm = _median_depth_in_mask(df, m_d)
            per_obj[oid].append({
                "cx": cx, "cy": cy, "area": int(np.count_nonzero(m)),
                "ang": ang_i, "elong": elong_i,
                "cdist": cdist,
                "depth_mm": depth_mm,
                "overlap": _overlap_px(hand_mask, m, k) if hand_mask is not None else 0,
                # Tâm bàn tay lưu kèm để tính TƯƠNG QUAN CHUYỂN ĐỘNG tay-vật
                # ở build_signals (xem chú thích HOLD_* bên dưới).
                "hcx": hand_c[0] if hand_c is not None else None,
                "hcy": hand_c[1] if hand_c is not None else None,
            })
    return per_obj


def pick_active_object(per_obj, switch_margin=1.3):
    """Chọn vật thể "đang được thao tác" cho từng frame.

    VẬT ĐANG CHẠM: chọn vật có DIỆN TÍCH TIẾP XÚC lớn nhất, và GIỮ NGUYÊN vật
    đang thao tác trừ khi vật khác vượt trội rõ rệt (switch_margin).

    Vì sao phải có cả 2 vế — đã kiểm chứng trên sequence DexYCB 154850 (chai
    và máy khoan nằm sát nhau trên bàn, tay ở giữa):
      - Chọn theo khoảng cách TÂM tay-vật (cách cũ) thì vật "đang thao tác"
        NHẢY QUA LẠI giữa 2 vật từ frame này sang frame kia, làm tín hiệu của
        hai vật trộn vào nhau và sinh ra chuỗi vô nghĩa kiểu "Nhấc lên" xuất
        hiện TRƯỚC "Cầm nắm".
      - Diện tích tiếp xúc là thước đo đúng hơn "đang cầm vật nào": tay cầm
        vật nào thì chồng lấn với vật đó nhiều hơn hẳn.
      - Nhưng chỉ so sánh diện tích từng frame vẫn còn rung ở các frame biên
        (2 vật có diện tích gần bằng nhau), nên cần thêm vế giữ nguyên.

    KHÔNG CHẠM: không vật nào chạm tay -> đang là pha Reach/Retract, chọn vật
    GẦN TAY NHẤT (khoảng cách tâm) để biết tay đang tiến tới vật nào."""
    n = len(next(iter(per_obj.values()))) if per_obj else 0
    active = []
    prev = None
    for i in range(n):
        cands = []
        for oid, series in per_obj.items():
            f = series[i]
            if f is None:
                continue
            cands.append((oid, f["overlap"] >= CONTACT_MIN_OVERLAP_PX, f["overlap"],
                          f["cdist"] if f["cdist"] is not None else 1e9))
        if not cands:
            active.append(None)
            prev = None
            continue

        touching = [c for c in cands if c[1]]
        if touching:
            best = max(touching, key=lambda c: c[2])          # tiếp xúc nhiều nhất
            if prev is not None:
                old = next((c for c in touching if c[0] == prev), None)
                if old is not None and old[2] * switch_margin >= best[2]:
                    best = old                                  # giữ vật cũ
            pick = best[0]
        else:
            pick = min(cands, key=lambda c: c[3])[0]            # gần tay nhất
        prev = pick
        active.append(pick)
    return active


# ------------------------------------------------------------- tín hiệu
def _nanmedian_filter(a, w):
    """Lọc median bỏ qua NaN — khử xung nhọn 1-2 frame mà không làm mờ bậc
    thang (khác trung bình trượt, vốn kéo đuôi ở mọi bước nhảy)."""
    a = np.asarray(a, dtype=float)
    if w <= 1 or a.size == 0:
        return a.copy()
    out = np.full_like(a, np.nan)
    half = w // 2
    for i in range(a.size):
        seg = a[max(0, i - half):min(a.size, i + half + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _smooth(a, w):
    """Trung bình trượt giữ nguyên NaN (frame thiếu dữ liệu)."""
    a = np.asarray(a, dtype=float)
    if w <= 1 or a.size == 0:
        return a.copy()
    out = np.full_like(a, np.nan)
    half = w // 2
    for i in range(a.size):
        seg = a[max(0, i - half):min(a.size, i + half + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _grad(a, fps, w):
    """Đạo hàm bậc 1 (đơn vị /giây) trên tín hiệu ĐÃ làm mịn, dùng hiệu trung
    tâm trong cửa sổ w frame. Đạo hàm thô theo từng frame quá nhiễu vì mask
    rung vài px giữa 2 frame liền nhau."""
    a = np.asarray(a, dtype=float)
    g = np.full_like(a, np.nan)
    half = max(1, w // 2)
    for i in range(a.size):
        lo, hi = max(0, i - half), min(a.size - 1, i + half)
        if hi <= lo:
            continue
        if np.isnan(a[lo]) or np.isnan(a[hi]):
            continue
        g[i] = (a[hi] - a[lo]) / ((hi - lo) / fps)
    return g


def deglitch_contact(contact):
    """Làm sạch tín hiệu "đang chạm" (bẫy 2): lấp khe hở ngắn rồi bỏ đoạn
    chạm quá ngắn. Mask tay-vật rung ở đúng lúc buông vật khiến chuỗi này nhấp
    nháy, và mỗi lần nhấp nháy lại sinh ra một cặp Release/Contact giả."""
    c = np.asarray(contact, dtype=bool).copy()
    n = c.size
    if n == 0:
        return c

    i = 0                                     # 1) lấp khe hở ngắn
    while i < n:
        if c[i]:
            i += 1
            continue
        j = i
        while j < n and not c[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= CONTACT_FILL_GAP_FRAMES:
            c[i:j] = True
        i = j

    i = 0                                     # 2) bỏ đoạn chạm quá ngắn
    while i < n:
        if not c[i]:
            i += 1
            continue
        j = i
        while j < n and c[j]:
            j += 1
        if (j - i) < CONTACT_MIN_RUN_FRAMES:
            c[i:j] = False
        i = j
    return c


def _object_references(per_obj):
    """Tham chiếu RIÊNG cho TỪNG VẬT: tư thế nghỉ (góc) và mốc mặt bàn (độ sâu).

    Phải tính riêng chứ không tính trên chuỗi "vật đang thao tác", vì chuỗi đó
    TRỘN nhiều vật: mốc của vật này bị kéo lệch bởi vật khác. Đo được thật trên
    video_test3 — góc tư thế nghỉ của cốc và của chai lệch nhau ~50°, lấy trung
    vị chung thì CẢ HAI đều sai, và chai bị gán "nghiêng 40–60°" suốt video dù
    nó đang dựng đứng."""
    refs = {}
    for oid, series in per_obj.items():
        ang = np.array([f["ang"] if (f and f["ang"] is not None) else np.nan
                        for f in series], dtype=float)
        ov = np.array([f["overlap"] if f else 0.0 for f in series], dtype=float)
        ang_s = _smooth(ang, SMOOTH_WINDOW)
        rest = ang_s[ov < CONTACT_MIN_OVERLAP_PX]
        rest = rest[~np.isnan(rest)]
        upright = float(np.median(rest)) if rest.size else np.nan

        dep = np.array([f["depth_mm"] if (f and not np.isnan(f.get("depth_mm", np.nan)))
                        else np.nan for f in series], dtype=float)
        ds = _smooth(dep, SMOOTH_WINDOW)
        dv = ds[~np.isnan(ds)]
        dref = float(np.percentile(dv, 90)) if dv.size >= 10 else np.nan
        refs[oid] = (upright, dref)
    return refs


def build_signals(per_obj, active, fps):
    """Gộp feature theo vật đang thao tác thành các chuỗi tín hiệu đã làm mịn
    + đạo hàm, sẵn sàng cho bước phân loại."""
    n = len(active)
    refs = _object_references(per_obj)
    cx = np.full(n, np.nan); cy = np.full(n, np.nan)
    ang = np.full(n, np.nan); cdist = np.full(n, np.nan)
    hcx = np.full(n, np.nan); hcy = np.full(n, np.nan)
    dmm = np.full(n, np.nan)
    ov = np.zeros(n)
    for i, oid in enumerate(active):
        if oid is None:
            continue
        f = per_obj[oid][i]
        if f is None:
            continue
        cx[i], cy[i], ang[i] = f["cx"], f["cy"], f["ang"]
        cdist[i], ov[i] = f["cdist"] if f["cdist"] is not None else np.nan, f["overlap"]
        dmm[i] = f.get("depth_mm", np.nan)
        if f["hcx"] is not None:
            hcx[i], hcy[i] = f["hcx"], f["hcy"]

    # Góc là một ĐƯỜNG (không có hướng) trong [0,180). Nhân đôi rồi unwrap để
    # biến nó thành liên tục trước khi lọc — nếu không, mỗi lần trục chính
    # quay qua mốc 0/180 lại sinh một bước nhảy 180° giả.
    ang_u = np.degrees(np.unwrap(np.radians(ang) * 2.0) / 2.0)
    ang_u = _nanmedian_filter(ang_u, ANG_MEDIAN_WINDOW)   # bẫy 3
    ang_s = _smooth(ang_u, SMOOTH_WINDOW)
    vang = _grad(ang_s, fps, VEL_WINDOW)
    # Chặn tốc độ xoay vô lý: mask đổi hình bậc thang có thể tạo 200°/s trong
    # 1 frame, trong khi rót thật hiếm khi vượt ~100°/s. Không chặn thì mỗi lần
    # mask "giật" lại sinh một đoạn Pour giả.
    vang = np.clip(vang, -ANG_MAX_DEG_PER_S, ANG_MAX_DEG_PER_S)

    cx_s, cy_s = _smooth(cx, SMOOTH_WINDOW), _smooth(cy, SMOOTH_WINDOW)
    cdist_s = _smooth(cdist, SMOOTH_WINDOW)
    vcx, vcy = _grad(cx_s, fps, VEL_WINDOW), _grad(cy_s, fps, VEL_WINDOW)
    vcdist = _grad(cdist_s, fps, VEL_WINDOW)

    # TƯƠNG QUAN CHUYỂN ĐỘNG TAY - VẬT: cos góc giữa vector vận tốc bàn tay và
    # vector vận tốc vật. Đây là phép đo TRỰC TIẾP định nghĩa của "đang cầm":
    # vật được cầm thì đi CÙNG HƯỚNG và cùng độ lớn với tay (cos ~ 1); vật chỉ
    # bị chạm/đẩy thì đứng yên (không tính được cos) hoặc đi lệch hướng.
    #
    # VÌ SAO KHÔNG DÙNG TƯ THẾ NGÓN TAY: đã thử chạy MediaPipe HandLandmarker
    # (cả toàn ảnh lẫn cắt theo mask) để đo độ cuộn ngón — xem hand_pose_masked.py.
    # Kết quả đo thật: với bàn tay ĐEO GĂNG đang nắm chai, model "phát hiện" tay
    # 97% số frame nhưng dựng skeleton NGÓN DUỖI THẲNG lên đúng vị trí bàn tay
    # đang nắm (curl ~0.05 suốt cả video, không phân biệt được lúc cầm và lúc
    # buông); với bàn tay trần thì chỉ phát hiện 53% số frame và curl nhiễu
    # (0.30-0.51). Nên tư thế ngón KHÔNG dùng được ở đây — còn tương quan
    # chuyển động thì cho kết quả rất rõ (cos trung vị 0.97 trên video demo).
    vhx, vhy = _grad(_smooth(hcx, SMOOTH_WINDOW), fps, VEL_WINDOW), \
               _grad(_smooth(hcy, SMOOTH_WINDOW), fps, VEL_WINDOW)
    sp_h = np.hypot(vhx, vhy)
    sp_o = np.hypot(vcx, vcy)
    denom = np.maximum(sp_h * sp_o, 1e-9)
    hold = np.where((sp_h > HOLD_MIN_SPEED) & (sp_o > HOLD_MIN_SPEED),
                    (vhx * vcx + vhy * vcy) / denom, np.nan)

    # ĐỘ CAO THẬT (mm) từ depth. Mốc quy chiếu = độ sâu lúc vật nằm trên bàn,
    # lấy bằng phân vị 90 của các giá trị đo được (vật ở xa camera nhất = thấp
    # nhất). Tự hiệu chuẩn theo chính video, không cần biết trước mặt bàn cao
    # bao nhiêu. Camera nhìn xuống nên vật NHẤC LÊN thì depth GIẢM -> đổi dấu
    # để "cao hơn" là số dương.
    depth_s = _smooth(dmm, SMOOTH_WINDOW)
    # mốc mặt bàn LẤY THEO TỪNG VẬT (xem _object_references)
    d_ref_arr = np.array([refs.get(a, (np.nan, np.nan))[1] for a in active], dtype=float)
    if not np.all(np.isnan(d_ref_arr)):
        height = np.where(np.isnan(depth_s) | np.isnan(d_ref_arr), np.nan,
                          np.maximum(0.0, d_ref_arr - depth_s))
        vheight = _grad(height, fps, VEL_WINDOW)
        valid_d = d_ref_arr[~np.isnan(d_ref_arr)]
        d_ref = float(np.median(valid_d)) if valid_d.size else np.nan
    else:
        d_ref = np.nan
        height = np.full(n, np.nan)
        vheight = np.full(n, np.nan)

    # ĐỘ LỆCH KHỎI TƯ THẾ THẲNG (độ). Tư thế thẳng lấy từ chính các frame
    # KHÔNG chạm tay (lúc đó vật đang đứng yên trên bàn ở đầu video) — tự hiệu
    # chuẩn. Góc trục chính là một ĐƯỜNG không có hướng trong [0,180) nên hiệu
    # phải lấy modulo 180 (lệch 5° và lệch 175° gần như nhau).
    # tư thế nghỉ LẤY THEO TỪNG VẬT (xem _object_references)
    upright_arr = np.array([refs.get(a, (np.nan, np.nan))[0] for a in active], dtype=float)
    with np.errstate(invalid="ignore"):
        dev = np.abs(((ang_s - upright_arr + 90.0) % 180.0) - 90.0)
    dev = np.where(np.isnan(upright_arr) | np.isnan(ang_s), np.nan, dev)
    valid_u = upright_arr[~np.isnan(upright_arr)]
    upright_ref = float(np.median(valid_u)) if valid_u.size else np.nan

    # BỎ TÍN HIỆU ĐẠO HÀM Ở CÁC FRAME ĐỔI VẬT. Khi vật "đang thao tác" đổi sang
    # vật khác, MỌI chuỗi (tâm, độ cao, góc, khoảng cách) đều nhảy bậc vì hai
    # vật nằm ở hai chỗ khác nhau — đạo hàm qua điểm nối đó là rác hoàn toàn.
    # Đo được thật trên sequence DexYCB 144839: vheight nhảy tới +560 và
    # -597 mm/s ngay tại frame đổi vật, sinh ra "Nhấc lên" giả trong khi tay
    # chưa hề cầm gì. Sequence chỉ có MỘT vật được thao tác (155212) không bao
    # giờ đổi vật nên không dính lỗi này — vì vậy lỗi chỉ lộ ra ở cảnh nhiều vật.
    switch = np.zeros(n, bool)
    for i in range(1, n):
        if active[i] != active[i - 1]:
            lo, hi = max(0, i - VEL_WINDOW), min(n, i + VEL_WINDOW + 1)
            switch[lo:hi] = True
    for arr in (vcx, vcy, vcdist, vang, vheight):
        arr[switch] = np.nan

    return {
        "cx": cx_s, "cy": cy_s, "cdist": cdist_s, "overlap": ov.astype(float),
        "ang": ang_s, "dev": dev,
        "vcx": vcx, "vcy": vcy, "vcdist": vcdist, "vang": vang,
        "hold": hold, "obj_speed": sp_o,
        "depth_mm": depth_s, "height_mm": height, "vheight": vheight,
        "_depth_ref_mm": d_ref, "_upright_ref_deg": upright_ref,
        "_obj_switch": switch,
    }


# -------------------------------------------------------- phân loại skill
def classify_frames(sig, fps):
    """Gán nhãn skill mức thấp cho từng frame bằng luật hình học.

    Thứ tự xét quan trọng: các dấu hiệu ĐẶC TRƯNG (xoay -> rót) phải xét
    trước các dấu hiệu CHUNG (đang giữ), nếu không thì mọi frame đang giữ đều
    bị gán Grasp và không bao giờ thấy được Pour."""
    n = len(sig["cx"])
    lab = ["Idle"] * n
    contact = deglitch_contact(sig["overlap"] >= CONTACT_MIN_OVERLAP_PX)

    # frame bắt đầu mỗi chuỗi chạm liên tục — để tách pha "Contact" (vừa chạm,
    # chưa kịp di chuyển) khỏi "Grasp" (đã giữ ổn định)
    contact_run_start = np.full(n, -1)
    run_start = -1
    for i in range(n):
        if contact[i] and run_start < 0:
            run_start = i
        if not contact[i]:
            run_start = -1
        contact_run_start[i] = run_start
    contact_phase_frames = max(2, int(round(CONTACT_PHASE_SEC * fps)))

    vheight = sig.get("vheight")
    dev = sig.get("dev")

    # RÓT chỉ được bắt đầu từ tư thế GẦN THẲNG. Nếu không có điều kiện này, việc
    # NHẤC một vật đang nằm nghiêng lên (vd chai đang nằm ngang trên bàn, nhặt
    # lên dựng lại) cũng làm góc trục chính xoay đủ nhanh để sinh "Pour" giả —
    # đã kiểm chứng bằng mắt trên video demo: đoạn 6.27-7.38 s từng bị gán Pour
    # nhưng thực ra là NHẤC CHAI LÊN. Có điều kiện này thì đoạn đó rơi về đúng
    # nhánh Lift.
    recently_upright = np.zeros(n, bool)
    if dev is not None:
        win = max(1, int(POUR_FROM_UPRIGHT_SEC * fps))
        for i in range(n):
            lo = max(0, i - win)
            seg = dev[lo:i + 1]
            recently_upright[i] = bool(np.any(seg < UPRIGHT_START_DEG))

    pour_locked = False
    for i in range(n):
        vcx, vcy, vang = sig["vcx"][i], sig["vcy"][i], sig["vang"][i]
        vcdist, hold = sig["vcdist"][i], sig["hold"][i]
        obj_speed = sig["obj_speed"][i]

        # Nhấc lên / đặt xuống: ưu tiên ĐỘ CAO THẬT (mm) khi có depth, vì đó
        # là phép đo có đơn vị vật lý và không phụ thuộc khoảng cách tới camera.
        # Không có depth thì lùi về vận tốc tâm vật theo pixel.
        vh = vheight[i] if vheight is not None else np.nan
        if not np.isnan(vh):
            is_lift, is_place = vh >= LIFT_MM_PER_S, vh <= -PLACE_MM_PER_S
            # đang bay lên nhanh -> không thể là rót (xem POUR_MAX_RISE_MM_PER_S).
            # Không có depth thì vh = NaN nên bỏ qua được điều kiện này.
            not_rising = vh < POUR_MAX_RISE_MM_PER_S
        else:
            is_lift = (not np.isnan(vcy)) and vcy <= -LIFT_ON
            is_place = (not np.isnan(vcy)) and vcy >= PLACE_ON
            not_rising = True

        if contact[i]:
            # Tách "đang CẦM" khỏi "chỉ CHẠM/ĐẨY": vật đi cùng hướng với tay
            # (hold ~ 1) là đang được cầm; vật tự đi trong lúc tay vẫn chạm mà
            # KHÔNG cùng hướng là đang bị ĐẨY trượt trên mặt bàn.
            being_pushed = (not np.isnan(obj_speed) and obj_speed >= PUSH_SPEED
                            and (np.isnan(hold) or hold < HOLD_COS))
            # CHỐT RÓT: đã vào rót thì chỉ thoát khi vật DỰNG THẲNG LẠI.
            # Không có chốt này, lúc giữ nguyên góc rót (vang tụt dưới ngưỡng)
            # bộ phân loại rơi về nhánh chung và gán "Cầm nắm" vào giữa vùng rót.
            if pour_locked:
                if dev is None or np.isnan(dev[i]) or dev[i] < POUR_STAY_DEG:
                    pour_locked = False
            elif (not np.isnan(vang) and abs(vang) >= TILT_ON
                    and recently_upright[i] and not_rising):
                pour_locked = True

            if pour_locked:
                lab[i] = "Pour"
                continue
            if is_lift:
                lab[i] = "Lift"
            elif is_place:
                lab[i] = "Place"
            elif not np.isnan(vcx) and abs(vcx) >= MOVE_ON:
                lab[i] = "MoveToTarget"
            elif being_pushed:
                lab[i] = "Push"
            elif contact_run_start[i] >= 0 and i - contact_run_start[i] < contact_phase_frames:
                lab[i] = "Contact"
            else:
                lab[i] = "Grasp"
        else:
            pour_locked = False
            # Vừa buông: frame trước còn chạm, frame này hết, mà vật vẫn đứng
            # yên -> thả vật xuống chứ không phải kéo tay đi.
            if i > 0 and contact[i - 1] and not np.isnan(obj_speed) and obj_speed < RELEASE_OBJ_SPEED:
                lab[i] = "Release"
            elif not np.isnan(vcdist) and vcdist <= -APPROACH_ON:
                lab[i] = "Reach"
            elif not np.isnan(vcdist) and vcdist >= RETRACT_ON:
                lab[i] = "Retract"
            elif i > 0 and lab[i - 1] in ("Reach", "Contact", "Grasp", "Lift",
                                          "Pour", "MoveToTarget", "Place", "Release",
                                          "Push", "Pull"):
                # không đủ tín hiệu để khẳng định đang tiến hay lùi, nhưng vừa
                # rời khỏi một pha thao tác -> nối tiếp bằng Retract thay vì
                # nhảy về Idle rồi lại Reach, gây nhấp nháy trên timeline
                lab[i] = "Retract"
            else:
                lab[i] = "Idle"
    return lab


def prune_orphan_approach(labels, fps, look_sec=1.2):
    """Reach/Retract chỉ có nghĩa khi GẮN VỚI một lần tương tác: Reach phải
    dẫn tới Contact, còn Retract phải theo sau Contact.

    Đoạn Reach/Retract đứng lẻ thường nằm ở đầu/cuối video, khi tay chỉ đang
    đổi vị trí chứ chưa/không còn thao tác gì — đã gặp thật: frame 10-20 bị
    gán "Retract" trong khi tay đang HẠ XUỐNG để chuẩn bị cầm chai (khoảng
    cách tâm tay-vật tăng nhẹ vì tay đi chéo xuống), và frame cuối bị gán
    "Reach" trong khi tay đang rời đi. Cả hai đều vô nghĩa về mặt ngữ nghĩa
    nên đổi thành Idle."""
    n = len(labels)
    look = max(1, int(round(look_sec * fps)))
    interacted = [l in ("Contact", "Grasp", "Lift", "Pour", "Place",
                        "MoveToTarget", "Release", "Push", "Pull") for l in labels]
    out = list(labels)
    for i, l in enumerate(labels):
        if l == "Reach":
            if not any(interacted[j] for j in range(i, min(n, i + look))):
                out[i] = "Idle"
        elif l == "Retract":
            if not any(interacted[j] for j in range(max(0, i - look), i)):
                out[i] = "Idle"
    return out


def label_to_index(labels):
    uniq = sorted(set(labels))
    idx = {l: i for i, l in enumerate(uniq)}
    return np.array([idx[l] for l in labels]), uniq


def median_filter_labels(labels, w):
    """Lọc median trên chuỗi NHÃN (theo chỉ số số nguyên): khử nhãn nhảy qua
    lại 1-2 frame do mask rung, nhưng vẫn giữ được pha ngắn thật (>= w/2
    frame) — tốt hơn bỏ phiếu đa số vì không kéo lệch ranh giới đoạn."""
    if w <= 1 or not labels:
        return list(labels)
    arr, uniq = label_to_index(labels)
    half = w // 2
    out = arr.copy()
    for i in range(arr.size):
        seg = arr[max(0, i - half):min(arr.size, i + half + 1)]
        out[i] = np.median(seg)
    return [uniq[int(v)] for v in out]


def merge_short_segments(labels, min_frames, transient_min_frames=None):
    """Gộp các đoạn ngắn hơn ngưỡng vào đoạn liền kề — tránh timeline bị vụn vì
    những pha chỉ kéo dài vài frame.

    Ngưỡng riêng cho TRANSIENT_SKILLS (Reach/Contact/Release): đây là các pha
    chuyển tiếp vốn ngắn, áp ngưỡng chung sẽ xoá mất chúng."""
    if min_frames <= 1 or not labels:
        return list(labels)
    t_frames = transient_min_frames if transient_min_frames is not None else min_frames

    def min_for(lab):
        return t_frames if lab in TRANSIENT_SKILLS else min_frames

    out = list(labels)
    changed = True
    while changed:
        changed = False
        runs = []
        i = 0
        while i < len(out):
            j = i
            while j < len(out) and out[j] == out[i]:
                j += 1
            runs.append((i, j, out[i]))
            i = j
        for k, (s, e, lab) in enumerate(runs):
            if e - s >= min_for(lab):
                continue
            prev_lab = runs[k - 1][2] if k > 0 else None
            next_lab = runs[k + 1][2] if k + 1 < len(runs) else None
            if prev_lab is None and next_lab is None:
                continue
            # hoà: ưu tiên đoạn DÀI hơn, để pha thật không bị đoạn nhiễu nuốt
            if prev_lab is None:
                target = next_lab
            elif next_lab is None:
                target = prev_lab
            else:
                prev_len = runs[k - 1][1] - runs[k - 1][0]
                next_len = runs[k + 1][1] - runs[k + 1][0]
                target = prev_lab if prev_len >= next_len else next_lab
            for m in range(s, e):
                out[m] = target
            changed = True
            break
    return out


def to_segments(labels, fps, active):
    """Nhãn theo frame -> list đoạn liên tục [(t0, t1, skill, obj_id)]."""
    segs = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        segs.append((i / fps, j / fps, labels[i], active[i]))
        i = j
    return segs


def render_summary(segments):
    return "\n".join(
        f"  {t0:5.2f}s - {t1:5.2f}s  ({t1 - t0:4.2f}s)  {lab:<13} "
        f"{VI_NAMES.get(lab, lab)}"
        for t0, t1, lab, _oid in segments)


def infer(mask_json_path, depth_path=None, verbose=True):
    data, per_frame = load_mask_series(mask_json_path)
    fps = float(data["fps"])
    hand_ids, obj_ids = pick_roles(data)

    if not hand_ids:
        raise RuntimeError("Không thấy object nào có nhãn chứa 'hand' trong "
                           "masks json — kiểm tra lại --prompt của gsam2_video.py")
    if not obj_ids:
        raise RuntimeError("Không thấy vật thể nào (mọi object đều là tay)")

    depth = None
    if depth_path:
        depth, dfps = load_depth(depth_path)
        if abs(dfps - fps) > 1.0:
            print(f"  CẢNH BÁO: fps depth ({dfps:.2f}) lệch fps mask ({fps:.2f}) — "
                  f"nếu depth không được căn theo frame màu thì kết quả sẽ sai")

    per_obj = frame_features(per_frame, hand_ids, obj_ids, depth=depth)
    active = pick_active_object(per_obj)
    sig = build_signals(per_obj, active, fps)

    labels = classify_frames(sig, fps)
    labels = prune_orphan_approach(labels, fps)
    labels = median_filter_labels(labels, MEDIAN_LABEL_WINDOW)
    labels = merge_short_segments(labels, max(1, int(round(MIN_SEGMENT_SEC * fps))),
                                  max(1, int(round(TRANSIENT_MIN_SEGMENT_SEC * fps))))

    segments = to_segments(labels, fps, active)
    id2label = {o["id"]: o["label"] for o in data["objects"]}

    # Độ cao lớn nhất đạt được trong mỗi đoạn — dữ liệu có đơn vị thật, hữu ích
    # cho Task Planner (biết vật được nhấc cao bao nhiêu mm).
    def seg_height(t0, t1):
        i0, i1 = int(t0 * fps), min(int(t1 * fps), len(sig["height_mm"]))
        h = sig["height_mm"][i0:i1]
        h = h[~np.isnan(h)]
        return round(float(h.max()), 1) if h.size else None

    out = {
        "source_masks": mask_json_path,
        "source_depth": depth_path,
        "fps": fps,
        "size": data.get("size"),
        "n_frames": len(per_frame),
        "hand_ids": hand_ids,
        "object_ids": obj_ids,
        "depth_reference_mm": (None if np.isnan(sig.get("_depth_ref_mm", np.nan))
                               else round(float(sig["_depth_ref_mm"]), 1)),
        "id_to_label": {str(k): v for k, v in id2label.items()},
        "segments": [
            {"time_sec": round(t0, 3), "end_sec": round(t1, 3),
             "duration_sec": round(t1 - t0, 3),
             "skill_en": lab, "skill_vi": VI_NAMES.get(lab, lab),
             "object_id": oid,
             "object_label": id2label.get(oid, ""),
             "max_height_mm": seg_height(t0, t1)}
            for t0, t1, lab, oid in segments
        ],
    }

    if verbose:
        print(f"Mask    : {mask_json_path}")
        if depth_path:
            print(f"Depth   : {depth_path}  (mốc mặt bàn {out['depth_reference_mm']} mm)")
        print(f"Video   : {data.get('video')}  {data.get('size')}  {fps:.2f} fps  "
              f"{len(per_frame)} frame")
        print(f"Tay     : {[id2label[i] for i in hand_ids]}")
        print(f"Vật thể : {[id2label[i] for i in obj_ids]}")
        print(f"\nChuỗi skill mức thấp:\n{render_summary(segments)}")
    return out, sig, active


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks_json", help="masks_gsam2.json của nhánh A")
    ap.add_argument("--out", default="skills_lowlevel.json")
    ap.add_argument("--depth", default=None,
                    help="file .npz của depth_from_mcap.py — dùng độ cao THẬT (mm) "
                         "cho Lift/Place thay vì vận tốc pixel")
    ap.add_argument("--dump-signals", default=None,
                    help="ghi tín hiệu thô ra JSON để soi/debug")
    args = ap.parse_args()

    out, sig, active = infer(args.masks_json, depth_path=args.depth)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi: {args.out}  ({len(out['segments'])} đoạn)")

    if args.dump_signals:
        dump = {k: [None if (v is None or (isinstance(v, float) and np.isnan(v)))
                    else round(float(v), 3) for v in arr]
                for k, arr in sig.items() if not k.startswith("_")}
        dump["active_object"] = active
        with open(args.dump_signals, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False)
        print(f"Đã ghi tín hiệu: {args.dump_signals}")


if __name__ == "__main__":
    main()
