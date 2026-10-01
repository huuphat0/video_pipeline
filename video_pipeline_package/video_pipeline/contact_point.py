"""
contact_point.py

Xác định **ĐIỂM CHẠM** — chỗ bàn tay đang tiếp xúc trên vật thể — và mô tả nó
nằm ở ĐÂU TRÊN VẬT (đầu trên / thân / đáy), hoàn toàn bằng hình học trên mask.

VÌ SAO CẦN: bản v3 (dùng VLM) có `touch_location` ("nắp", "thân", "quai") nhưng
bản suy luận từ mask đã BỎ MẤT thông tin này. Với việc học từ video người thật
thì đây là dữ liệu quan trọng: biết robot phải NẮM CHỖ NÀO trên vật, không chỉ
biết là "đang nắm".

CÁCH LÀM — ba bước, không có model nào đoán bằng chữ:

  1. VÙNG CHẠM = pixel của vật nằm trong bán kính `dilate_px` quanh bàn tay.
     Nếu mask tay và mask vật không chạm nhau (khe hở nhỏ do SAM2), lấy pixel
     vật GẦN tay nhất — nhưng chỉ nhận khi khoảng cách đó đủ nhỏ, nếu không thì
     coi như CHƯA chạm và trả None.

  2. ĐIỂM CHẠM = trọng tâm vùng chạm. Dùng trọng tâm cả vùng chứ không lấy một
     pixel, vì rìa mask rung vài pixel giữa các frame.

  3. VỊ TRÍ TRÊN VẬT = chiếu điểm chạm lên TRỤC CHÍNH của vật (PCA trên mask),
     chuẩn hoá về 0..1 theo chiều dài vật. Neo lại sao cho t=1 là ĐẦU Ở TRÊN
     trong ảnh — nhờ vậy với chai dựng đứng thì t=1 là phía nắp, t=0 là đáy.
     Vật nằm ngang thì hai đầu gần bằng y nhau nên việc neo không còn ý nghĩa
     "trên/dưới", khi đó chỉ nên đọc giá trị `along` như "đầu này hay đầu kia".

Trục chính dùng PCA giống `hoi_skill_inference._principal_angle_deg` (không dùng
cv2.minAreaRect vì góc của nó phụ thuộc thứ tự đỉnh và hay nhảy ±90°).
"""

import cv2
import numpy as np

# Ngưỡng phân vùng theo chiều dài vật
TOP_FRAC = 0.68        # along >= ngưỡng này -> phần trên (với chai: nắp/cổ)
BOTTOM_FRAC = 0.32     # along <= ngưỡng này -> phần dưới (đáy)
SIDE_FRAC = 0.30       # lệch ngang quá ngưỡng này (so với bề rộng) -> lệch hẳn sang một bên

# Ngưỡng độ dài để chọn TRỤC mô tả vị trí (giống ELONG_MIN bên
# hoi_skill_inference): dưới ngưỡng này vật coi là tròn, trục chính vô nghĩa.
ELONG_MIN = 2.0

CONTACT_DILATE_PX = 6      # nới bàn tay bao nhiêu px để tìm vùng chạm
CONTACT_MIN_PIXELS = 8     # vùng chạm nhỏ hơn -> không tính là đã chạm
MAX_GAP_PX = 15            # mask không chạm nhau: chỉ nhận nếu hở dưới ngưỡng này
MIN_OBJECT_PIXELS = 40     # vật nhỏ hơn -> PCA vô nghĩa, bỏ mô tả vị trí

VI_TOP = "phần trên"
VI_MID = "thân"
VI_BOTTOM = "phần dưới"
VI_LEFT = "bên trái"
VI_RIGHT = "bên phải"


def find_contact(obj_mask, hand_mask, dilate_px=CONTACT_DILATE_PX):
    """Trả (x, y, n_pixels, exact) hoặc None nếu tay chưa chạm vật.

    exact=True  : vùng chạm thật sự chồng lấn (đáng tin)
    exact=False : mask hở một khe nhỏ, điểm trả về là pixel vật gần tay nhất
    """
    if obj_mask is None or hand_mask is None:
        return None
    if not obj_mask.any() or not hand_mask.any():
        return None

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    near = obj_mask & (cv2.dilate(hand_mask.astype(np.uint8), k) > 0)
    n = int(np.count_nonzero(near))
    if n >= CONTACT_MIN_PIXELS:
        ys, xs = np.nonzero(near)
        cx, cy = float(xs.mean()), float(ys.mean())
        # TRỌNG TÂM CÓ THỂ RƠI RA NGOÀI VÙNG CHẠM. Khi bàn tay ôm quanh vật,
        # vùng chạm là một hình VÀNH KHUYÊN theo rìa vật, và trọng tâm của vành
        # khuyên nằm ở GIỮA — tức là rơi vào bàn tay, không phải trên vật. Đo
        # được thật trên video demo: ở t=5.0s và t=8.5s điểm trả về nằm trong
        # mask TAY và ngoài mask CHAI.
        # Khắc phục: kéo về pixel THẬT của vùng chạm gần trọng tâm nhất
        # (medoid) — điểm trả về luôn nằm trên bề mặt vật.
        ii = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        return float(xs[ii]), float(ys[ii]), n, True

    # mask không chạm nhau: đo khoảng cách thật từ vật tới tay
    dt = cv2.distanceTransform((~hand_mask).astype(np.uint8), cv2.DIST_L2, 5)
    d = np.where(obj_mask, dt, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
    if not np.isfinite(d[iy, ix]) or d[iy, ix] > MAX_GAP_PX:
        return None                      # hở quá xa -> chưa chạm
    return float(ix), float(iy), 0, False


def _pca_axis(mask):
    """(tâm, trục chính đơn vị) của mask. Trả None nếu mask quá nhỏ."""
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_OBJECT_PIXELS:
        return None, None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    q = pts - c
    cov = q.T @ q / len(q)
    w, v = np.linalg.eigh(cov)
    return c, v[:, -1]


def describe_location(obj_mask, px, py):
    """Vị trí của điểm chạm trên vật thể. Trả dict hoặc None nếu vật quá nhỏ.

    TRỤC MÔ TẢ ĐƯỢC CHỌN THEO HÌNH DẠNG VẬT — đây là điều kiện để mở rộng ra
    nhiều loại vật:

      - Vật DÀI (chai, kéo, thìa — độ dài >= ELONG_MIN): dùng TRỤC CHÍNH (PCA).
        Với chai dựng đứng thì along=1 là phía nắp, along=0 là đáy.
      - Vật TRÒN (cốc, ly, bát, bóng): trục chính VÔ NGHĨA (nó nhảy loạn giữa
        các frame vì mọi phương gần như tương đương). Dùng phương THẲNG ĐỨNG
        trong ảnh — với cốc/bát thì đó mới là "miệng / thân / đáy" thật.

    along : 0..1 — 1 = phía TRÊN, 0 = phía dưới
    label : 'phần trên' / 'thân' / 'phần dưới'
    side  : 'bên trái' / 'bên phải' / '' — lệch ngang
    axis  : 'trục chính' hoặc 'phương đứng' — để biết đã dùng trục nào
    """
    c, axis = _pca_axis(obj_mask)
    if c is None:
        return None
    ys, xs = np.nonzero(obj_mask)
    pts = np.stack([xs, ys], 1).astype(np.float64) - c
    p = np.array([px, py], dtype=np.float64) - c

    # độ dài = căn bậc hai tỉ lệ 2 trị riêng (giống hoi_skill_inference.ELONG_MIN)
    cov = pts.T @ pts / len(pts)
    w = np.linalg.eigvalsh(cov)
    elong = float(np.sqrt(w[1] / w[0])) if w[0] > 1e-9 else np.inf

    flipped = False
    if elong >= ELONG_MIN:
        proj = pts @ axis
        lo, hi = float(proj.min()), float(proj.max())
        if hi - lo < 1e-6:
            return None
        along = (float(p @ axis) - lo) / (hi - lo)
        # neo: t=1 phải là đầu có y NHỎ HƠN (cao hơn trong ảnh). Trước khi neo,
        # along=1 đang nằm ở đầu `hi` — nên chỉ lật khi đầu `hi` có y LỚN HƠN
        # đầu `lo` (tức `hi` đang là đầu DƯỚI, cần đổi cho `lo` làm along=1).
        # BẢN CŨ dùng dấu `<` — LẬT NGƯỢC hoàn toàn quy ước, làm mọi giá trị
        # `along`/`part` của VẬT DÀI (trục chính) bị đảo top/bottom. Đo trên
        # chính mask chai video demo (frame 10, chai đứng thẳng chưa bị cầm):
        # bản cũ cho NẮP CHAI (y=121, đầu ảnh) along=0.007 "phần dưới" và ĐÁY
        # CHAI (y=329) along=0.995 "phần trên" — ngược hoàn toàn với thực tế.
        # Vật TRÒN (nhánh else bên dưới, dùng phương đứng) không bị lỗi này.
        if (c + hi * axis)[1] > (c + lo * axis)[1]:
            along = 1.0 - along
            flipped = True
        perp = np.array([-axis[1], axis[0]])
        pl = pts @ perp
        axis_name = "trục chính"
    else:
        # vật tròn -> dùng phương ĐỨNG của ảnh
        y0, y1 = float(ys.min()), float(ys.max())
        span = y1 - y0
        if span < 1e-6:
            return None
        along = 1.0 - (py - y0) / span        # 1 = trên, 0 = dưới
        pl = pts[:, 0]                        # lệch ngang = lệch theo x
        axis_name = "phương đứng"

    along = float(np.clip(along, 0.0, 1.0))
    width = float(pl.max() - pl.min())
    if width > 1e-6:
        if axis_name == "trục chính":
            side_off = (float(p @ np.array([-axis[1], axis[0]])) - float(pl.min())) / width
        else:
            side_off = (px - float(xs.min())) / width
    else:
        side_off = 0.5
    if side_off < SIDE_FRAC:
        side = VI_LEFT
    elif side_off > 1.0 - SIDE_FRAC:
        side = VI_RIGHT
    else:
        side = ""

    label = VI_TOP if along >= TOP_FRAC else (VI_BOTTOM if along <= BOTTOM_FRAC else VI_MID)
    out = {"along": round(along, 3), "label": label, "side": side,
           "offset": round(float(side_off), 3), "axis": axis_name,
           "elongation": round(elong, 2)}

    # ĐIỂM ĐỂ VẼ = giao của toạ độ `along` (đã đo, ổn định) với TRỤC CHÍNH của
    # vật. Vẽ ở đây thay vì ở medoid: medoid nằm trên bề mặt nhưng NHẢY quanh
    # vành tiếp xúc nên nhìn video thấy điểm gắp trôi. Điểm trên trục chính thì
    # ổn định (vì `along` ổn định) và vẫn nằm trong lòng vật.
    if axis_name == "trục chính":
        proj_target = (hi - along * (hi - lo)) if flipped else (lo + along * (hi - lo))
        axis_pt = c + proj_target * axis
        out["axis_x"] = float(axis_pt[0])
        out["axis_y"] = float(axis_pt[1])
    else:
        y0, y1 = float(ys.min()), float(ys.max())
        out["axis_x"] = float(c[0])
        out["axis_y"] = float(y1 - along * (y1 - y0))
    return out


_SIDE_LABEL_TO_FRAC = {"": 0.5, VI_LEFT: SIDE_FRAC / 2, VI_RIGHT: 1.0 - SIDE_FRAC / 2}


def apply_grasp_point(obj_mask, along, side="", depth_frame=None, intr=None,
                      patch_radius=3):
    """NGHỊCH ĐẢO của `describe_location`: đã HỌC được `along`/`side` từ video
    khác (camera khác, vật khác, vị trí khác trên bàn) — giờ áp vào 1 MASK VẬT
    MỚI (phát hiện lúc robot sắp gắp thật) để ra pixel + toạ độ 3D HÔM NAY.

    Đây là bước còn thiếu để dùng dữ liệu học từ video cho robot: pipeline sẵn
    có chỉ tính CHIỀU THUẬN (điểm chạm đã biết -> along). Hàm này tính CHIỀU
    NGƯỢC (along đã học -> điểm trên vật mới), dùng ĐÚNG cách chọn trục và ĐÚNG
    hướng neo với `describe_location` (cùng ELONG_MIN, cùng điều kiện flip) —
    nếu hai bên lệch quy ước thì điểm áp ra sẽ sai chỗ mà không có cách nào
    phát hiện bằng mắt (không giống lỗi flip cũ, thấy được vì có ảnh đối chiếu).

    along : 0..1 đã học từ video khác (1 = phía trên, 0 = phía dưới — cùng quy
            ước với `describe_location`, ví dụ lấy từ `grasp_episodes()["grasp_point"]["along"]`)
    side  : 0..1 (offset chính xác nếu có) HOẶC nhãn chữ "bên trái"/"bên phải"/""
            (từ `grasp_points_per_object`/`grasp_episodes`, kém chính xác hơn vì
            đã lượng tử hoá thành 3 mức — dùng số thực nếu có, không thì để "").
    depth_frame, intr: tuỳ chọn (mask depth 1 frame + `load_intrinsics()`) — có
            đủ cả hai thì trả thêm toạ độ (x_mm, y_mm, z_mm) HỆ CAMERA CỦA ẢNH
            NÀY (không phải hệ vật, không phải hệ robot — xem cảnh báo ở
            `contact_point_xyz_mm`, phải tự cộng thêm hiệu chuẩn camera->robot).

    Trả None nếu mask quá nhỏ (< MIN_OBJECT_PIXELS) hoặc suy biến (0 chiều dài).
    """
    if isinstance(side, str):
        side_frac = _SIDE_LABEL_TO_FRAC.get(side, 0.5)
    else:
        side_frac = float(side)

    c, axis = _pca_axis(obj_mask)
    if c is None:
        return None
    ys, xs = np.nonzero(obj_mask)
    pts = np.stack([xs, ys], 1).astype(np.float64) - c
    cov = pts.T @ pts / len(pts)
    w = np.linalg.eigvalsh(cov)
    elong = float(np.sqrt(w[1] / w[0])) if w[0] > 1e-9 else np.inf

    if elong >= ELONG_MIN:
        proj = pts @ axis
        lo, hi = float(proj.min()), float(proj.max())
        if hi - lo < 1e-6:
            return None
        # CÙNG điều kiện flip với describe_location — bắt buộc giống hệt,
        # không thì along học được sẽ áp ngược chiều lên vật mới.
        flipped = (c + hi * axis)[1] > (c + lo * axis)[1]
        proj_target = (hi - along * (hi - lo)) if flipped else (lo + along * (hi - lo))
        perp = np.array([-axis[1], axis[0]])
        pl = pts @ perp
        pw = float(pl.max() - pl.min())
        perp_target = pl.min() + side_frac * pw if pw > 1e-6 else 0.0
        point = c + proj_target * axis + perp_target * perp
        axis_name = "trục chính"
    else:
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        py = y1 - along * (y1 - y0)
        px = x0 + side_frac * (x1 - x0)
        point = np.array([px, py])
        axis_name = "phương đứng"

    px, py = float(point[0]), float(point[1])
    out = {"x_px": round(px, 1), "y_px": round(py, 1), "axis_used": axis_name}

    xyz = contact_point_xyz_mm({"axis_x": px, "axis_y": py}, obj_mask.shape,
                               depth_frame, intr, patch_radius)
    if xyz:
        out["x_mm"], out["y_mm"], out["z_mm"] = xyz["x_mm"], xyz["y_mm"], xyz["z_mm"]
    return out


# Góc theo quy ước TOÁN (y hướng LÊN, đã đảo dấu dy ở dưới):
#   0° = tay ở bên PHẢI vật · 90° = tay ở PHÍA TRÊN · 180° = bên trái · 270° = phía dưới
# Bảng này từng bị viết đảo (45° ghi thành "từ dưới-phải") nên mọi nhãn hướng
# đều ngược — đã kiểm chứng bằng mắt trên video demo: tay hạ xuống từ phía
# trên-bên phải, góc đo ra 46°.
DIR_NAMES = [(0, "từ bên phải"), (45, "từ trên-phải"), (90, "từ phía trên"),
             (135, "từ trên-trái"), (180, "từ bên trái"), (225, "từ dưới-trái"),
             (270, "từ phía dưới"), (315, "từ dưới-phải")]


def approach_direction(obj_mask, hand_mask, contact_xy=None):
    """HƯỚNG TAY ĐI TỚI VẬT, tính trong hệ ẢNH.

    Trả (góc_độ, nhãn) hoặc (None, None).

    Đây là thứ DUY NHẤT cần đến tâm tay, và chỉ cần TRƯỚC khi chạm. Sai lệch do
    mask gồm cẳng tay chỉ làm tâm tay lệch về phía cổ tay, KHÔNG làm đổi hướng
    tay-tới-vật — nên hướng vẫn dùng được dù tâm không chính xác.

    Góc tính theo quy ước ảnh: 0° = tay ở bên PHẢI vật, tăng ngược chiều kim
    đồng (90° = tay ở dưới vật trong ảnh)."""
    if obj_mask is None or hand_mask is None:
        return None, None
    if not obj_mask.any() or not hand_mask.any():
        return None, None
    ys, xs = np.nonzero(obj_mask)
    ox, oy = float(xs.mean()), float(ys.mean())
    if contact_xy is not None:
        ox, oy = contact_xy            # neo vào ĐIỂM CHẠM nếu đã có
    ys, xs = np.nonzero(hand_mask)
    hx, hy = float(xs.mean()), float(ys.mean())

    dx, dy = hx - ox, hy - oy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None, None
    # ảnh có y hướng xuống -> đảo dấu dy để góc đọc theo trực giác hình học
    deg = float(np.degrees(np.arctan2(-dy, dx)) % 360.0)
    best = min(DIR_NAMES, key=lambda kv: min(abs(deg - kv[0]),
                                            360 - abs(deg - kv[0])))
    return round(deg, 1), best[1]


def contact_point_xyz_mm(ci, mask_shape, depth_frame, intr, patch_radius=3):
    """Toạ độ 3D THẬT (mm, hệ CAMERA — không phải hệ vật/hệ robot) của điểm
    chạm `ci` (dict trả về từ `contact_info`).

    Dùng `axis_x`/`axis_y` (điểm ổn định trên trục chính vật, xem
    `describe_location`) làm pixel gốc, KHÔNG dùng medoid (`ci["x"]/["y"]"`) —
    cùng lý do đã đổi bên `render_hoi_video.py::draw_contact_marker`: medoid
    nhảy quanh vành tiếp xúc ~4x nhiều hơn khi tay nắm vòng quanh vật lúc rót.

    `mask_shape`: kích thước mask vật (H, W) — có thể NHỎ HƠN `depth_frame`
    nếu `gsam2_video.py --max-side` đã thu nhỏ frame trước khi detect, nên
    phải quy đổi pixel về đúng hệ của depth_frame (= hệ camera_info) trước
    khi tính X, Y bằng công thức pinhole.

    Trả None nếu thiếu depth/intrinsics hoặc vùng lấy mẫu không có pixel hợp lệ.
    """
    if ci is None or depth_frame is None or intr is None:
        return None
    # LƯU Ý: `ci.get("axis_x", ci["x"])` sẽ CRASH nếu thiếu cả 2 khoá "x" —
    # dict.get() luôn tính giá trị default (ci["x"]) NGAY CẢ KHI "axis_x" đã
    # có, đây là bẫy kinh điển của Python. `apply_grasp_point` gọi hàm này với
    # dict chỉ có "axis_x"/"axis_y" (không có "x") nên phải dùng `in` thay vì
    # `.get(..., ci["x"])`.
    px = ci["axis_x"] if "axis_x" in ci else ci["x"]
    py = ci["axis_y"] if "axis_y" in ci else ci["y"]
    dh, dw = depth_frame.shape[:2]
    mh, mw = mask_shape[:2]
    ix = int(round(px * dw / mw))
    iy = int(round(py * dh / mh))
    x0, x1 = max(0, ix - patch_radius), min(dw, ix + patch_radius + 1)
    y0, y1 = max(0, iy - patch_radius), min(dh, iy + patch_radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_frame[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size < 3:              # quá ít pixel hợp lệ -> không tin số này
        return None
    z_mm = float(np.median(valid))
    x_mm = (ix - intr["cx"]) * z_mm / intr["fx"]
    y_mm = (iy - intr["cy"]) * z_mm / intr["fy"]
    return {"x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1), "z_mm": round(z_mm, 1)}


def contact_info(obj_mask, hand_mask, depth_frame=None, obj_depth_mm=None):
    """Gộp cả hai bước, dùng cho renderer. Trả dict hoặc None.

    Thêm `depth_hint` khi có depth: tay đang chạm từ PHÍA GẦN CAMERA hay phía
    xa — chạm cùng tầm sâu với vật nghĩa là đang nắm đúng mặt hướng về camera.
    """
    hit = find_contact(obj_mask, hand_mask)
    if hit is None:
        return None
    x, y, n, exact = hit
    info = {"x": x, "y": y, "n_pixels": n, "exact": exact}

    # TRỌNG TÂM vùng chạm — dùng để ĐO, không dùng để vẽ.
    # Medoid (x,y) ở trên nằm trên bề mặt nên vẽ mới đúng chỗ, nhưng nó NHẢY
    # lung tung quanh vành tiếp xúc khi tay ôm quanh vật. Trọng tâm thì trơn.
    # Đo được trên video demo, pha rót (tay KHÔNG trượt trên chai):
    #     tâm tay              trôi 44.4 px
    #     medoid chiếu lên trục  trôi 0.309
    #     trọng tâm chiếu lên trục trôi 0.097  <- ổn định nhất
    import cv2 as _cv2
    k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE,
                                   (2 * CONTACT_DILATE_PX + 1,) * 2)
    vung = obj_mask & (_cv2.dilate(hand_mask.astype(np.uint8), k) > 0)
    ys, xs = np.nonzero(vung)
    if xs.size:
        info["cx"], info["cy"] = float(xs.mean()), float(ys.mean())

    loc = describe_location(obj_mask, info.get("cx", x), info.get("cy", y))
    if loc:
        info.update(loc)
        info["text"] = loc["label"] + (f" · {loc['side']}" if loc["side"] else "")

    if depth_frame is not None and obj_depth_mm is not None and not np.isnan(obj_depth_mm):
        ix, iy = int(round(x)), int(round(y))
        hh, ww = depth_frame.shape[:2]
        if 0 <= ix < ww and 0 <= iy < hh:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            patch = cv2.dilate(hand_mask.astype(np.uint8), k) > 0
            vals = depth_frame[patch]
            vals = vals[vals > 0]
            if vals.size >= 10:
                info["hand_depth_mm"] = float(np.median(vals))
                info["depth_delta_mm"] = float(np.median(vals) - obj_depth_mm)
    return info
