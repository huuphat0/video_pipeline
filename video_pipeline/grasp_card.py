"""
grasp_card.py

Tạo **THẺ VẬT (grasp card)** cho mỗi lần cầm trong video demo: ảnh vật LÚC
CHƯA BỊ CHẠM + mask VÙNG NẮM đánh dấu trên chính ảnh đó + đám mây điểm vùng nắm
trong HỆ CỦA VẬT. Thẻ này là đầu vào cho bước sau (`grasp_transfer.py`): so
khớp CẢ VẬT sang góc camera khác rồi chuyển nhãn vùng nắm theo.

VÌ SAO KHÔNG DÙNG ĐIỂM NẮM CŨ: `contact_point.py` đo trên mask 2D nên toạ độ
px/xyz đổi theo camera, `side` bị đảo khi camera ở phía đối diện, và `along`
chỉ ổn khi camera nhìn ngang. Thẻ vật gắn vùng nắm vào BỀ MẶT VẬT (ảnh + 3D).

VÌ SAO CROP CẢ VẬT chứ không crop riêng chỗ nắm: thân chai trơn thì mảnh ảnh
chỗ nắm giống mọi đoạn thân khác -> so khớp nhảy lung tung. Nắp, nhãn, đáy
xung quanh mới là thứ định vị được, nên thẻ giữ cả vật và dùng vùng nắm như
một NHÃN đặt lên ảnh.

CÁCH LÀM cho mỗi lần cầm [a, b) (định nghĩa "đang chạm" dùng chung với
`skill_params.grasp_episodes` qua `validated_contacts`):

  1. FRAME THAM CHIẾU = frame gần nhất TRƯỚC a mà tay (nới REF_HAND_MARGIN_PX)
     chưa chồng lên mask vật -> ảnh bề mặt vật chưa bị che.

  2. CỬA SỔ GIỮ = các frame từ a trở đi mà VẬT CÒN ĐỨNG YÊN (chưa nhấc): mask
     vật không tràn ra ngoài dấu chân ở frame tham chiếu và phần bề mặt còn
     nhìn thấy không đổi độ sâu. Nhờ vậy so được TỪNG PIXEL, không cần căn ảnh.

  3. VÙNG BỊ CHE ở mỗi frame giữ = bỏ phiếu 3 tín hiệu, cần >= 2:
       - mask tay GSAM2 phủ lên pixel đó
       - depth: pixel gần camera hơn bề mặt vật tham chiếu 8..80 mm (ngón tay
         áp trên mặt vật; xa hơn 80 mm là cánh tay đi ngang phía trước)
       - ngoại hình: theo ô 8x8, NCC thấp (ô có vân) hoặc lệch màu LAB lớn
         (ô trơn) so với ảnh tham chiếu, đã bù phơi sáng tự động
     Chỉ giữ các mảng dính vào vùng chạm của `contact_point` (bỏ phần cánh tay
     che vật mà không chạm).

  4. VÙNG NẮM = pixel bị che ở >= GRASP_FREQ_MIN số frame ĐÃ NẮM CHẮC (vùng
     che >= SETTLED_FRAC mức lớn nhất — tay khép dần nên frame đầu chỉ có
     đầu ngón). Đám mây điểm lấy
     từ depth của FRAME THAM CHIẾU — lúc đó chưa có ngón tay nên đây đúng là
     bề mặt vật nằm dưới ngón tay.

  5. NEO + MẢNG (cần depth): fit MẶT BÀN quanh chân vật (RANSAC + SVD), lấy
     pháp tuyến bàn làm chiều "lên". B = trục vật cắt mặt bàn (đáy thật, tính
     được cả khi đáy bị che), T = đỉnh. Vùng đỏ tách thành các MẢNG dọc trục
     (cắt ở khe giữa các ngón); mỗi mảng ghi mép dưới/mép trên/đỉnh theo mm
     từ B, mm từ T và tỉ lệ along theo B–T. Không lấy trung vị cả vùng đỏ:
     trên demo nó bị mảng phụ kéo lên ~14 mm và có thể rơi vào khe. Đo trên
     demo, che đáy tới 60 px: B lệch < 1 mm dọc trục, trong khi `along` theo
     phần nhìn thấy (cách cũ) lệch tới +44 mm.

GIỚI HẠN: ngón tay ở MẶT SAU vật (khuất camera) không đo được; vùng nắm là
phần NHÌN THẤY từ camera demo. Cần ít nhất một frame vật chưa bị che trước
khi nắm.

Cách dùng (DINOv2 cần transformers -> chạy bằng venv gốc của repo):
    ../../../venv/bin/python grasp_card.py ../../out/gsam2_demo/masks_gsam2_merged.json \\
        --depth ../../out/npz/demo_depth.npz --video ../../out/mp4/demo_color.mp4
    python3 grasp_card.py ... --no-dino      # không cần transformers
"""

import argparse
import json
import os
import re

import cv2
import numpy as np

import hoi_skill_inference as H
import contact_point as CP
import skill_params as SP
import subskill_inference as S

MIN_EPISODE_FRAMES = 5         # giống mặc định skill_params.grasp_episodes
REF_HAND_MARGIN_PX = 12        # tay phải cách mask vật ít nhất chừng này ở frame tham chiếu
REF_LOOKBACK_SEC = 3.0         # tìm frame tham chiếu trong khoảng này trước lúc chạm
MAX_HOLD_SEC = 2.0             # cửa sổ giữ dài tối đa
STATIC_OUTSIDE_FRAC = 0.04     # mask vật tràn ra ngoài dấu chân tham chiếu quá -> vật đã dịch
STATIC_DEPTH_MM = 6.0          # bề mặt còn thấy lệch độ sâu quá -> vật đã dịch
STATIC_MAX_MISSES = 3          # cho phép vài frame nhiễu trước khi coi là đã nhấc

DEPTH_FRONT_MIN_MM = 8.0       # pixel gần camera hơn bề mặt tham chiếu ít nhất chừng này
DEPTH_FRONT_MAX_MM = 80.0      # ... và không quá chừng này (xa hơn = cánh tay phía trước)
CELL_PX = 8                    # ô so sánh ngoại hình
CELL_MIN_STD = 4.0             # ô có độ lệch chuẩn xám dưới ngưỡng = ô trơn -> dùng LAB
NCC_CHANGED = 0.5              # NCC dưới ngưỡng -> ô đã đổi
LAB_CHANGED = 12.0             # lệch màu LAB trung bình trên ngưỡng -> ô đã đổi
SETTLED_FRAC = 0.6             # frame có vùng che >= 60% mức lớn nhất = đã nắm chắc
GRASP_FREQ_MIN = 0.5           # bị che ở >= 50% frame nắm chắc -> thuộc vùng nắm
MIN_REGION_PX = 20             # mảng nhỏ hơn -> bỏ
CROP_PAD_FRAC = 0.15           # lề crop quanh bbox vật
SLICE_MM = 10.0                # bề dày lát khi dựng trục trụ
SLICE_MIN_POINTS = 30          # lát ít điểm hơn -> bỏ
ON_SURFACE_MM = 8.0            # điểm cách mặt trụ dự đoán dưới ngưỡng này = khớp

TABLE_RING_PX = 80             # vành quanh vật để tìm mặt bàn
TABLE_INLIER_MM = 4.0          # ngưỡng inlier ở z = 500 mm (tăng theo z^2)
TABLE_RANSAC_ITERS = 300
TABLE_MIN_INLIERS = 500        # ít hơn -> không tin mặt bàn
TABLE_MAX_TILT_DEG = 20.0      # trục vật lệch pháp tuyến bàn quá -> không phải bàn / vật không đứng
TABLE_MAX_PLANES = 3           # thử tối đa chừng này mặt phẳng (bỏ tường)
BOTTOM_BELOW_MM = 10.0         # điểm vật nằm dưới mặt phẳng quá -> mặt phẳng sai
BOTTOM_GAP_MM = 15.0           # đáy nhìn thấy cách mặt bàn quá -> cảnh báo (bị che / khay)
TOP_PCT = 99.5                 # đỉnh = phân vị này của điểm vật dọc trục (bỏ nhiễu)
TOP_SLICE_MM = 10.0            # lát trên cùng để kiểm tra đỉnh có đủ điểm
TOP_MIN_POINTS = 20

BAND_BIN_MM = 5.0              # bin histogram dọc trục khi tách mảng vùng nắm
BAND_EMPTY_FRAC = 0.08         # bin thấp hơn chừng này x đỉnh = rỗng (cắt)
VALLEY_FRAC = 0.5              # thung lũng thấp hơn chừng này x đỉnh thấp hơn hai bên = khe
BAND_MIN_SHARE = 0.08          # mảng ít hơn 8% điểm vùng đỏ -> bỏ (nhiễu, chạm thoáng)

# kiểm tra chéo ở góc nhìn mới (resolve_grasp)
BT_TOL_MM = 8.0                # |B–T mới − B–T thẻ| trong ngưỡng này (hoặc BT_TOL_FRAC) = hai neo khớp
BT_TOL_FRAC = 0.04
RADIUS_TOL_FRAC = 0.10         # bán kính cùng độ cao lệch quá 10% -> vật khác cỡ
PROFILE_MIN_SLICES = 4         # cần ít nhất chừng này lát chồng lấn để so bán kính

DINO_ID = "facebook/dinov2-base"   # cùng model với sam_dinov2/, đã có trong cache HF
DINO_LONG_SIDE = 448               # cạnh dài ảnh đưa vào DINOv2 (bội số của 14)


# --------------------------------------------------------------- tiện ích
def _resize_mask(m, shape):
    if m.shape[:2] == tuple(shape):
        return m
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _hand_mask(frame_masks, hand_ids, shape):
    hm = np.zeros(shape, bool)
    for h in hand_ids:
        if h in frame_masks:
            hm |= _resize_mask(frame_masks[h], shape)
    return hm


def _dilate(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.dilate(m.astype(np.uint8), k) > 0


def _erode(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.erode(m.astype(np.uint8), k) > 0


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def resolve_video(masks_json, data, video_arg):
    """Đường dẫn video trong masks json là TƯƠNG ĐỐI với nơi chạy gsam2 lúc
    đó (vd `../out/demo_color.mp4` từ sam_dino/), và file có thể đã bị dời
    (out/ đã chia thư mục con theo đuôi file). Thử lần lượt vài chỗ."""
    if video_arg:
        return video_arg
    v = data.get("video", "")
    base = os.path.dirname(os.path.abspath(masks_json))
    cands = [v, os.path.join(base, v), os.path.join(base, "..", v)]
    name = os.path.basename(v)
    for root in (os.path.join(base, ".."), os.path.join(base, "..", "..")):
        cands += [os.path.join(root, "mp4", name), os.path.join(root, "out", "mp4", name)]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.normpath(c)
    raise FileNotFoundError(f"không tìm thấy video '{v}' — truyền --video")


class FrameReader:
    def __init__(self, path, shape):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"không mở được video {path}")
        self.shape = shape

    def get(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = self.cap.read()
        if not ok:
            return None
        if fr.shape[:2] != self.shape:
            fr = cv2.resize(fr, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        return fr


def load_depth_aligned(depth_path, shape):
    """Depth + intrinsics đưa về đúng cỡ mask (gsam2 --max-side có thể đã thu
    nhỏ). Co giãn depth thì phải co giãn cả fx/fy/cx/cy theo."""
    if not depth_path:
        return None, None
    depth = H.load_depth(depth_path)[0]
    intr = H.load_intrinsics(depth_path)
    dh, dw = depth.shape[1:3]
    if (dh, dw) != tuple(shape):
        sx, sy = shape[1] / dw, shape[0] / dh
        depth = np.stack([cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                          for d in depth])
        if intr:
            intr = {"fx": intr["fx"] * sx, "fy": intr["fy"] * sy,
                    "cx": intr["cx"] * sx, "cy": intr["cy"] * sy}
    return depth, intr


# ----------------------------------------------------- bước 1: frame tham chiếu
def find_reference_frame(per_frame, hand_ids, oid, a, fps, shape, depth=None):
    """Frame gần nhất trước `a` mà tay chưa phủ lên vật. None nếu không có."""
    lo = max(0, a - int(round(REF_LOOKBACK_SEC * fps)))
    for i in range(a - 1, lo - 1, -1):
        om = per_frame[i].get(oid)
        if om is None or not om.any():
            continue
        om = _resize_mask(om, shape)
        hm = _hand_mask(per_frame[i], hand_ids, shape)
        if hm.any() and (_dilate(hm, REF_HAND_MARGIN_PX) & om).any():
            continue
        if depth is not None and i < len(depth) and np.count_nonzero(depth[i][om] > 0) < 30:
            continue
        return i
    return None


# ------------------------------------------------------- bước 2: cửa sổ giữ
def hold_window(per_frame, hand_ids, oid, a, b, fps, ref_mask, ref_depth, depth, shape):
    """Các frame từ `a` mà vật còn nằm yên ở chỗ cũ. Trả list chỉ số frame."""
    foot = _dilate(ref_mask, 4)
    out, misses = [], 0
    stop = min(b, a + int(round(MAX_HOLD_SEC * fps)), len(per_frame))
    for t in range(a, stop):
        om = per_frame[t].get(oid)
        if om is None or not om.any():
            misses += 1
            if misses >= STATIC_MAX_MISSES:
                break
            continue
        om = _resize_mask(om, shape)
        outside = np.count_nonzero(om & ~foot) / max(1, np.count_nonzero(om))
        static = outside < STATIC_OUTSIDE_FRAC
        if static and depth is not None and ref_depth is not None and t < len(depth):
            hm = _dilate(_hand_mask(per_frame[t], hand_ids, shape), 3)
            vis = ref_mask & om & ~hm
            dr, dt = ref_depth[vis].astype(np.float64), depth[t][vis].astype(np.float64)
            ok = (dr > 0) & (dt > 0)
            # mask tay GSAM2 hay SÓT ngón tay -> `vis` lẫn pixel ngón nằm TRƯỚC
            # bề mặt vật. Đo trên demo: không loại chúng thì cửa sổ giữ dừng ở
            # frame 68 dù vật đứng yên tới ~80. Pixel "ở trước" là che, không
            # phải vật dịch; nhưng nếu gần như MỌI pixel đều ở trước thì là vật
            # bị kéo về phía camera -> coi như đã dịch.
            front = ok & (dr - dt >= DEPTH_FRONT_MIN_MM)
            rest = ok & ~front
            if ok.sum() >= 30:
                if rest.sum() < 0.3 * ok.sum():
                    static = False
                else:
                    static = float(np.median(np.abs(dt[rest] - dr[rest]))) < STATIC_DEPTH_MM
        if static:
            out.append(t)
            misses = 0
        else:
            misses += 1
            if misses >= STATIC_MAX_MISSES:
                break
    return out


# ----------------------------------------------------- bước 3: vùng bị che
def appearance_changed(ref_bgr, cur_bgr, region, vis_uncovered):
    """Mask pixel trong `region` có ngoại hình khác ảnh tham chiếu, so theo ô
    CELL_PX. Bù phơi sáng: nhân kênh L của frame hiện tại cho khớp trung vị L
    tham chiếu trên phần vật chắc chắn KHÔNG bị che (`vis_uncovered`)."""
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_lab = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if np.count_nonzero(vis_uncovered) >= 50:
        lr = float(np.median(ref_lab[..., 0][vis_uncovered]))
        lc = float(np.median(cur_lab[..., 0][vis_uncovered]))
        if lc > 1.0:
            cur_lab[..., 0] = np.clip(cur_lab[..., 0] * (lr / lc), 0, 255)
    rg, cg = ref_lab[..., 0], cur_lab[..., 0]

    out = np.zeros(region.shape, bool)
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    for cy in range(y0, y1, CELL_PX):
        for cx in range(x0, x1, CELL_PX):
            sl = (slice(cy, min(cy + CELL_PX, y1)), slice(cx, min(cx + CELL_PX, x1)))
            m = region[sl]
            if np.count_nonzero(m) < CELL_PX * CELL_PX // 4:
                continue
            a, c = rg[sl][m], cg[sl][m]
            if a.std() >= CELL_MIN_STD and c.std() >= CELL_MIN_STD:
                ncc = float(np.mean((a - a.mean()) * (c - c.mean())) / (a.std() * c.std()))
                changed = ncc < NCC_CHANGED
            else:
                de = np.linalg.norm(ref_lab[sl][m] - cur_lab[sl][m], axis=1)
                changed = float(de.mean()) > LAB_CHANGED
            if changed:
                out[sl] |= m
    return out


def occlusion_mask(ref_bgr, ref_mask, ref_depth, cur_bgr, cur_masks, hand_ids, cur_depth, shape):
    """Pixel của vật tham chiếu đang bị TAY che ở frame hiện tại (xem docstring
    module, bước 3). Trả (mask, dict số pixel từng tín hiệu)."""
    region = _erode(ref_mask, 2)           # bỏ rìa mask: rung vài px giữa các frame
    hm = _hand_mask(cur_masks, hand_ids, shape)
    hand_v = _dilate(hm, 3) & region

    votes = hand_v.astype(np.uint8)
    depth_v = np.zeros(shape, bool)
    if ref_depth is not None and cur_depth is not None:
        dr, dt = ref_depth.astype(np.float32), cur_depth.astype(np.float32)
        front = dr - dt
        depth_v = region & (dr > 0) & (dt > 0) & (front >= DEPTH_FRONT_MIN_MM) & (front <= DEPTH_FRONT_MAX_MM)
        votes += depth_v

    vis_uncovered = region & ~_dilate(hm, 8) & ~depth_v
    app_v = appearance_changed(ref_bgr, cur_bgr, region, vis_uncovered)
    votes += app_v
    occ = region & (votes >= 2)

    # chỉ giữ mảng dính vào vùng chạm (tay áp lên vật), bỏ phần chỉ che ngang
    contact = region & _dilate(hm, CP.CONTACT_DILATE_PX)
    seed = _dilate(contact, 5)
    n, lab = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
    keep = np.zeros(shape, bool)
    for k in range(1, n):
        comp = lab == k
        if (comp & seed).any():
            keep |= comp
    return keep, {"hand": int(hand_v.sum()), "depth": int(depth_v.sum()),
                  "appearance": int(app_v.sum()), "kept": int(keep.sum())}


def clean_region(m):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros(m.shape, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= MIN_REGION_PX:
            out |= lab == k
    return out


# ----------------------------------------------------- bước 4: mô tả vùng nắm
def describe_region_2d(obj_mask, grasp_mask, max_samples=300):
    """`along` 2D (đúng quy ước contact_point.describe_location) của các pixel
    vùng nắm -> khoảng [p5, p95]. Chỉ để đối chiếu với cách cũ; số chính là 3D."""
    ys, xs = np.nonzero(grasp_mask)
    if xs.size == 0:
        return None
    idx = np.linspace(0, xs.size - 1, min(max_samples, xs.size)).astype(int)
    al = []
    for i in idx:
        d = CP.describe_location(obj_mask, float(xs[i]), float(ys[i]))
        if d:
            al.append(d["along"])
    if not al:
        return None
    al = np.array(al)
    return {"along_min": round(float(np.percentile(al, 5)), 3),
            "along_max": round(float(np.percentile(al, 95)), 3),
            "along_center": round(float(np.median(al)), 3),
            "axis": d["axis"]}


def _backproject_pixels(mask, depth_frame, intr, zlim=None):
    ys, xs = np.nonzero(mask)
    z = depth_frame[ys, xs].astype(np.float64)
    ok = z > 0
    if zlim is not None:
        ok &= (z >= zlim[0]) & (z <= zlim[1])
    xs, ys, z = xs[ok], ys[ok], z[ok]
    x = (xs - intr["cx"]) * z / intr["fx"]
    y = (ys - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], 1)


def _backproject_silhouette(mask, depth_frame, intr, zlim):
    """MỌI pixel của mask -> 3D, để đo BỀ NGANG (bán kính) theo viền bóng vật.

    `H.backproject_mask_xyz` lọc Tukey bỏ pixel rìa có depth lẫn nền — đúng
    cho PCA nhưng làm HẸP bề ngang: đo trên demo, bỏ ~1000/12500 pixel rìa,
    đường kính thân chai ra 46-49 mm trong khi viền mask 2D cho ~57 mm. Vị
    trí PIXEL của chúng vẫn đúng, chỉ depth sai -> giữ pixel, thay depth hỏng
    bằng depth trung vị hàng xóm trong mask."""
    ys, xs = np.nonzero(mask)
    z = depth_frame[ys, xs].astype(np.float64)
    good = (z >= zlim[0]) & (z <= zlim[1])
    if good.sum() < 10:
        return None
    zf = np.where(good, z, np.nan)
    img = np.full(mask.shape, np.nan)
    img[ys, xs] = zf
    # lấp depth hỏng bằng trung vị các pixel tốt trong cửa sổ 15x15
    bad = np.nonzero(~good)[0]
    for i in bad:
        y, x = ys[i], xs[i]
        win = img[max(0, y - 7):y + 8, max(0, x - 7):x + 8]
        v = win[np.isfinite(win)]
        zf[i] = np.median(v) if v.size else np.nanmedian(zf)
    x3 = (xs - intr["cx"]) * zf / intr["fx"]
    y3 = (ys - intr["cy"]) * zf / intr["fy"]
    return np.stack([x3, y3, zf], 1)


def _frame_basis(e1, c):
    """e1 (trục vật) + tâm c -> ma trận R (hàng = e1, e2, e3) trong hệ camera.
    e2 = phần vuông góc e1 của hướng TỪ TÂM VẬT TỚI CAMERA demo ("mặt trước"),
    e3 = e1 x e2."""
    to_cam = -c / np.linalg.norm(c)
    e2 = to_cam - (to_cam @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)])


def _slice_profile(q, lo, hi, qs=None):
    """Cắt đám mây (đã ở hệ vật: cột 0 = dọc trục, 1 = về phía camera, 2 =
    ngang) thành lát SLICE_MM, mỗi lát ước lượng một mặt cắt tròn.

    KHÔNG FIT ĐƯỜNG TRÒN trên cung nhìn thấy: depth RealSense trên nhựa trong
    rất nhiễu (đọc xuyên ra nền), cung <= 180° nên fit tròn bị kéo lệch mạnh.
    Dùng hai đại lượng bền hơn:
      - bán kính r = nửa bề NGANG (cột 2, vuông góc cả trục lẫn hướng nhìn) —
        đo trên đám mây VIỀN BÓNG `qs` (`_backproject_silhouette`), không cần
        depth chính xác ở rìa;
      - mặt trước = điểm GẦN CAMERA NHẤT (cột 1 lớn nhất) — với mặt trụ, điểm
        này cách trục đúng r, nên tâm lát = mặt trước lùi lại r.
    Trả list dict {s, along, r, cu, cw, n} theo thứ tự dọc trục."""
    out = []
    edges = np.arange(lo, hi + SLICE_MM, SLICE_MM)
    for s0, s1 in zip(edges[:-1], edges[1:]):
        sel = (q[:, 0] >= s0) & (q[:, 0] < s1)
        if sel.sum() < SLICE_MIN_POINTS:
            continue
        u = q[sel, 1]
        if qs is not None:
            sel_s = (qs[:, 0] >= s0) & (qs[:, 0] < s1)
            w = qs[sel_s, 2] if sel_s.sum() >= SLICE_MIN_POINTS else q[sel, 2]
        else:
            w = q[sel, 2]
        w_lo, w_hi = np.percentile(w, [0.5, 99.5])
        r = (w_hi - w_lo) / 2.0
        if r < 2.0:
            continue
        front = np.percentile(u, 97)
        s = (s0 + s1) / 2.0
        out.append({"s": s, "along": (s - lo) / max(hi - lo, 1e-6), "r": r,
                    "cu": front - r, "cw": (w_hi + w_lo) / 2.0, "n": int(sel.sum())})
    return out


def _orient_up(e1, up):
    """Chọn chiều e1 hướng LÊN. Có mặt bàn -> theo pháp tuyến bàn (đúng với
    mọi góc camera). Không có -> dự phòng theo y ảnh (đầu CAO HƠN trong ảnh),
    quy ước cũ của contact_point — sai khi camera xoay nghiêng mạnh."""
    if up is not None:
        return e1 if e1 @ up >= 0 else -e1
    return -e1 if e1[1] > 0 else e1        # y camera hướng XUỐNG


def object_frame_3d(obj_mask, depth_frame, intr, up=None):
    """Hệ toạ độ CỦA VẬT dựng từ đám mây điểm frame tham chiếu.

    e1 = trục vật, chiều theo `_orient_up` (+e1 = phía nắp với chai đứng).
    e2 = phần vuông góc e1 của hướng TỪ TRỤC VẬT TỚI CAMERA demo ("mặt trước").
    e3 = e1 x e2.

    TÂM PHẢI NẰM TRÊN TRỤC THẬT, không phải trung bình điểm nhìn thấy: camera
    chỉ thấy MẶT TRƯỚC nên trung bình rơi lên VỎ vật. Bản đầu làm vậy -> đo
    trên demo: điểm vùng nắm chỉ cách "tâm" -15..+14 mm theo hướng camera (bán
    kính chai ~30 mm), azimuth ra -131..79° (trải 210°, vô lý vì camera chỉ
    thấy <= 180°). Vật DÀI: coi như trụ tròn, dựng trục từ tâm từng lát
    (`_slice_profile`) rồi fit thẳng. Vật KHÔNG dài: trục PCA vô nghĩa, chỉ lùi
    tâm về sau mặt trước một nửa bề ngang và KHÔNG báo azimuth.

    Trả (frame_dict, geo) — geo là dữ liệu nội bộ cho `compute_anchors` /
    `grasp_bands`: c, R, q (điểm vật trong hệ vật), prof, zlim, elongated."""
    P = H.backproject_mask_xyz(obj_mask, depth_frame, intr)
    pa = H.principal_axis_3d(P)
    if pa is None:
        return None, None
    c = pa["center_mm"]
    e1 = _orient_up(pa["axis"] / np.linalg.norm(pa["axis"]), up)
    elongated = pa["elongation"] >= CP.ELONG_MIN
    method = "trụ theo lát" if elongated else "lùi nửa bề ngang"

    zlim = (float(P[:, 2].min()), float(P[:, 2].max()))
    Ps = _backproject_silhouette(obj_mask, depth_frame, intr, zlim)
    R = _frame_basis(e1, c)
    q = (P - c) @ R.T
    qs = (Ps - c) @ R.T if Ps is not None else None
    lo, hi = float(q[:, 0].min()), float(q[:, 0].max())
    prof = _slice_profile(q, lo, hi, qs) if elongated else []

    if elongated and len(prof) >= 3:
        # fit thẳng tâm các lát theo s (trọng số = số điểm) -> trục thật
        s = np.array([p["s"] for p in prof])
        wts = np.array([p["n"] for p in prof], float)
        bu, au = np.polyfit(s, [p["cu"] for p in prof], 1, w=np.sqrt(wts))
        bw, aw = np.polyfit(s, [p["cw"] for p in prof], 1, w=np.sqrt(wts))
        c = c + au * R[1] + aw * R[2]
        e1 = R[0] + bu * R[1] + bw * R[2]
        e1 = _orient_up(e1 / np.linalg.norm(e1), up)
        R = _frame_basis(e1, c)
        q = (P - c) @ R.T
        qs = (Ps - c) @ R.T if Ps is not None else None
        lo, hi = float(q[:, 0].min()), float(q[:, 0].max())
        prof = _slice_profile(q, lo, hi, qs)
    else:
        if elongated:
            method = "lùi nửa bề ngang (quá ít lát)"
        w_lo, w_hi = np.percentile((qs if qs is not None else q)[:, 2], [0.5, 99.5])
        r = (w_hi - w_lo) / 2.0
        c = c + (np.percentile(q[:, 1], 97) - r) * R[1] + (w_hi + w_lo) / 2.0 * R[2]
        R = _frame_basis(e1, c)
        q = (P - c) @ R.T

    radii = np.array([p["r"] for p in prof]) if prof else None
    frame = {"method": method,
             "up_from": "mặt bàn" if up is not None else "trục y ảnh (dự phòng)",
             "center_cam_mm": c.round(2).tolist(), "R_obj_in_cam": R.round(6).tolist(),
             "elongation_3d": round(float(pa["elongation"]), 2),
             "radius_mm_median": (round(float(np.median(radii)), 1) if radii is not None else None)}
    geo = {"c": c, "R": R, "q": q, "prof": prof, "zlim": zlim, "elongated": elongated}
    return frame, geo


# ----------------------------------------------------- neo mặt bàn / nắp
def fit_table_plane(depth_frame, intr, frame_masks, obj_mask, shape, axis_hint=None, seed=0):
    """Mặt phẳng BÀN quanh chân vật -> (plane dict hoặc None, lý do).

    Ứng viên = vành RING_PX quanh vật, bỏ mọi mask (vật, tay, vật khác) nới
    thêm 8 px. Không lấy cả ảnh: mặt phẳng lớn nhất có thể là TƯỜNG/vách sau.
    RANSAC 3 điểm (ngưỡng tăng theo độ sâu vì sai số RealSense ~ z^2), tinh
    chỉnh SVD trên inlier, pháp tuyến quay về phía camera (= "lên" với bàn
    nhìn từ trên). Mặt phẳng thắng mà lệch trục vật > TABLE_MAX_TILT_DEG thì
    bỏ inlier của nó và thử mặt phẳng kế (tối đa TABLE_MAX_PLANES) — vành
    quanh vật có thể lẫn tường phía sau.

    Đo trên demo (cả nửa dưới ảnh): 133 212 / 144 524 điểm khớp, camera nhìn
    xuống bàn ~38°, trục chai lệch pháp tuyến 5.8°."""
    busy = np.zeros(shape, bool)
    for m in frame_masks.values():
        busy |= _resize_mask(m, shape)
    ring = _dilate(obj_mask, TABLE_RING_PX) & ~_dilate(busy, 8) & (depth_frame > 0)
    ys, xs = np.nonzero(ring)
    if xs.size < TABLE_MIN_INLIERS:
        return None, f"quá ít pixel quanh vật có depth ({xs.size})"
    z = depth_frame[ys, xs].astype(np.float64)
    X = np.stack([(xs - intr["cx"]) * z / intr["fx"], (ys - intr["cy"]) * z / intr["fy"], z], 1)
    thr = float(np.clip(TABLE_INLIER_MM * (np.median(z) / 500.0) ** 2, 3.0, 12.0))

    rng = np.random.default_rng(seed)
    alive = np.ones(len(X), bool)
    reason = "không có mặt phẳng nào"
    for _ in range(TABLE_MAX_PLANES):
        idx = np.nonzero(alive)[0]
        if idx.size < TABLE_MIN_INLIERS:
            break
        best_n, best_mask = 0, None
        for _ in range(TABLE_RANSAC_ITERS):
            p = X[rng.choice(idx, 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            m = alive & (np.abs((X - p[0]) @ (n / nn)) < thr)
            k = int(m.sum())
            if k > best_n:
                best_n, best_mask = k, m
        if best_mask is None or best_n < TABLE_MIN_INLIERS:
            reason = f"mặt phẳng tốt nhất chỉ có {best_n} điểm khớp"
            break
        pts = X[best_mask]
        c = pts.mean(0)
        n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
        if n @ (-c) < 0:
            n = -n
        inl = np.abs((X - c) @ n) < thr
        tilt = None
        if axis_hint is not None:
            tilt = float(np.degrees(np.arccos(min(1.0, abs(float(n @ axis_hint))))))
        if tilt is not None and tilt > TABLE_MAX_TILT_DEG:
            reason = f"mặt phẳng lớn nhất lệch trục vật {tilt:.0f}° (tường?)"
            alive &= ~inl
            continue
        return {"normal": n, "point": c, "n_inliers": int(inl.sum()), "n_candidates": int(len(X)),
                "inlier_mm": round(thr, 1), "tilt_deg": tilt,
                "camera_pitch_deg": float(np.degrees(np.arcsin(min(1.0, abs(n[2])))))}, None
    return None, reason


def compute_anchors(geo, plane, obj_mask, shape, intr):
    """Hai MỐC trên trục vật, tính bằng s (mm dọc e1 tính từ tâm c):

      B = trục vật CẮT MẶT BÀN — đáy thật, tính được cả khi đáy bị che vì
          chỉ cần mặt bàn và trục. Không có bàn hợp lệ -> dùng điểm thấp nhất
          nhìn thấy (`bottom_from` ghi rõ), khi đó neo đáy KHÔNG bền với che.
      T = đỉnh = phân vị TOP_PCT của điểm vật chiếu lên trục.

    Kèm các kiểm tra (xem `checks`): đáy nhìn thấy có chạm mặt phẳng không
    (khay/hộp), nắp có bị cắt bởi mép ảnh không, đỉnh có đủ điểm depth không."""
    q, c, R = geo["q"], geo["c"], geo["R"]
    e1 = R[0]
    s_vis_lo = float(np.percentile(q[:, 0], 100 - TOP_PCT))
    s_T = float(np.percentile(q[:, 0], TOP_PCT))
    checks, warnings = {}, []

    table_ok = plane is not None
    s_B = None
    if table_ok:
        n, p0 = plane["normal"], plane["point"]
        denom = float(n @ e1)
        if abs(denom) < 0.3:
            table_ok = False
            warnings.append("trục vật gần song song mặt bàn (vật nằm?) — không dùng neo bàn")
        else:
            s_B = float(-(n @ (c - p0)) / denom)
            heights = (q @ R) @ n + float(n @ (c - p0))        # độ cao từng điểm vật so với bàn
            below = float(np.percentile(heights, 1))
            checks["lowest_visible_height_mm"] = round(below, 1)
            if below < -BOTTOM_BELOW_MM:
                table_ok = False
                warnings.append(f"điểm vật nằm DƯỚI mặt phẳng {below:.0f} mm — mặt phẳng sai")
            elif s_vis_lo - s_B > BOTTOM_GAP_MM:
                warnings.append(f"đáy nhìn thấy cách mặt bàn {s_vis_lo - s_B:.0f} mm: đáy bị che, "
                                f"hoặc vật đặt trên khay/hộp (khi đó neo bàn SAI)")
            checks["bottom_gap_mm"] = round(s_vis_lo - s_B, 1)
    if not table_ok:
        s_B = s_vis_lo

    # nắp: bị mép ảnh cắt? (pixel mask sát mép ở nửa TRÊN của vật)
    top_ok = True
    ys, xs = np.nonzero(obj_mask)
    edge = (xs <= 1) | (ys <= 1) | (xs >= shape[1] - 2) | (ys >= shape[0] - 2)
    if edge.any():
        top_px = H.project_xyz_to_pixel(c + s_T * e1, intr)
        bot_px = H.project_xyz_to_pixel(c + s_vis_lo * e1, intr)
        if top_px and bot_px:
            ex, ey = xs[edge].mean(), ys[edge].mean()
            if np.hypot(ex - top_px[0], ey - top_px[1]) < np.hypot(ex - bot_px[0], ey - bot_px[1]):
                top_ok = False
                warnings.append("vật chạm mép ảnh ở phía nắp — đỉnh có thể bị cắt, không dùng neo nắp")
    n_top = int(np.sum(q[:, 0] >= s_T - TOP_SLICE_MM))
    checks["top_slice_points"] = n_top
    if n_top < TOP_MIN_POINTS:
        top_ok = False
        warnings.append(f"chỉ {n_top} điểm depth ở {TOP_SLICE_MM:.0f} mm trên cùng — đỉnh không chắc")

    B = c + s_B * e1
    T = c + s_T * e1
    return {
        "table_valid": bool(table_ok), "top_valid": bool(top_ok),
        "bottom_from": "trục cắt mặt bàn" if table_ok else "điểm thấp nhất nhìn thấy (dự phòng)",
        "B_cam_mm": B.round(2).tolist(), "T_cam_mm": T.round(2).tolist(),
        "length_BT_mm": round(s_T - s_B, 1),
        "visible_length_mm": round(s_T - s_vis_lo, 1),
        "checks": checks, "warnings": warnings,
        "_s_B": s_B, "_s_T": s_T,
    }


# ----------------------------------------------------- mảng vùng nắm
def _split_runs(h):
    """Chỉ số bin -> các đoạn [a, b) là MẢNG: cắt ở khe (thung lũng sâu hơn
    VALLEY_FRAC lần đỉnh thấp hơn ở hai bên) và ở bin gần rỗng."""
    n = len(h)
    if n == 0 or h.max() <= 0:
        return []
    cut = h < BAND_EMPTY_FRAC * h.max()
    for m in range(1, n - 1):
        if h[m] <= h[m - 1] and h[m] <= h[m + 1]:
            lp, rp = h[:m].max(), h[m + 1:].max()
            if h[m] < VALLEY_FRAC * min(lp, rp):
                cut[m] = True
    runs, i = [], 0
    while i < n:
        if cut[i]:
            i += 1
            continue
        j = i
        while j < n and not cut[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def grasp_bands(grasp_mask, depth_frame, intr, geo, anchors):
    """Vùng đỏ -> các MẢNG dọc trục vật (dict, điểm vùng nắm hệ vật gốc B).

    Không dùng trung vị cả vùng: vùng đỏ demo có HAI mảng (eo chai + ngón trên
    nhãn) và khe ở giữa; trung vị bị mảng trên kéo lên ~14 mm so với tâm mảng
    chính, và có thể rơi đúng vào khe (chỗ tay không chạm). Mỗi mảng ghi mép
    dưới / mép trên / đỉnh theo 3 đơn vị: mm từ B (neo bàn), mm từ T (neo
    nắp), tỉ lệ along theo B–T (0 = đáy, 1 = đỉnh, đúng quy ước world.py)."""
    G = _backproject_pixels(grasp_mask, depth_frame, intr, geo["zlim"])
    if len(G) < 10:
        return None, None
    c, R = geo["c"], geo["R"]
    g = (G - c) @ R.T
    s_B, s_T = anchors["_s_B"], anchors["_s_T"]
    L = max(s_T - s_B, 1e-6)
    sb = g[:, 0] - s_B                                   # mm dọc trục tính từ B

    edges = np.arange(np.floor(sb.min() / BAND_BIN_MM) * BAND_BIN_MM,
                      sb.max() + BAND_BIN_MM, BAND_BIN_MM)
    h, _ = np.histogram(sb, bins=edges)
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0
    bands = []
    for a, b in _split_runs(hs):
        sel = (sb >= edges[a]) & (sb < edges[b])
        if sel.sum() < BAND_MIN_SHARE * len(sb):
            continue
        v = sb[sel]
        lo_e, hi_e = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        pk = a + int(np.argmax(hs[a:b]))
        peak = float((edges[pk] + edges[pk + 1]) / 2.0)
        band = {"share": round(float(sel.mean()), 3), "n_points": int(sel.sum()),
                "mm_from_B": {"lower_edge": round(lo_e, 1), "upper_edge": round(hi_e, 1),
                              "peak": round(peak, 1)},
                "mm_from_T": {"lower_edge": round(L - lo_e, 1), "upper_edge": round(L - hi_e, 1),
                              "peak": round(L - peak, 1)},
                "along": {"lower_edge": round(lo_e / L, 3), "upper_edge": round(hi_e / L, 3),
                          "peak": round(peak / L, 3)},
                "azimuth_deg": None, "diameter_at_peak_mm": None}
        if geo["elongated"] and geo["prof"]:
            az = np.degrees(np.arctan2(g[sel, 2], g[sel, 1]))    # 0° = mặt về camera demo
            band["azimuth_deg"] = {"min": round(float(np.percentile(az, 5)), 1),
                                   "max": round(float(np.percentile(az, 95)), 1)}
            ps = np.array([p["s"] for p in geo["prof"]]) - s_B
            pr = np.array([p["r"] for p in geo["prof"]])
            band["diameter_at_peak_mm"] = round(2 * float(np.interp(peak, ps, pr)), 1)
        bands.append(band)
    if not bands:
        return None, None
    bands.sort(key=lambda x: -x["share"])
    for k, bd in enumerate(bands):
        bd["role"] = "main" if k == 0 else "secondary"
    bands.sort(key=lambda x: x["mm_from_B"]["peak"])

    region = {"n_points": int(len(G)), "bands": bands,
              "envelope_along": {"lower": round(float(np.percentile(sb, 5)) / L, 3),
                                 "upper": round(float(np.percentile(sb, 95)) / L, 3),
                                 "note": "khung bao cả vùng đỏ (gồm cả khe) — chỉ tham khảo, KHÔNG phải khoảng kẹp"}}
    if geo["elongated"] and geo["prof"]:
        rr = np.hypot(g[:, 1], g[:, 2])
        ps = np.array([p["s"] for p in geo["prof"]])
        r_exp = np.interp(g[:, 0], ps, np.array([p["r"] for p in geo["prof"]]))
        region["on_surface_frac"] = round(float(np.mean(np.abs(rr - r_exp) < ON_SURFACE_MM)), 3)
    pts_B = np.column_stack([sb, g[:, 1], g[:, 2]]).astype(np.float32)
    return region, pts_B


def radius_profile_from_B(geo, anchors):
    L = max(anchors["_s_T"] - anchors["_s_B"], 1e-6)
    return [{"mm_from_B": round(p["s"] - anchors["_s_B"], 1),
             "along": round((p["s"] - anchors["_s_B"]) / L, 3), "radius_mm": round(p["r"], 1)}
            for p in geo["prof"]]


# ------------------------------------------- phía ÁP DỤNG thẻ (góc nhìn mới)
def measure_view(obj_mask, depth_frame, intr, frame_masks, shape):
    """Đo ở GÓC NHÌN MỚI đúng những gì thẻ đã đo: mặt bàn, trục, B, T, hồ sơ
    bán kính. Trả dict đưa thẳng vào `resolve_grasp`, hoặc None."""
    P = H.backproject_mask_xyz(obj_mask, depth_frame, intr)
    pa = H.principal_axis_3d(P)
    if pa is None:
        return None
    hint = pa["axis"] / np.linalg.norm(pa["axis"])
    plane, _ = fit_table_plane(depth_frame, intr, frame_masks, obj_mask, shape, hint)
    _, geo = object_frame_3d(obj_mask, depth_frame, intr, up=plane["normal"] if plane else None)
    if geo is None:
        return None
    an = compute_anchors(geo, plane, obj_mask, shape, intr)
    e1 = geo["R"][0]
    return {"B": geo["c"] + an["_s_B"] * e1, "T": geo["c"] + an["_s_T"] * e1, "e1": e1,
            "table_valid": an["table_valid"], "top_valid": an["top_valid"],
            "radius_profile": radius_profile_from_B(geo, an) if geo["prof"] else [],
            "warnings": list(an["warnings"])}


def _profile_ratio(card_prof, view_prof):
    """Trung vị tỉ lệ bán kính (góc mới / thẻ) tại CÙNG độ cao mm từ B, trên
    đoạn hai bên cùng có. ~1.0 = cùng vật. None nếu chồng lấn quá ít."""
    if not card_prof or not view_prof:
        return None
    cs = np.array([p["mm_from_B"] for p in card_prof])
    cr = np.array([p["radius_mm"] for p in card_prof])
    vs = np.array([p["mm_from_B"] for p in view_prof])
    vr = np.array([p["radius_mm"] for p in view_prof])
    ok = (vs >= cs.min()) & (vs <= cs.max())
    if ok.sum() < PROFILE_MIN_SLICES:
        return None
    return float(np.median(vr[ok] / np.interp(vs[ok], cs, cr)))


def resolve_grasp(card, view):
    """Chọn NEO cho góc nhìn mới + KIỂM TRA CHÉO B–T -> vị trí mảng nắm.

    Lỗ hổng cần vá: cờ `top_valid` của `compute_anchors` chỉ bắt nắp bị MÉP
    ẢNH cắt hoặc thiếu depth, KHÔNG bắt nắp bị VẬT KHÁC che. Mô phỏng trên
    demo: che nắp 30 px -> neo nắp lệch -16.4 mm mà cờ vẫn "ok". Có mặt bàn
    thì đo lại độ dài B–T và so với thẻ:

      |B–T mới − B–T thẻ| <= BT_TOL  -> hai neo khớp, dùng neo bàn (kiểm chéo ok)
      ngắn hơn rõ                   -> NẮP bị che/mở HOẶC vật NHỎ hơn
      dài hơn rõ                    -> có nắp (thẻ không có) / vật đặt trên khay
                                       (B quá thấp) HOẶC vật LỚN hơn

    "Ngắn/dài hơn" mơ hồ giữa "cùng vật bị che" và "vật khác cỡ" — hai trường
    hợp phải xử lý NGƯỢC nhau (neo mm vs tỉ lệ). Phân xử bằng HỒ SƠ BÁN KÍNH ở
    cùng độ cao từ B (`_profile_ratio`): cùng vật thì ~1.0.

    `view` = dict của `measure_view`. Trả dict: method, status, warnings, và
    toạ độ (hệ camera góc mới) của mép/đỉnh từng mảng."""
    an = card.get("anchors") or {}
    bands = card.get("grasp_bands") or []
    if not bands or not an:
        return {"status": "fail", "method": None, "warnings": ["thẻ không có neo/mảng (thẻ thiếu depth)"]}
    L_card = float(an["length_BT_mm"])
    B, T, e1 = np.asarray(view["B"]), np.asarray(view["T"]), np.asarray(view["e1"])
    L_now = float(np.linalg.norm(T - B))
    tol = max(BT_TOL_MM, BT_TOL_FRAC * L_card)
    warnings = list(view.get("warnings", []))
    table_ok, top_ok = bool(view["table_valid"]), bool(view["top_valid"])
    ratio = _profile_ratio(card.get("object_frame_3d", {}).get("radius_profile"),
                           view.get("radius_profile")) if table_ok else None
    same_size = None if ratio is None else abs(ratio - 1.0) <= RADIUS_TOL_FRAC
    check = {"length_BT_card_mm": round(L_card, 1), "length_BT_view_mm": round(L_now, 1),
             "diff_mm": round(L_now - L_card, 1), "tol_mm": round(tol, 1),
             "radius_ratio": None if ratio is None else round(ratio, 3)}

    method, status = None, "ok"
    if table_ok and top_ok:
        d = L_now - L_card
        if abs(d) <= tol:
            method = "neo bàn"                            # hai neo khớp nhau
            check["result"] = "khớp"
        elif same_size is False:
            method = "tỉ lệ B–T"
            check["result"] = "vật khác cỡ"
            warnings.append(f"B–T lệch {d:+.0f} mm và bán kính lệch x{ratio:.2f} — vật KHÁC CỠ, dùng tỉ lệ")
        elif d < 0:
            method, top_ok = "neo bàn", False
            check["result"] = "nắp ngắn"
            warnings.append(f"B–T ngắn hơn thẻ {-d:.0f} mm" + (" (bán kính khớp)" if same_size else "")
                            + ": nắp bị che hoặc đã mở — bỏ neo nắp, dùng neo bàn")
            if same_size is None:
                warnings.append("không so được bán kính — nếu là vật NHỎ hơn thì neo bàn sai")
                status = "uncertain"
        else:
            method, status = "neo bàn", "conflict"
            check["result"] = "dài hơn"
            warnings.append(f"B–T dài hơn thẻ {d:.0f} mm dù bán kính khớp: có nắp (thẻ không có) "
                            f"thì neo bàn đúng; nếu vật đặt trên khay thì B sai — cần xác nhận")
    elif table_ok:
        method = "neo bàn"
        warnings.append("không kiểm chéo được: đỉnh không tin được")
    elif top_ok:
        method = "neo nắp"
        warnings.append("không kiểm chéo được: không có mặt bàn hợp lệ")
    else:
        return {"status": "fail", "method": None, "check": check,
                "warnings": warnings + ["không có neo nào dùng được"]}
    if status == "ok" and check.get("result") is None:
        status = "unchecked"

    def place(key):
        out = []
        for bd in bands:
            if method == "neo bàn":
                s = {k: bd["mm_from_B"][k] for k in ("lower_edge", "peak", "upper_edge")}
                pts = {k: (B + v * e1) for k, v in s.items()}
            elif method == "neo nắp":
                pts = {k: (T - bd["mm_from_T"][k] * e1) for k in ("lower_edge", "peak", "upper_edge")}
            else:
                pts = {k: (B + bd["along"][k] * L_now * e1) for k in ("lower_edge", "peak", "upper_edge")}
            out.append({"role": bd["role"], **{k: np.round(v, 2).tolist() for k, v in pts.items()}})
        return out

    return {"status": status, "method": method, "check": check, "warnings": warnings,
            "table_valid": table_ok, "top_valid": top_ok, "bands_cam_mm": place(method)}


# ------------------------------------------------------------- DINOv2
_DINO = None


def dino_features(crop_bgr, device=None):
    """Bản đồ đặc trưng patch DINOv2 của ảnh crop -> (h, w, C) float16.
    Trả None (kèm lý do) nếu thiếu transformers/model — thẻ vẫn tạo được."""
    global _DINO
    try:
        import torch
        from transformers import AutoModel
    except ImportError as e:
        return None, f"thiếu thư viện ({e.name}) — chạy bằng venv có transformers"
    if _DINO is None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = AutoModel.from_pretrained(DINO_ID, local_files_only=True).to(dev).eval()
        except Exception as e:                      # noqa: BLE001 — lỗi tải model đủ loại
            return None, f"không tải được {DINO_ID}: {e}"
        _DINO = (model, dev)
    model, dev = _DINO
    h, w = crop_bgr.shape[:2]
    s = DINO_LONG_SIDE / max(h, w)
    nh, nw = max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)
    rgb = cv2.cvtColor(cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None].to(dev)
    with torch.no_grad():
        tok = model(pixel_values=x).last_hidden_state[0, 1:]      # bỏ CLS
    gh, gw = nh // 14, nw // 14
    return tok.reshape(gh, gw, -1).cpu().numpy().astype(np.float16), None


def _grid_frac(mask, gh, gw):
    """Tỉ lệ pixel `mask` trong từng ô lưới gh x gw (ô = 1 patch DINOv2)."""
    return cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------- preview
def render_preview(ref_crop, cur_crop, obj_crop, grasp_crop, freq_crop, info_lines, draw=None):
    def tint(img, m, color, a=0.55):
        o = img.copy()
        o[m] = (o[m] * (1 - a) + np.array(color) * a).astype(np.uint8)
        return o

    p1 = ref_crop.copy()
    cnt, _ = cv2.findContours(obj_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cnt, -1, (0, 255, 0), 1)
    p2 = tint(ref_crop, grasp_crop, (0, 0, 255))
    cnt, _ = cv2.findContours(grasp_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, cnt, -1, (0, 0, 255), 1)
    if draw is not None:
        draw(p2)
    heat = cv2.applyColorMap((np.clip(freq_crop, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    p3 = np.where(obj_crop[..., None], (0.5 * ref_crop + 0.5 * heat).astype(np.uint8), ref_crop // 3)
    p4 = cur_crop.copy()

    titles = ["reference (before touch)", "grasp region", "occlusion frequency", "during grasp"]
    panels = []
    for p, t in zip([p1, p2, p3, p4], titles):
        s = 320 / max(p.shape[:2])
        p = cv2.resize(p, (int(p.shape[1] * s), int(p.shape[0] * s)), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((340, 320, 3), 30, np.uint8)
        canvas[20:20 + p.shape[0], :p.shape[1]] = p
        cv2.putText(canvas, t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(canvas)
    grid = np.hstack(panels)
    foot = np.full((18 * len(info_lines) + 8, grid.shape[1], 3), 30, np.uint8)
    for k, line in enumerate(info_lines):
        cv2.putText(foot, line, (6, 18 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([grid, foot])


def draw_anchors(img, anchors, region3d, frame3d, intr, origin):
    """Vẽ lên crop: trục B->T (vàng), B (xanh lơ), T (tím), vạch mép + đỉnh
    từng mảng (xanh lá = mảng chính, cam = mảng phụ)."""
    ox, oy = origin
    B, T = np.array(anchors["B_cam_mm"]), np.array(anchors["T_cam_mm"])
    e1 = (T - B) / max(np.linalg.norm(T - B), 1e-6)
    e3 = np.array(frame3d["R_obj_in_cam"][2])

    def px(X):
        uv = H.project_xyz_to_pixel(X, intr)
        return None if uv is None else (int(round(uv[0] - ox)), int(round(uv[1] - oy)))

    pb, pt = px(B), px(T)
    if pb and pt:
        cv2.line(img, pb, pt, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, pb, 4, (255, 255, 0), -1)
        cv2.circle(img, pt, 4, (255, 0, 255), -1)
        cv2.putText(img, "B", (pb[0] + 5, pb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(img, "T", (pt[0] + 5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    if not region3d:
        return
    half = 18.0                                              # nửa độ dài vạch, mm
    for bd in region3d["bands"]:
        col = (0, 220, 0) if bd["role"] == "main" else (0, 150, 255)
        for key, w in (("lower_edge", 1), ("upper_edge", 1), ("peak", 2)):
            X = B + bd["mm_from_B"][key] * e1
            a, b = px(X - half * e3), px(X + half * e3)
            if a and b:
                cv2.line(img, a, b, col, w, cv2.LINE_AA)


# ------------------------------------------------------------- chính
def build_card(ep, ctx, outdir, use_dino=True):
    per_frame, hand_ids, fps, shape = ctx["per_frame"], ctx["hand_ids"], ctx["fps"], ctx["shape"]
    depth, intr, reader = ctx["depth"], ctx["intr"], ctx["reader"]
    oid, a, b = ep["oid"], ep["a"], ep["b"]

    r = find_reference_frame(per_frame, hand_ids, oid, a, fps, shape, depth)
    if r is None:
        return None, "không có frame nào trước lúc chạm mà vật chưa bị tay che"
    ref_mask = _resize_mask(per_frame[r][oid], shape)
    ref_bgr = reader.get(r)
    ref_depth = depth[r] if depth is not None and r < len(depth) else None

    hold = hold_window(per_frame, hand_ids, oid, a, b, fps, ref_mask, ref_depth, depth, shape)
    if len(hold) < 3:
        return None, f"vật di chuyển ngay khi chạm (chỉ {len(hold)} frame đứng yên)"

    per_t = []
    for t in hold:
        cur = reader.get(t)
        if cur is None:
            continue
        cd = depth[t] if depth is not None and t < len(depth) else None
        occ, st = occlusion_mask(ref_bgr, ref_mask, ref_depth, cur, per_frame[t], hand_ids, cd, shape)
        per_t.append((t, occ, st, cur))
    # TAY KHÉP DẦN: đo trên demo, vùng bị che tăng từ 12 px (frame chạm đầu)
    # lên ~4000 px lúc nắm chắc. Lấy tần suất trên cả cửa sổ thì chỉ còn chỗ
    # đầu ngón chạm trước. Chỉ gộp các frame ĐÃ NẮM CHẮC.
    peak = max(st["kept"] for _, _, st, _ in per_t) if per_t else 0
    settled = [p for p in per_t if p[2]["kept"] >= SETTLED_FRAC * peak]
    if not settled:
        return None, "không tách được vùng nắm (không frame nào có tay che vật)"
    acc = np.zeros(shape, np.float32)
    for _, occ, _, _ in settled:
        acc += occ
    stats = [st for _, _, st, _ in settled]
    last_bgr = settled[-1][3]
    freq = acc / len(settled)
    grasp = clean_region(freq >= GRASP_FREQ_MIN)
    if not grasp.any():
        return None, "không tách được vùng nắm (không pixel nào bị che đủ thường xuyên)"

    # crop quanh vật
    ys, xs = np.nonzero(ref_mask)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    px, py = int(bw * CROP_PAD_FRAC), int(bh * CROP_PAD_FRAC)
    x0, y0 = max(0, xs.min() - px), max(0, ys.min() - py)
    x1, y1 = min(shape[1], xs.max() + 1 + px), min(shape[0], ys.max() + 1 + py)
    cs = (slice(y0, y1), slice(x0, x1))

    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, "ref_rgb.png"), ref_bgr[cs])
    cv2.imwrite(os.path.join(outdir, "obj_mask.png"), ref_mask[cs].astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(outdir, "grasp_mask.png"), grasp[cs].astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(outdir, "grasp_freq.png"), (np.clip(freq[cs], 0, 1) * 255).astype(np.uint8))
    if ref_depth is not None:
        cv2.imwrite(os.path.join(outdir, "ref_depth.png"), ref_depth[cs].astype(np.uint16))

    region_2d = describe_region_2d(ref_mask, grasp)
    frame3d = plane = anchors = region3d = None
    plane_note = "không có depth"
    if ref_depth is not None and intr is not None:
        # trục PCA thô trước để kiểm mặt bàn (bàn phải gần vuông góc trục vật)
        pa = H.principal_axis_3d(H.backproject_mask_xyz(ref_mask, ref_depth, intr))
        hint = pa["axis"] / np.linalg.norm(pa["axis"]) if pa else None
        plane, plane_note = fit_table_plane(ref_depth, intr, per_frame[r], ref_mask, shape, hint)
        frame3d, geo = object_frame_3d(ref_mask, ref_depth, intr,
                                       up=plane["normal"] if plane else None)
        if geo is not None:
            anchors = compute_anchors(geo, plane, ref_mask, shape, intr)
            frame3d["radius_profile"] = radius_profile_from_B(geo, anchors) if geo["prof"] else []
            region3d, pts_B = grasp_bands(grasp, ref_depth, intr, geo, anchors)
            if pts_B is not None:
                np.savez_compressed(os.path.join(outdir, "grasp_points_obj.npz"), points_obj_mm=pts_B,
                                    columns=np.array(["mm_from_B_along_axis", "e2_toward_camera", "e3"]),
                                    origin=np.array("B (trục vật cắt mặt bàn, hoặc đáy nhìn thấy nếu không có bàn)"))

    dino_note = "tắt (--no-dino)"
    if use_dino:
        feat, err = dino_features(ref_bgr[cs])
        if feat is None:
            dino_note = err
        else:
            gh, gw = feat.shape[:2]
            np.savez_compressed(os.path.join(outdir, "dino_features.npz"), features=feat,
                                obj_frac=_grid_frac(ref_mask[cs], gh, gw).astype(np.float16),
                                grasp_frac=_grid_frac(grasp[cs], gh, gw).astype(np.float16),
                                patch_px=14, model=DINO_ID)
            dino_note = f"{gh}x{gw} patch x {feat.shape[2]} chiều"

    card = {
        "object": ep["label"],
        "episode": {"start": round(a / fps, 2), "end": round(b / fps, 2), "index": ep["index"]},
        "source": {"masks": ctx["masks_json"], "depth": ctx["depth_path"], "video": ctx["video"]},
        "fps": fps,
        "image_size_hw": list(shape),
        "ref_frame": int(r),
        "ref_time": round(r / fps, 2),
        "hold_frames": [int(hold[0]), int(hold[-1])],
        "n_hold_frames": len(per_t),
        "settled_frames": [int(settled[0][0]), int(settled[-1][0])],
        "n_settled_frames": len(settled),
        "crop_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "intrinsics": ({k: round(v, 3) for k, v in intr.items()} if intr else None),
        "grasp_region_px": int(grasp.sum()),
        "grasp_region_frac_of_object": round(float(grasp.sum() / max(1, ref_mask.sum())), 3),
        "table_plane": (None if plane is None else {
            "valid": True,
            "normal_cam": np.round(plane["normal"], 5).tolist(),
            "point_cam_mm": np.round(plane["point"], 2).tolist(),
            "n_inliers": plane["n_inliers"], "n_candidates": plane["n_candidates"],
            "inlier_mm": plane["inlier_mm"],
            "camera_pitch_deg": round(plane["camera_pitch_deg"], 1),
            "axis_tilt_deg": (round(plane["tilt_deg"], 1) if plane["tilt_deg"] is not None else None)}),
        "table_plane_note": plane_note,
        "anchors": ({k: v for k, v in anchors.items() if not k.startswith("_")} if anchors else None),
        "grasp_bands": region3d["bands"] if region3d else None,
        # cho task_planner_ai/execution/world.py: đỉnh MẢNG CHÍNH, tỉ lệ theo B–T
        "along": (next(b["along"]["peak"] for b in region3d["bands"] if b["role"] == "main")
                  if region3d else None),
        "envelope": region3d["envelope_along"] if region3d else None,
        "on_surface_frac": region3d.get("on_surface_frac") if region3d else None,
        "object_frame_3d": frame3d,
        "legacy_2d": region_2d,
        "signals_median_px": {k: int(np.median([s[k] for s in stats])) for k in stats[0]},
        "dino_features": dino_note,
        "files": sorted(os.listdir(outdir)) + ["card.json", "preview.jpg"],
    }

    info = [f"{ep['label']}  ep{ep['index']}  ref f{r}  hold f{hold[0]}-{hold[-1]}  settled f{settled[0][0]}-{settled[-1][0]} ({len(settled)} frames)"]
    if anchors:
        info.append(f"B-T {anchors['length_BT_mm']} mm  bottom: {'table' if anchors['table_valid'] else 'lowest visible'}"
                    f"  top: {'ok' if anchors['top_valid'] else 'NOT RELIABLE'}"
                    + (f"  table tilt {plane['tilt_deg']:.1f} deg" if plane and plane['tilt_deg'] is not None else ""))
    if region3d:
        for bd in region3d["bands"]:
            mb, mt = bd["mm_from_B"], bd["mm_from_T"]
            info.append(f"{bd['role']:>9} {bd['share'] * 100:3.0f}%: from B {mb['lower_edge']:.0f}-{mb['upper_edge']:.0f} "
                        f"(peak {mb['peak']:.0f}) mm | from T {mt['upper_edge']:.0f}-{mt['lower_edge']:.0f} "
                        f"(peak {mt['peak']:.0f}) mm | along {bd['along']['peak']:.2f}"
                        + (f" | diam {bd['diameter_at_peak_mm']} mm" if bd["diameter_at_peak_mm"] else ""))
    if region_2d:
        info.append(f"legacy 2D along {region_2d['along_min']:.2f}-{region_2d['along_max']:.2f} "
                    f"(center {region_2d['along_center']:.2f})")
    draw = None
    if anchors and intr:
        draw = lambda img: draw_anchors(img, anchors, region3d, frame3d, intr, (x0, y0))
    prev = render_preview(ref_bgr[cs], last_bgr[cs], ref_mask[cs], grasp[cs], freq[cs], info, draw)
    cv2.imwrite(os.path.join(outdir, "preview.jpg"), prev)
    card["files"] = sorted(set(card["files"]))
    with open(os.path.join(outdir, "card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return card, None


def list_episodes(per_frame, hand_ids, obj_ids, id2label, depth, intr, object_filter=None):
    eps = []
    for oid in obj_ids:
        lab = id2label.get(oid, "")
        if object_filter and object_filter.lower() not in lab.lower():
            continue
        cflags, _, _ = SP.validated_contacts(per_frame, hand_ids, oid, depth, intr)
        k = 0
        for a, b in S.runs_of(cflags):
            if b - a < MIN_EPISODE_FRAMES:
                continue
            eps.append({"oid": oid, "label": lab, "a": a, "b": b, "index": k})
            k += 1
    eps.sort(key=lambda e: e["a"])
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks_json")
    ap.add_argument("--depth", default=None, help="demo_depth.npz (nên có: dùng cho phiếu depth + 3D)")
    ap.add_argument("--video", default=None, help="video màu; mặc định lấy từ masks json")
    ap.add_argument("--object", default=None, help="chỉ làm vật có nhãn chứa chuỗi này")
    ap.add_argument("--episode", type=int, default=None, help="chỉ làm lần cầm thứ k của vật")
    ap.add_argument("--outdir", default=None, help="mặc định <out>/grasp_cards/")
    ap.add_argument("--no-dino", action="store_true", help="bỏ trích đặc trưng DINOv2")
    args = ap.parse_args()

    data, per_frame = H.load_mask_series(args.masks_json)
    fps = float(data["fps"])
    shape = tuple(int(v) for v in data["size"])
    hand_ids, obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}
    depth, intr = load_depth_aligned(args.depth, shape)
    video = resolve_video(args.masks_json, data, args.video)
    outroot = args.outdir or os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(args.masks_json)), "..", "grasp_cards"))
    stem = os.path.splitext(os.path.basename(video))[0]

    ctx = {"per_frame": per_frame, "hand_ids": hand_ids, "fps": fps, "shape": shape,
           "depth": depth, "intr": intr, "reader": FrameReader(video, shape),
           "masks_json": args.masks_json, "depth_path": args.depth, "video": video}

    eps = list_episodes(per_frame, hand_ids, obj_ids, id2label, depth, intr, args.object)
    if args.episode is not None:
        eps = [e for e in eps if e["index"] == args.episode]
    if not eps:
        print("Không có lần cầm nào khớp điều kiện.")
        return
    print(f"Video : {video}\nDepth : {args.depth or '(không có — bỏ phiếu depth và phần 3D)'}")
    for ep in eps:
        name = f"{stem}_{_slug(ep['label'])}_ep{ep['index']}"
        outdir = os.path.join(outroot, name)
        card, err = build_card(ep, ctx, outdir, use_dino=not args.no_dino)
        head = f"\n[{name}] {ep['label']}  {ep['a'] / fps:.2f}-{ep['b'] / fps:.2f}s"
        if err:
            print(f"{head}\n  BỎ QUA: {err}")
            continue
        print(head)
        print(f"  frame tham chiếu : {card['ref_frame']} ({card['ref_time']}s)")
        print(f"  cửa sổ giữ      : frame {card['hold_frames'][0]}-{card['hold_frames'][1]} "
              f"({card['n_hold_frames']} frame, nắm chắc {card['settled_frames'][0]}-"
              f"{card['settled_frames'][1]})")
        print(f"  vùng nắm        : {card['grasp_region_px']} px "
              f"({card['grasp_region_frac_of_object'] * 100:.1f}% vật)")
        an = card["anchors"]
        if an:
            tp = card["table_plane"]
            print(f"  mặt bàn         : " + (f"{tp['n_inliers']}/{tp['n_candidates']} điểm khớp, camera nhìn xuống "
                                          f"{tp['camera_pitch_deg']}°, trục vật lệch {tp['axis_tilt_deg']}°"
                                          if tp else f"KHÔNG CÓ ({card['table_plane_note']})"))
            print(f"  neo             : đáy = {an['bottom_from']}; nắp {'ổn' if an['top_valid'] else 'KHÔNG tin được'}; "
                  f"B–T = {an['length_BT_mm']} mm (nhìn thấy {an['visible_length_mm']} mm)")
            for w in an["warnings"]:
                print(f"    ! {w}")
        for bd in card["grasp_bands"] or []:
            mb, mt, al = bd["mm_from_B"], bd["mm_from_T"], bd["along"]
            print(f"  mảng {bd['role']:<9} : {bd['share'] * 100:.0f}% điểm | từ B {mb['lower_edge']}–{mb['upper_edge']} "
                  f"(đỉnh {mb['peak']}) mm | từ T {mt['upper_edge']}–{mt['lower_edge']} (đỉnh {mt['peak']}) mm | "
                  f"along {al['lower_edge']}–{al['upper_edge']} (đỉnh {al['peak']})"
                  + (f" | Ø {bd['diameter_at_peak_mm']} mm" if bd["diameter_at_peak_mm"] else ""))
        r2 = card["legacy_2d"]
        if r2:
            print(f"  along 2D (cũ)   : {r2['along_min']}..{r2['along_max']} (tâm {r2['along_center']})")
        print(f"  DINOv2          : {card['dino_features']}")
        print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
