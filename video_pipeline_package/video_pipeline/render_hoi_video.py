"""
render_hoi_video.py

Xuất video có NHÃN SKILL MỨC THẤP, ghép từ 2 nguồn:
  - mask của Grounded-SAM (masks_gsam2.json)  -> vẽ vùng tay/vật + trạng thái chạm
  - skill suy luận từ hình học (skills_lowlevel.json, do hoi_skill_inference.py)
    -> nhãn skill đang diễn ra + dải timeline ở dưới

KHÁC GÌ segment_and_overlay_vi_v3.py:
  Bản v3 hỏi VLM ở từng mốc thưa (1.5 s) rồi vẽ chữ VLM trả lời, và xác thực
  chạm bằng MediaPipe (mất dấu tay khi nắm chặt). Bản này KHÔNG gọi VLM lần
  nào: nhãn đến từ phép đo hình học trên mask ở MỌI frame, nên thấy được cả
  các pha ngắn (Reach 0.20 s, Contact 0.23 s) mà bản v3 không có mốc nào rơi
  vào. Chạy lại cho kết quả y hệt và không tốn request model nào.

Cách dùng:
    python3 render_hoi_video.py video.mp4 \\
        --masks masks_gsam2.json --skills skills_lowlevel.json \\
        --out out_lowlevel.mp4 --scale 2
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import hoi_skill_inference as H

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Màu theo skill — nhóm cùng họ dùng tông gần nhau để nhìn dải timeline đoán
# được đang ở giai đoạn nào mà không cần đọc chữ.
SKILL_COLORS = {
    "Idle":         (70, 74, 80),
    "Reach":        (120, 144, 156),
    "Contact":      (255, 193, 7),
    "Grasp":        (46, 125, 50),
    "Lift":         (2, 136, 209),
    "MoveToTarget": (25, 118, 210),
    "Pour":         (230, 81, 0),
    "Place":        (129, 212, 250),
    "Release":      (165, 214, 167),
    "Retract":      (123, 31, 162),
}
DEFAULT_COLOR = (97, 97, 97)

# Màu mask theo VAI TRÒ (không theo id, vì 1 bàn tay có thể bị tách thành
# nhiều id #2/#3/#4 — tô khác màu trông như có 3 bàn tay). Giá trị dưới đây
# theo thứ tự RGB; khi ghi vào ảnh BGR của OpenCV phải đảo lại (xem draw_masks).
MASK_COLOR_OTHER = (255, 82, 82)
MASK_COLOR_HAND = (66, 165, 245)

# Tên vật thể hiển thị tiếng Việt. Nhãn gốc đến từ --prompt của gsam2_video.py
# (người dùng nhập tiếng Anh), không phải từ VLM, nên dịch bằng bảng tra thay
# vì hỏi model. Vật không có trong bảng thì giữ nguyên tên gốc.
OBJECT_VI = {
    "plastic water bottle": "chai nước",
    "water bottle": "chai nước",
    "bottle": "chai nước",
    "white cup": "cốc",
    "glass cup": "ly thuỷ tinh",
    "cup": "cốc",
    "mug": "cốc",
    "hand": "tay",
}


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def load_skills(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def skill_at(skills, t):
    """Đoạn skill chứa thời điểm t. Duyệt tuyến tính là đủ vì chỉ có ~13 đoạn
    cho cả video (khác bản v3 phải dò theo mốc sample)."""
    for s in skills["segments"]:
        if s["time_sec"] <= t < s["end_sec"]:
            return s
    return skills["segments"][-1] if skills["segments"] else None


def subskill_at(subs, t):
    """(episode, subskill) chứa thời điểm t, hoặc (None, None). Chỉ Grasp và
    Pour mới có sub-skill; các đoạn khác không nằm trong episode nào."""
    if not subs:
        return None, None
    for ep in subs.get("episodes", []):
        if ep["start"] <= t < ep["end"]:
            for s in ep["subskills"]:
                if s["start"] <= t < s["end"]:
                    return ep, s
            return ep, (ep["subskills"][-1] if ep["subskills"] else None)
    return None, None


def subskill_timeline(subs, skills):
    """Dải timeline cho chế độ sub-skill, PHỦ KÍN cả video.

    Chỉ vẽ các episode có sub-skill là KHÔNG đủ: Grasp và Pour chỉ chiếm một
    phần video, nên phần còn lại sẽ trống. Đã đo thật: cách đó để trống
    8.92/11.64 giây (77%) dải timeline — người xem mất hẳn Lift / Place /
    MoveToTarget / Retract. Ở đây lấy các đoạn skill cấp cao làm NỀN, rồi
    khoét các episode ra và thay bằng các đoạn con."""
    # mốc cắt = mọi biên của cả đoạn cấp cao lẫn đoạn con
    cuts = set()
    for s in skills["segments"]:
        cuts.add(s["time_sec"]); cuts.add(s["end_sec"])
    for ep in subs.get("episodes", []):
        for s in ep["subskills"]:
            cuts.add(s["start"]); cuts.add(s["end"])
    cuts = sorted(c for c in cuts if c is not None)

    def sub_at(mid):
        for ep in subs.get("episodes", []):
            for s in ep["subskills"]:
                if s["start"] <= mid < s["end"]:
                    return ep, s
        return None, None

    def high_at(mid):
        for s in skills["segments"]:
            if s["time_sec"] <= mid < s["end_sec"]:
                return s
        return skills["segments"][-1] if skills["segments"] else None

    segs = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 1e-6:
            continue
        mid = (a + b) / 2
        ep, sub = sub_at(mid)
        if ep is not None:
            # trong episode: nhãn là sub-skill, màu theo skill cấp cao cha
            segs.append({"time_sec": a, "end_sec": b,
                         "skill_en": ep["skill_en"], "skill_vi": sub["skill_vi"],
                         "sub": sub["skill"]})
        else:
            hs = high_at(mid)
            if hs is not None:
                segs.append({"time_sec": a, "end_sec": b,
                             "skill_en": hs["skill_en"], "skill_vi": hs["skill_vi"],
                             "sub": None})
    return segs


def draw_masks(bgr, masks, hand_ids):
    """Tô màu bán trong suốt lên vùng tay/vật + vẽ bbox và nhãn."""
    overlay = bgr.copy()
    items = []
    for oid, m in masks.items():
        if not m.any():
            continue
        color = MASK_COLOR_HAND if oid in hand_ids else MASK_COLOR_OTHER
        overlay[m] = color[::-1]        # RGB -> BGR
        ys, xs = np.nonzero(m)
        items.append((oid, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], color))
    bgr = cv2.addWeighted(overlay, 0.40, bgr, 0.60, 0)
    for oid, (x0, y0, x1, y1), color in items:
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color[::-1], 2)
    return bgr


def draw_hud(bgr, seg, obj_label, touched, scale, height_mm=None,
             sub=None, parent=None):
    """Khối chữ góc trên: skill hiện tại + trạng thái chạm + độ cao thật (nếu
    có depth). Độ cao ghi bằng cm vì đó là đơn vị đọc tự nhiên cho thao tác
    bàn tay (nhấc lên 7 cm), mm chỉ dùng trong log."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    f_skill = _font(FONT_BOLD, int(26 * scale))
    f_sub = _font(FONT_BOLD, int(17 * scale))

    skill_vi = seg["skill_vi"] if seg else "-"
    skill_en = seg["skill_en"] if seg else ""
    color = SKILL_COLORS.get(skill_en, DEFAULT_COLOR)

    pad = int(8 * scale)
    y = 0
    n_rows = 2 + (1 if height_mm is not None else 0) + (1 if sub else 0)
    d.rectangle([0, 0, int(bgr.shape[1]), int(18 * scale + 22 * scale * n_rows)], fill=(0, 0, 0))

    # chấm màu + tên skill
    r = int(9 * scale)
    d.ellipse([pad, y + int(11 * scale), pad + 2 * r, y + int(11 * scale) + 2 * r], fill=color)
    d.text((pad + 2 * r + int(8 * scale), y + int(2 * scale)), f"Skill: {skill_vi}",
           font=f_skill, fill=(255, 255, 255))
    y += int(34 * scale)

    if sub:
        d.text((pad, y), f"   ↳ {sub}", font=f_sub, fill=(255, 235, 100))
        y += int(24 * scale)

    if touched:
        d.text((pad, y), f"● đang chạm: {obj_label}", font=f_sub, fill=(0, 235, 120))
    else:
        d.text((pad, y), "○ chưa chạm", font=f_sub, fill=(150, 155, 165))
    y += int(24 * scale)

    if height_mm is not None:
        cm = height_mm / 10.0
        # Xanh khi vật nằm trên bàn, vàng khi đã nhấc khỏi mặt bàn — nhìn màu
        # là biết ngay, không phải đọc số.
        col = (255, 205, 60) if height_mm > 15 else (140, 200, 255)
        d.text((pad, y), f"↑ cao: {cm:5.1f} cm", font=f_sub, fill=col)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_contact_marker(bgr, info, scale, obj_vi):
    """Vẽ ĐIỂM CHẠM trên vật: vòng ngắm + nhãn vị trí.

    Đây là thứ bản suy luận từ mask từng bỏ mất (bản v3 dùng VLM mới có
    `touch_location`). Vẽ SAU khi phóng to ảnh để nét, nên toạ độ phải nhân
    theo cùng hệ số `scale`."""
    if not info:
        return bgr
    # Vẽ bằng điểm TRÊN TRỤC CHÍNH của vật (axis_x/axis_y, từ describe_location
    # trong contact_point.py) khi có, KHÔNG dùng medoid (info["x"]/["y"]).
    # Medoid là pixel THẬT của vùng chạm nên không bao giờ rơi ra ngoài vật,
    # nhưng nó nhảy quanh vành tiếp xúc mỗi khi tay hơi đổi lực nắm — đo được
    # trên đoạn Rót của video demo: medoid nhảy trung bình 9.67 px/frame so
    # với chỉ 2.42 px/frame của điểm trục chính (xem contact_point.py, mục
    # "ĐIỂM ĐỂ VẼ"). axis_x/axis_y vắng mặt khi vật quá nhỏ hoặc quá tròn để
    # có trục chính đáng tin (MIN_OBJECT_PIXELS) — khi đó mới rơi về medoid.
    px = info.get("axis_x", info["x"])
    py = info.get("axis_y", info["y"])
    x, y = int(round(px * scale)), int(round(py * scale))
    h, w = bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return bgr

    C = (255, 0, 200)          # hồng cánh sen — khác hẳn màu mask (xanh/đỏ)
    r = int(9 * scale)
    cv2.circle(bgr, (x, y), r, C, max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.circle(bgr, (x, y), max(1, int(2 * scale)), C, -1, cv2.LINE_AA)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):     # 4 vạch chữ thập
        cv2.line(bgr, (x + dx * r, y + dy * r),
                 (x + dx * int(r * 1.7), y + dy * int(r * 1.7)),
                 C, max(1, int(2 * scale)), cv2.LINE_AA)

    txt = info.get("text", "")
    if txt:
        label = f"{txt} · {obj_vi}" if obj_vi else txt
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        f = _font(FONT_BOLD, int(15 * scale))
        tw = d.textlength(label, font=f)
        bx = min(max(0, x - int(tw // 2) - int(5 * scale)), w - int(tw) - int(10 * scale))
        by = y - int(r * 1.7) - int(22 * scale)
        if by < 0:
            by = y + int(r * 1.7) + int(4 * scale)
        d.rectangle([bx, by, bx + tw + int(10 * scale), by + int(21 * scale)], fill=(0, 0, 0))
        d.text((bx + int(5 * scale), by + int(2 * scale)), label, font=f, fill=(255, 220, 250))
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr


# Màu chấm tâm XYZ + trục — 1 màu/vật, khác hẳn chấm hồng (điểm chạm) và
# mũi tên (hướng tới) để không lẫn 3 loại điểm khác nhau trên cùng khung hình.
_POSE_PALETTE = [(0, 220, 255), (255, 0, 255), (0, 255, 120), (255, 210, 0)]


def _pose_color_for(oid):
    return _POSE_PALETTE[oid % len(_POSE_PALETTE)]


def draw_pose_axis(bgr, obj_mask, depth_frame, intr, scale, label="", color=None):
    """Vẽ POSE 3D của MỘT vật lên video — để KIỂM BẰNG MẮT `object_pose.py`
    tính đúng thật không, thay vì chỉ tin số trong JSON:

      - LUÔN vẽ 1 CHẤM tại `center_mm` (tâm khối cả vật, xem docstring
        `principal_axis_3d`) kèm nhãn tên vật + toạ độ X,Y,Z (mm, hệ camera)
        — vẽ được cho MỌI vật có mask+depth hợp lệ, kể cả vật TRÒN (ly, cốc)
        vì tâm khối không cần trục chính đáng tin.
      - CHỈ KHI vật đủ dài (`elongation >= ELONG_MIN`, trục chính đáng tin)
        mới vẽ THÊM đoạn thẳng xuyên vật (2 đầu mút thật, chiếu từ 3D ra
        pixel bằng `project_xyz_to_pixel`) + nhãn góc nghiêng — dựng sai thì
        đường sẽ lệch hẳn ra ngoài vật, nhìn là biết ngay (đã xảy ra thật lúc
        depth dính outlier, xem docstring `backproject_mask_xyz`).

    Khác `draw_contact_marker` (chấm hồng = điểm TAY ĐANG CHẠM, chỉ 1 vật
    đang thao tác): hàm này vẽ được đồng thời cho NHIỀU vật/frame, không cần
    đang bị chạm."""
    import hoi_skill_inference as H
    C = color or (0, 220, 255)
    pts = H.backproject_mask_xyz(obj_mask, depth_frame, intr)
    pose = H.principal_axis_3d(pts) if pts is not None else None
    if pose is None:
        return bgr
    h, w = bgr.shape[:2]

    # --- chấm tâm XYZ: luôn vẽ, kể cả vật tròn ---
    pc = H.project_xyz_to_pixel(pose["center_mm"], intr)
    if pc is None:
        return bgr
    pc = (int(round(pc[0] * scale)), int(round(pc[1] * scale)))
    if not (0 <= pc[0] < w and 0 <= pc[1] < h):
        return bgr
    r = max(2, int(5 * scale))
    cv2.circle(bgr, pc, r, C, -1, cv2.LINE_AA)
    cv2.circle(bgr, pc, r, (0, 0, 0), max(1, int(1 * scale)), cv2.LINE_AA)

    x, y, z = pose["center_mm"]
    txt = f"{label} ({x:.0f},{y:.0f},{z:.0f})mm" if label else f"({x:.0f},{y:.0f},{z:.0f})mm"
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    f = _font(FONT_REG, int(13 * scale))
    tw = d.textlength(txt, font=f)
    ty = pc[1] + r + int(4 * scale)
    d.rectangle([pc[0] + 6, ty, pc[0] + 6 + tw + 6, ty + int(18 * scale)], fill=(0, 0, 0))
    d.text((pc[0] + 9, ty + int(1 * scale)), txt, font=f, fill=C[::-1])
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # --- đường trục: chỉ khi đủ dài để trục chính đáng tin ---
    if pose["elongation"] < H.ELONG_MIN:
        return bgr
    pa = H.project_xyz_to_pixel(pose["endpoint_a_mm"], intr)
    pb = H.project_xyz_to_pixel(pose["endpoint_b_mm"], intr)
    if pa is None or pb is None:
        return bgr
    pa = (int(round(pa[0] * scale)), int(round(pa[1] * scale)))
    pb = (int(round(pb[0] * scale)), int(round(pb[1] * scale)))
    if not (0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h):
        return bgr

    cv2.line(bgr, pa, pb, C, max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.circle(bgr, pa, max(2, int(4 * scale)), (255, 160, 0), -1, cv2.LINE_AA)  # đầu A: cam
    cv2.circle(bgr, pb, max(2, int(4 * scale)), (0, 0, 255), -1, cv2.LINE_AA)    # đầu B: đỏ

    # Nhãn tilt đặt CẠNH ĐẦU MÚT B (không đặt giữa đoạn) — giữa đoạn thường
    # trùng chỗ với chấm tâm + nhãn XYZ (tâm PCA nằm gần giữa 2 đầu mút),
    # 2 nhãn đè lên nhau đọc không nổi. Cạnh đầu mút gần như luôn trống chỗ.
    tilt_deg = H.axis_tilt_deg(pose["axis"])
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    txt2 = f"tilt {tilt_deg:.0f}°"
    tw2 = d.textlength(txt2, font=f)
    tx, ty = pb[0] + int(8 * scale), pb[1] - int(9 * scale)
    d.rectangle([tx, ty, tx + tw2 + 6, ty + int(18 * scale)], fill=(0, 0, 0))
    d.text((tx + 3, ty + int(1 * scale)), txt2, font=f, fill=C[::-1])
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_approach_arrow(bgr, obj_mask, hand_mask, scale, label):
    """Mũi tên HƯỚNG TAY ĐI TỚI VẬT — chỉ vẽ khi tay CHƯA chạm.

    Đây là thứ duy nhất cần đến tâm tay, và chỉ cần trước khi chạm: sau khi đã
    chạm thì vị trí tương tác neo vào VẬT (điểm chạm) nên tâm tay không còn cần.
    Sai lệch do mask gồm cẳng tay chỉ làm tâm lệch về phía cổ tay, không đổi
    hướng tay-tới-vật."""
    import contact_point as CP
    if obj_mask is None or hand_mask is None:
        return bgr
    if not obj_mask.any() or not hand_mask.any():
        return bgr
    ys, xs = np.nonzero(obj_mask)
    ox, oy = float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(hand_mask)
    hx, hy = float(xs.mean()), float(ys.mean())

    # mũi tên đi từ phía tay VỀ vật, dừng trước tâm vật một đoạn
    vx, vy = ox - hx, oy - hy
    L = float(np.hypot(vx, vy))
    if L < 20 * scale:
        return bgr
    ux, uy = vx / L, vy / L
    start = (int(hx * scale + ux * 20 * scale), int(hy * scale + uy * 20 * scale))
    end = (int(ox * scale - ux * 30 * scale), int(oy * scale - uy * 30 * scale))
    C = (0, 220, 255)
    cv2.arrowedLine(bgr, start, end, C, max(1, int(3 * scale)), cv2.LINE_AA,
                    tipLength=0.10)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    f = _font(FONT_BOLD, int(14 * scale))
    tw = d.textlength(label, font=f)
    d.rectangle([start[0], start[1] - int(21 * scale),
                 start[0] + tw + int(10 * scale), start[1]],
                fill=(0, 0, 0))
    d.text((start[0] + int(5 * scale), start[1] - int(20 * scale)), label,
           font=f, fill=C)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_timeline(segments, duration, cur_time, width, panel_h, scale):
    """Dải timeline vẽ TOÀN BỘ chuỗi skill mức thấp, khối đang phát được tô
    sáng + viền, kèm con trỏ thời gian và trục giây."""
    img = Image.new("RGB", (width, panel_h), (26, 26, 30))
    d = ImageDraw.Draw(img)
    f_t = _font(FONT_BOLD, int(15 * scale))
    f_s = _font(FONT_REG, int(12 * scale))

    x_pad = int(10 * scale)
    bar_w = width - 2 * x_pad
    y_bar = int(10 * scale)
    bar_h = int(34 * scale)

    def x_of(t):
        return x_pad + int(bar_w * max(0.0, min(t, duration)) / duration) if duration > 0 else x_pad

    d.rectangle([x_pad, y_bar, x_pad + bar_w, y_bar + bar_h], fill=(52, 52, 58))
    for s in segments:
        x1, x2 = x_of(s["time_sec"]), x_of(s["end_sec"])
        x2 = max(x2, x1 + 1)
        playing = s["time_sec"] <= cur_time < s["end_sec"]
        base = SKILL_COLORS.get(s["skill_en"], DEFAULT_COLOR)
        fill = tuple(min(255, int(c * 1.5) + 35) for c in base) if playing else base
        d.rectangle([x1, y_bar, x2, y_bar + bar_h], fill=fill)
        lab = s["skill_vi"]
        if x2 - x1 >= d.textlength(lab, font=f_t) + int(8 * scale):
            d.text(((x1 + x2) // 2, y_bar + bar_h // 2), lab, font=f_t,
                   fill=(20, 20, 20), anchor="mm")
        if playing:
            d.rectangle([x1, y_bar, x2, y_bar + bar_h], outline=(255, 255, 255),
                        width=max(1, int(2 * scale)))

    xc = x_of(cur_time)
    d.line([xc, y_bar - int(3 * scale), xc, y_bar + bar_h + int(3 * scale)],
           fill=(255, 235, 59), width=max(1, int(2 * scale)))

    y_tick = y_bar + bar_h + int(5 * scale)
    d.text((width - x_pad, y_bar + bar_h + int(4 * scale)), f"{cur_time:.1f}s / {duration:.1f}s",
           font=f_s, fill=(160, 165, 175), anchor="ra")
    for t in range(0, int(duration) + 1, 2):
        d.text((x_of(t), y_tick), f"{t}s", font=f_s, fill=(150, 155, 165), anchor="ma")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render(video_path, masks_json, skills_json, out_path, scale=2.0, show_masks=True,
           depth_path=None, subskills_json=None, dump_contacts_path=None,
           show_pose_axis=True):
    data, per_frame = H.load_mask_series(masks_json)
    hand_ids, _obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}
    skills = load_skills(skills_json)

    # Độ cao thật: cần cả stack depth lẫn mốc quy chiếu (mặt bàn) mà bước suy
    # luận đã tính và ghi vào file skills.
    depth, d_ref = None, skills.get("depth_reference_mm")
    if depth_path and d_ref is not None:
        depth, _ = H.load_depth(depth_path)
    elif depth_path:
        print("  CẢNH BÁO: skills json không có depth_reference_mm — chạy lại "
              "hoi_skill_inference.py với --depth để có mốc mặt bàn")
    intr = H.load_intrinsics(depth_path) if depth_path else None
    if depth_path and intr is None:
        print("  CẢNH BÁO: file depth không có nội tham số camera (fx/fy/cx/cy) "
              "— chạy lại depth_from_mcap.py bản mới để có toạ độ X,Y,Z mm")

    # Chế độ sub-skill: dải timeline vẽ các đoạn CON (Approach/Align/...), tô
    # màu theo skill cấp cao cha.
    subs = load_skills(subskills_json) if subskills_json else None
    tl_segments = subskill_timeline(subs, skills) if subs else skills["segments"]
    if subs and not tl_segments:
        print("  CẢNH BÁO: subskills json không có episode nào — dải timeline "
              "quay về dùng skill cấp cao")
        tl_segments = skills["segments"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_h = int(round(74 * scale))
    out_w = int(round(w * scale))
    out_h = int(round(h * scale)) + panel_h
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter cho: {out_path}")

    # Độ chồng lấn mask tay-vật theo từng frame, để hiển thị trạng thái chạm.
    # Tính lại từ mask đã giải (không đọc từ signals) cho khớp đúng ngưỡng mà
    # bước suy luận skill đã dùng.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                 (2 * H.CONTACT_DILATE_PX + 1,) * 2)

    # Trạng thái "đang chạm" phải dùng CÙNG thước đo với bước suy luận sub-skill,
    # nếu không video sẽ tự mâu thuẫn: ở pha Align tay còn cách vật 4-12 cm
    # nhưng mask 2D chiếu vuông góc xuống trùng chỗ chai nên báo "đã chạm".
    # Có depth thì dùng khoảng cách theo chiều sâu (gap) cho nhất quán.
    gap_by_obj = {}
    if depth is not None:
        import subskill_inference as S
        for o in data["objects"]:
            gap_by_obj[o["id"]] = S.compute_gap(per_frame, depth, hand_ids, o["id"])

    # Log tọa độ TỪNG FRAME của chấm hồng (điểm chạm đã vẽ) — bản thân video
    # không lưu lại số nào, chỉ vẽ rồi mất, nên muốn TRA LẠI tọa độ ở đúng
    # giây nào phải bật cờ này. Toạ độ ghi theo hệ khung hình GỐC (chưa nhân
    # `scale`) để khớp với `grasp_point_px` bên skill_params.py.
    contact_log = [] if dump_contacts_path else None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        masks = per_frame[idx] if idx < len(per_frame) else {}
        # Mask có thể nhỏ hơn video: gsam2_video.py có cờ --max-side thu nhỏ
        # frame trước khi detect, nên mask theo hệ toạ độ đã thu nhỏ. Kéo mask
        # về đúng cỡ frame gốc trước khi vẽ (INTER_NEAREST để giữ mask nhị phân).
        if masks:
            any_m = next(iter(masks.values()))
            if any_m.shape[:2] != (h, w):
                masks = {k: cv2.resize(m.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                         for k, m in masks.items()}

        touched, obj_label, height_mm, contact = False, "", None, None
        if masks:
            hand_mask = None
            for hid in hand_ids:
                if hid in masks:
                    hand_mask = masks[hid] if hand_mask is None else (hand_mask | masks[hid])
            seg = skill_at(skills, t)
            active_oid = seg.get("object_id") if seg else None
            # ĐIỂM CHẠM trên vật — tính trên hệ toạ độ mask (chưa phóng to)
            if hand_mask is not None and active_oid in masks:
                import contact_point as CP
                df = None
                odm = None
                if depth is not None and idx < len(depth):
                    df = depth[idx]
                    m = masks[active_oid]
                    if m.shape[:2] != df.shape[:2]:
                        m = cv2.resize(m.astype(np.uint8), (df.shape[1], df.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    odm = H._median_depth_in_mask(df, m)
                contact = CP.contact_info(masks[active_oid], hand_mask, df, odm)
            if hand_mask is not None and active_oid in masks:
                ov = int(np.count_nonzero(cv2.dilate(hand_mask.astype(np.uint8), k) & masks[active_oid]))
                touched = ov >= H.CONTACT_MIN_OVERLAP_PX
                # Có depth thì CHỈ tin gap, không rơi về mask 2D nữa: gap = NaN
                # nghĩa là KHÔNG có pixel tay nào nằm sát vật, tức là CHƯA chạm
                # — nhưng mask 2D nới rộng vẫn chồng lấn khi tay ở TRÊN CAO (bàn
                # tay chiếu vuông góc xuống trùng chỗ vật), nên rơi về 2D sẽ báo
                # "đang chạm" đúng lúc tay còn cách cả gang tay. Đã thấy thật
                # trên video DexYCB ở pha Approach.
                g = gap_by_obj.get(active_oid)
                if g is not None and idx < len(g):
                    import subskill_inference as S
                    gi = g[idx]
                    touched = bool((not np.isnan(gi)) and gi <= S.GAP_ALIGN_MM)
                raw = id2label.get(active_oid, "")
                obj_label = OBJECT_VI.get(raw.lower(), raw)
            if depth is not None and active_oid in masks and idx < len(depth):
                df = depth[idx]
                m = masks[active_oid]
                if m.shape[:2] != df.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (df.shape[1], df.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                d_mm = H._median_depth_in_mask(df, m)
                if not np.isnan(d_mm):
                    height_mm = max(0.0, d_ref - d_mm)

        # Vẽ mask ở ĐỘ PHÂN GIẢI GỐC rồi mới phóng to: mask được lưu đúng
        # kích thước frame gốc, phóng to trước sẽ lệch chỉ số.
        if show_masks and masks:
            frame = draw_masks(frame, masks, hand_ids)
        big = cv2.resize(frame, (out_w, int(round(h * scale))), interpolation=cv2.INTER_CUBIC)

        # POSE 3D (chấm XYZ + trục nếu đủ dài) cho MỌI VẬT có mặt trong khung
        # hình — không chỉ vật đang thao tác, và BẤT KỂ đã chạm hay chưa (pose
        # không phụ thuộc trạng thái chạm, khác chấm hồng/mũi tên). Vẽ TRƯỚC
        # chấm chạm/mũi tên để 2 thứ đó (đang dùng cho HUD) nổi rõ hơn.
        if show_pose_axis and intr is not None and depth is not None and idx < len(depth):
            df_now = depth[idx]
            for oid, om in masks.items():
                if oid in hand_ids or not om.any():
                    continue
                raw = id2label.get(oid, "")
                lbl = OBJECT_VI.get(raw.lower(), raw)
                big = draw_pose_axis(big, om, df_now, intr, scale, label=lbl,
                                     color=_pose_color_for(oid))

        # Vẽ điểm chạm SAU khi phóng to cho nét (toạ độ đã nhân theo `scale`).
        # CHỈ vẽ khi ĐÃ XÁC NHẬN CHẠM: `find_contact` dựa trên mask 2D nới rộng
        # nên vẫn ra "điểm chạm" khi tay ở TRÊN CAO (bàn tay chiếu vuông góc
        # xuống trùng chỗ vật). Trạng thái `touched` mới là thứ đã kiểm chứng
        # bằng độ sâu, nên dùng nó làm cổng.
        # Đã chạm -> vòng ngắm ĐIỂM CHẠM. Chưa chạm -> mũi tên HƯỚNG TỚI.
        # Hai thứ này không bao giờ hiện cùng lúc, đúng như thiết kế: trước khi
        # chạm thì cần biết tay tới từ đâu, sau khi chạm thì cần biết nắm chỗ nào.
        draw_arrow = False
        if contact and touched:
            big = draw_contact_marker(big, contact, scale, obj_label)
            if contact_log is not None:
                # CÙNG logic chọn điểm với draw_contact_marker (axis_x/axis_y,
                # rơi về medoid nếu vật không có trục chính đáng tin) — nếu
                # không thì log ghi một điểm, video vẽ một điểm khác.
                entry = {
                    "frame": idx, "t": round(t, 3), "object": obj_label,
                    "x_px": round(contact.get("axis_x", contact["x"]), 1),
                    "y_px": round(contact.get("axis_y", contact["y"]), 1),
                    "along": contact.get("along"), "part": contact.get("label"),
                    "side": contact.get("side", ""),
                }
                if depth is not None and intr is not None and idx < len(depth):
                    import contact_point as CP
                    xyz = CP.contact_point_xyz_mm(contact, masks[active_oid].shape,
                                                  depth[idx], intr)
                    if xyz:
                        entry["x_mm"], entry["y_mm"], entry["z_mm"] = (
                            xyz["x_mm"], xyz["y_mm"], xyz["z_mm"])
                contact_log.append(entry)
        elif hand_mask is not None and active_oid in masks:
            import contact_point as CP
            _d, _lb = CP.approach_direction(masks[active_oid], hand_mask)
            if _lb:
                big = draw_approach_arrow(big, masks[active_oid], hand_mask, scale,
                                          f"tới: {_lb}")
                draw_arrow = True
        # Khi đang trong một episode có sub-skill (Grasp/Pour), HUD hiện SKILL
        # CẤP CAO của episode đó + dòng sub-skill bên dưới; ngoài episode thì
        # hiện như thường.
        ep, sub = subskill_at(subs, t)
        if ep is not None and sub is not None:
            seg_hud = {"skill_vi": ep["skill_vi"], "skill_en": ep["skill_en"]}
            sub_txt = f"{sub['skill']} ({sub['skill_vi']})"
            # Chỉ báo "đang chạm" lấy CÙNG nguồn với nhãn sub-skill, không đọc
            # gap thô từng frame. Hai bên lệch nhau vì nhãn đã qua bước gộp
            # đoạn ngắn còn HUD đọc giá trị thô: đo được thật trên DexYCB,
            # gap ở frame 42 bằng 11 mm (dưới ngưỡng 40) nhưng nằm giữa một
            # đoạn Approach — nhãn gộp thành Approach, HUD lại báo "đang chạm".
            #
            # LUẬT KHÁC NHAU CHO TỪNG LOẠI EPISODE — không dùng chung một luật:
            #   Grasp: chỉ Close_Gripper mới là đã tiếp xúc (Approach/Align thì chưa)
            #   Pour : RÓT LUÔN LÀ ĐANG CẦM. Nếu áp luật của Grasp cho Pour thì
            #          Orient_Tilt bị ghi "chưa chạm" — vô lý, vì rót mà không
            #          cầm thì không rót được. Đây là lỗi có thật đã thấy trên
            #          video demo: gap lúc rót chỉ −23…35 mm (đều dưới ngưỡng
            #          40 mm) nhưng HUD vẫn báo "chưa chạm".
            if ep["skill_en"] == "Pour":
                touched = True
            else:
                touched = (sub["skill"] == "Close_Gripper")
        else:
            seg_hud, sub_txt = skill_at(skills, t), None
        big = draw_hud(big, seg_hud, obj_label, touched, scale,
                       height_mm=height_mm, sub=sub_txt)
        panel = draw_timeline(tl_segments, duration, t, out_w, panel_h, scale)
        writer.write(np.vstack([big, panel]))
        idx += 1

    cap.release()
    writer.release()
    print(f"  {total} frame  {fps:.2f} fps  {duration:.2f}s  ->  {out_w}x{out_h}")
    print(f"  {len(skills['segments'])} đoạn skill")
    print(f"  Đã ghi: {out_path}")

    if dump_contacts_path:
        with open(dump_contacts_path, "w", encoding="utf-8") as f:
            json.dump({"source_video": video_path, "fps": fps,
                       "note": ("Toạ độ chấm hồng (điểm chạm) TỪNG FRAME, hệ "
                                "khung hình gốc (chưa nhân --scale). Chỉ có "
                                "frame nào video thực sự vẽ chấm hồng (đã "
                                "xác nhận touched=True)."),
                       "contacts": contact_log}, f, ensure_ascii=False, indent=2)
        print(f"  Đã ghi tọa độ điểm chạm từng frame: {dump_contacts_path} "
              f"({len(contact_log)} frame)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--masks", required=True, help="masks_gsam2.json")
    ap.add_argument("--skills", required=True, help="skills_lowlevel.json")
    ap.add_argument("--out", default="out_lowlevel.mp4")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--no-mask", action="store_true", help="không tô màu vùng mask")
    ap.add_argument("--depth", default=None,
                    help="file .npz của depth_from_mcap.py — hiện độ cao thật (cm)")
    ap.add_argument("--subskills", default=None,
                    help="subskills.json của subskill_inference.py — hiện thêm "
                         "dòng sub-skill và vẽ dải timeline theo sub-skill")
    ap.add_argument("--dump-contacts", default=None,
                    help="ghi tọa độ chấm hồng (điểm chạm) TỪNG FRAME ra file "
                         "JSON này — video chỉ VẼ rồi mất, không tự lưu số")
    ap.add_argument("--no-pose-axis", action="store_true",
                    help="tắt vẽ hệ trục 3D (PCA 3D, object_pose.py) của vật "
                         "đang thao tác — cần --depth có nội tham số camera")
    args = ap.parse_args()

    render(args.video, args.masks, args.skills, args.out, args.scale,
           show_masks=not args.no_mask, depth_path=args.depth,
           subskills_json=args.subskills, dump_contacts_path=args.dump_contacts,
           show_pose_axis=not args.no_pose_axis)
    print("\nNếu không phát được, convert bằng ffmpeg:")
    print(f"  ffmpeg -i {args.out} -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p "
          f"{os.path.splitext(args.out)[0]}_h264.mp4")


if __name__ == "__main__":
    main()
