"""
subskill_inference.py

Xác định SUB-SKILL đang diễn ra — bước con bên trong một skill cấp cao:

    Grasp  ->  { Approach, Align, Close_Gripper }
    Pour   ->  { Orient_Tilt, Hold_Pour_Angle, Return_Upright }

--------------------------------------------------------------------------------
ĐỊNH NGHĨA NÀO ĐO ĐƯỢC, ĐỊNH NGHĨA NÀO KHÔNG — đọc trước khi sửa ngưỡng
--------------------------------------------------------------------------------
Định nghĩa yêu cầu có 3 tiêu chí dựa trên TRẠNG THÁI NGÓN TAY: "ngón còn mở"
(Approach), "CHƯA khép ngón" (Align), "ngón đang khép lại" (Close_Gripper).

Các tiêu chí đó KHÔNG đo được trên dữ liệu này. Đã đo bằng MediaPipe
HandLandmarker (xem hand_pose_masked.py): với bàn tay ĐEO GĂNG đang nắm chai,
model phát hiện "có tay" ở 97% số frame nhưng dựng skeleton NGÓN DUỖI THẲNG
lên đúng vị trí bàn tay đang nắm — độ cuộn ngón `curl` ~0.05 và KHÔNG ĐỔI suốt
video, tức không phân biệt được lúc mở và lúc khép. Tay trần chỉ phát hiện được
53% số frame. Nên không thể code đúng nguyên văn 3 tiêu chí đó.

Thay bằng ĐẠI LƯỢNG VẬT LÝ đo được — khoảng cách THẬT giữa tay và vật theo
chiều sâu (depth), đo trên các pixel tay nằm SÁT vật trong ảnh:

    gap_mm = trung vị depth(pixel tay sát vật) - trung vị depth(vật)

    gap = NaN (không có pixel tay nào sát vật)  ->  Approach
    gap >  GAP_ALIGN_MM (tay ở trên cao)        ->  Align
    gap <= GAP_ALIGN_MM (tay ngang tầm vật)     ->  Close_Gripper

VÌ SAO PHẢI DÙNG DEPTH Ở ĐÂY — đây là điểm quan trọng nhất của file:
  Cách cũ (segment_and_overlay_vi_v3.py, và bản mask 2D trước đó) xác định
  "đã chạm" bằng mask tay nới rộng chồng lấn mask vật trong ẢNH 2D. Cách đó
  SAI khi tay ở TRÊN CAO vật: bàn tay chiếu vuông góc xuống trùng vị trí chai
  nên 2D báo "chạm" trong khi thực tế còn cách cả 12 cm. Đo được thật trên
  video demo: t=1.01-1.31 s, mask 2D báo CHẠM nhưng gap = 96-121 mm — tay vẫn
  đang lơ lửng trên chai. Depth phân biệt được vì tay ở trên cao thì GẦN
  camera hơn (depth nhỏ hơn), còn khi chạm thì cùng độ sâu với vật.
  Dùng depth còn tránh được luôn cái bẫy cẳng tay: chỉ xét các pixel tay SÁT
  vật trong ảnh, nên cẳng tay ở xa không kéo lệch kết quả.

Cách dùng:
    python3 subskill_inference.py masks_gsam2.json --depth demo_depth.npz \
        --out subskills.json
"""

import argparse
import json

import cv2
import numpy as np

import hoi_skill_inference as H

# --- ngưỡng cho tách Approach / Align / Close_Gripper bằng độ sâu ---
GAP_ALIGN_MM = 40.0     # gap lớn hơn -> tay còn ở trên cao, chưa tới vật
GAP_CONTACT_MM = 35.0   # gap nhỏ hơn -> tay đã ngang tầm vật (coi như tới)
GAP_DILATE_PX = 3       # nới mask vật bao nhiêu px để tìm pixel tay "sát vật"
GAP_MIN_PIXELS = 10     # số pixel tối thiểu mỗi bên để tính trung vị

# --- ngưỡng cho tách Orient_Tilt / Hold_Pour_Angle / Return_Upright ---
TILT_MIN_DEG = 20.0     # lệch khỏi tư thế thẳng bao nhiêu thì coi là đang nghiêng
TILT_RATE_DEG_S = 15.0  # tốc độ đổi góc để tách "đang xoay" khỏi "giữ nguyên"
UPRIGHT_START_DEG = 15.0  # episode rót phải BẮT ĐẦU từ tư thế gần thẳng

GRASP_SUBS = ["Approach", "Align", "Close_Gripper"]
POUR_SUBS = ["Orient_Tilt", "Hold_Pour_Angle", "Return_Upright"]

SUB_VI = {
    "Approach": "Tiến tới", "Align": "Chỉnh hướng", "Close_Gripper": "Khép ngón",
    "Orient_Tilt": "Nghiêng để rót", "Hold_Pour_Angle": "Giữ góc rót",
    "Return_Upright": "Dựng thẳng lại",
}


# --------------------------------------------------------- tín hiệu độ sâu
def compute_gap(per_frame, depth, hand_ids, obj_id):
    """Khoảng cách theo CHIỀU SÂU giữa tay và vật, đo trên các pixel tay nằm
    sát vật trong ảnh. Trả mảng (N,) mm, NaN ở frame không có pixel tay nào
    sát vật (tay còn xa)."""
    n = len(per_frame)
    gap = np.full(n, np.nan)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * GAP_DILATE_PX + 1,) * 2)
    for i in range(min(n, len(depth))):
        masks = per_frame[i]
        om = masks.get(obj_id)
        if om is None or not om.any():
            continue
        hm = None
        for h in hand_ids:
            if h in masks:
                hm = masks[h] if hm is None else (hm | masks[h])
        if hm is None:
            continue
        df = depth[i]
        # mask và depth có thể khác cỡ (--max-side của gsam2_video.py)
        if om.shape[:2] != df.shape[:2]:
            om = cv2.resize(om.astype(np.uint8), (df.shape[1], df.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        if hm.shape[:2] != df.shape[:2]:
            hm = cv2.resize(hm.astype(np.uint8), (df.shape[1], df.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        near = (cv2.dilate(om.astype(np.uint8), k) > 0) & hm
        d_obj = df[om]; d_obj = d_obj[d_obj > 0]
        d_near = df[near]; d_near = d_near[d_near > 0]
        if d_obj.size >= GAP_MIN_PIXELS and d_near.size >= GAP_MIN_PIXELS:
            gap[i] = float(np.median(d_near) - np.median(d_obj))
    return gap


def object_motion(per_obj, oid, fps):
    """Tín hiệu CHUYỂN ĐỘNG riêng của MỘT vật thể: (vcx, vcy, hold, speed).

    Phải tính riêng từng vật chứ không dùng chung sig của "vật đang thao tác":
    khi tay đang cầm chai, cái cốc đứng yên bên cạnh không hề được chạm tới,
    nhưng nếu dùng chung tín hiệu thì cốc cũng bị gán là đang được thao tác."""
    series = per_obj[oid]
    n = len(series)
    cx = np.full(n, np.nan); cy = np.full(n, np.nan)
    hx = np.full(n, np.nan); hy = np.full(n, np.nan)
    for i, f in enumerate(series):
        if f is None:
            continue
        cx[i], cy[i] = f["cx"], f["cy"]
        if f["hcx"] is not None:
            hx[i], hy[i] = f["hcx"], f["hcy"]
    s = lambda a: H._smooth(a, H.SMOOTH_WINDOW)
    vcx = H._grad(s(cx), fps, H.VEL_WINDOW)
    vcy = H._grad(s(cy), fps, H.VEL_WINDOW)
    vhx = H._grad(s(hx), fps, H.VEL_WINDOW)
    vhy = H._grad(s(hy), fps, H.VEL_WINDOW)
    sp_h, sp_o = np.hypot(vhx, vhy), np.hypot(vcx, vcy)
    hold = np.where((sp_h > H.HOLD_MIN_SPEED) & (sp_o > H.HOLD_MIN_SPEED),
                    (vhx * vcx + vhy * vcy) / np.maximum(sp_h * sp_o, 1e-9), np.nan)
    return vcx, vcy, hold, sp_o


def compute_dev(ang, overlap):
    """Độ lệch góc trục chính khỏi TƯ THẾ THẲNG (độ, luôn >= 0). Tư thế thẳng
    lấy từ chính các frame KHÔNG chạm tay (lúc đó vật đang đứng yên trên bàn ở
    đầu video) — tự hiệu chuẩn, không cần biết trước vật nằm thế nào.

    Góc trục chính là một ĐƯỜNG không có hướng, xác định trong [0,180), nên
    hiệu phải lấy modulo 180 (lệch 5° và lệch 175° là gần như nhau)."""
    rest = ang[overlap < H.CONTACT_MIN_OVERLAP_PX]
    rest = rest[~np.isnan(rest)]
    ref = float(np.median(rest)) if rest.size else np.nan
    dev = np.abs(((ang - ref + 90.0) % 180.0) - 90.0) if not np.isnan(ref) else ang * np.nan
    return dev, ref


# ------------------------------------------------------------- episode
def runs_of(mask):
    """Các đoạn True liên tục -> [(start, end)] (end loại trừ)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def fill_short_gaps(mask, max_gap):
    """Gộp 2 đoạn True bị ngăn bởi khe hở ngắn hơn max_gap frame — mask rung
    vài frame không nên cắt episode thành nhiều mảnh."""
    m = np.asarray(mask, bool).copy()
    for s, e in runs_of(~m):
        if s > 0 and e < len(m) and (e - s) <= max_gap:
            m[s:e] = True
    return m


def find_grasp_episodes(contact, gap, vcdist, hold, sp, fps):
    """Episode GRASP = từ lúc bắt đầu tiến tới vật cho tới khi CẦM XONG (vật
    bắt đầu di chuyển cùng tay). Bên trong chia 3 sub-skill theo độ sâu.

    Mọi tín hiệu truyền vào đều là CỦA RIÊNG vật thể đang xét."""
    n = len(contact)

    # thời điểm "cầm xong": vật bắt đầu đi cùng tay
    grasped = np.zeros(n, bool)
    for i in range(n):
        if (not np.isnan(hold[i]) and hold[i] >= H.HOLD_COS
                and not np.isnan(sp[i]) and sp[i] >= H.PUSH_SPEED):
            grasped[i] = True
    grasped = fill_short_gaps(grasped, int(0.2 * fps))

    episodes = []
    for cs, ce in runs_of(contact):
        # điểm kết thúc: frame "cầm xong" đầu tiên trong/kèm sau đoạn chạm này
        end = None
        for i in range(cs, min(n, ce + int(1.5 * fps))):
            if grasped[i]:
                end = i
                break
        if end is None:
            # `ce` là mốc LOẠI TRỪ (runs_of trả [start, end)), nên phải lùi 1
            # rồi mới cộng 1 — nếu không, đoạn cuối video sẽ có mốc kết thúc
            # vượt quá số frame và gây IndexError khi tra mảng tín hiệu.
            end = ce - 1

        # điểm bắt đầu: lùi về trước cho tới khi tay THÔI tiến lại gần vật.
        # Dùng vcdist (đạo hàm khoảng cách tâm tay-vật): âm = đang tiến lại.
        start = cs
        i = cs - 1
        while i >= 0 and not np.isnan(vcdist[i]) and vcdist[i] <= 0:
            start = i
            i -= 1
        if start >= cs:
            start = max(0, cs - int(0.3 * fps))
        episodes.append((start, min(end + 1, n)))

    # gộp episode chồng nhau / nối liền
    merged = []
    for s, e in sorted(episodes):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def split_grasp_subskills(start, end, gap, fps):
    """Chia [start, end) thành Approach / Align / Close_Gripper.

    Approach      : chưa có pixel tay nào sát vật (gap = NaN)
    Align         : có pixel sát vật nhưng còn ở trên cao (gap > GAP_ALIGN_MM)
    Close_Gripper : tay đã ngang tầm vật (gap <= GAP_ALIGN_MM) — gộp cả pha
                    khép ngón và pha đã khép, vì không đo được ranh giới giữa
                    hai pha đó khi không có trạng thái ngón tay."""
    lab = np.array(["Approach"] * (end - start), dtype=object)
    for i in range(start, end):
        g = gap[i]
        if np.isnan(g):
            lab[i - start] = "Approach"
        elif g > GAP_ALIGN_MM:
            lab[i - start] = "Align"
        else:
            lab[i - start] = "Close_Gripper"

    lab = _merge_short(lab, max(1, int(0.12 * fps)), order=GRASP_SUBS)
    # ÉP THỨ TỰ TĂNG DẦN: Approach -> Align -> Close_Gripper. Đây là 3 pha của
    # MỘT hành động cầm nắm nên không thể quay lui — đo được thật trên DexYCB:
    # sau Close_Gripper lại có một đoạn Align, do khe hở tay-vật nới ra lúc
    # nhấc vật lên. Bàn tay không "chỉnh hướng lại" sau khi đã nắm xong.
    rank = {name: i for i, name in enumerate(GRASP_SUBS)}
    top = -1
    for i in range(len(lab)):
        r = rank.get(lab[i])
        if r is None:
            continue
        if r < top:
            lab[i] = GRASP_SUBS[top]
        else:
            top = r
    out, i = [], 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        out.append((start + i, start + j, lab[i]))
        i = j
    return out


def _merge_short(lab, min_frames, order):
    """Gộp đoạn ngắn hơn min_frames vào đoạn liền kề, ưu tiên đoạn dài hơn.
    Giữ nguyên thứ tự tên trong `order` khi hoà."""
    if len(lab) == 0:
        return lab
    out = list(lab)
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
        for k, (s, e, name) in enumerate(runs):
            if e - s >= min_frames:
                continue
            prev_n = runs[k - 1][2] if k > 0 else None
            next_n = runs[k + 1][2] if k + 1 < len(runs) else None
            if prev_n is None and next_n is None:
                continue
            if prev_n is None:
                target = next_n
            elif next_n is None:
                target = prev_n
            else:
                pl, nl = runs[k - 1][1] - runs[k - 1][0], runs[k + 1][1] - runs[k + 1][0]
                target = prev_n if pl >= nl else next_n
            for m in range(s, e):
                out[m] = target
            changed = True
            break
    return np.array(out, dtype=object)


def find_pour_episodes(sig, dev, contact, held, fps):
    """Episode POUR = khoảng vật đang ĐƯỢC TAY GIỮ và NGHIÊNG khỏi tư thế nghỉ.
    Bên trong chia 3 sub-skill theo dấu tốc độ xoay.

    ĐIỀU KIỆN ĐỂ VÀO EPISODE (chỉ xét ở frame BẮT ĐẦU, không xét từng frame):
      1. đang CHẠM tay
      2. trước đó vật vừa ở tư thế nghỉ (dev < UPRIGHT_START_DEG)
      3. lúc bắt đầu nghiêng thì vật KHÔNG đang được nhấc lên nhanh

    BA ĐIỀU KIỆN NÀY ĐÃ SAI Ở BẢN TRƯỚC — ghi lại để không lặp lại:

    (a) Bản trước đòi `held` (độ cao > 15 mm) cho MỌI frame. Sai: người rót có
        thể giữ nguyên góc rót trong khi bình còn chạm/nằm sát mặt bàn — đo được
        trên video demo, từ 5.2 s đến 6.7 s bình giữ nguyên tư thế nằm ngang mà
        độ cao ≈ 0. Đòi `held` khiến pha GIỮ GÓC RÓT bị cắt mất hoàn toàn.
        Thay bằng `contact` — tay vẫn trên bình suốt thời gian đó.

    (b) Bản trước áp `vheight < 30 mm/s` cho MỌI frame. Sai: pha DỰNG THẲNG LẠI
        thường đi kèm nhấc bình lên (đo được: vheight dương từ 6.4 s đến 7.4 s),
        nên luật đó chặn luôn chính pha cần nhận ra. Điều kiện "không nhấc" chỉ
        có ý nghĩa lúc BẮT ĐẦU nghiêng — nó dùng để loại trường hợp NHẤC một vật
        đang nằm nghiêng lên (kéo trên khối gỗ ở DexYCB), chứ không để mô tả
        toàn bộ diễn biến.

    (c) Bản trước bắt buộc episode phải kết thúc khi hết `held`. Sai cùng lý do
        (a): episode phải kéo dài tới khi vật DỰNG THẲNG LẠI, không phải tới khi
        nó chạm bàn.
    """
    n = len(contact)
    vheight = sig.get("vheight")

    tilted = contact & (dev > TILT_MIN_DEG) & (dev < 180 - TILT_MIN_DEG)
    upright_seen = (dev < UPRIGHT_START_DEG) & (~np.isnan(dev))
    rising = ((vheight >= H.POUR_MAX_RISE_MM_PER_S) if vheight is not None
              else np.zeros(n, bool))

    episodes = []
    for s, e in runs_of(tilted):
        lo = max(0, s - int(1.5 * fps))
        if not upright_seen[lo:s].any():
            continue                       # bắt đầu khi đã nghiêng -> không phải rót
        # chỉ xét frame ĐẦU của đoạn nghiêng, không xét cả đoạn
        if rising[s:s + max(1, int(0.15 * fps))].all():
            continue                       # vừa nghiêng đã bay lên -> đang NHẤC, không phải rót
        if (e - s) < int(0.2 * fps):
            continue
        episodes.append((s, e))
    return episodes


def split_pour_subskills(start, end, sig, fps):
    """Chia [start, end) thành Orient_Tilt / Hold_Pour_Angle / Return_Upright
    theo DẤU và ĐỘ LỚN tốc độ xoay trục chính."""
    vang = sig["vang"]
    lab = np.array(["Hold_Pour_Angle"] * (end - start), dtype=object)
    for i in range(start, end):
        v = vang[i]
        if np.isnan(v):
            continue
        if v >= TILT_RATE_DEG_S:
            lab[i - start] = "Orient_Tilt"
        elif v <= -TILT_RATE_DEG_S:
            lab[i - start] = "Return_Upright"
    lab = _merge_short(lab, max(1, int(0.15 * fps)), order=POUR_SUBS)
    out, i = [], 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        out.append((start + i, start + j, lab[i]))
        i = j
    return out


# --------------------------------------------------------------- tổng hợp
def infer(mask_json, depth_path=None, verbose=True):
    data, per_frame = H.load_mask_series(mask_json)
    fps = float(data["fps"])
    hand_ids, obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}

    depth = None
    if depth_path:
        depth, _ = H.load_depth(depth_path)

    per_obj = H.frame_features(per_frame, hand_ids, obj_ids, depth=depth)
    active = H.pick_active_object(per_obj)
    sig = H.build_signals(per_obj, active, fps)

    n = len(per_frame)
    dev, ref_ang = compute_dev(sig["ang"], sig["overlap"])

    # "đang được giữ" = nhấc khỏi mặt bàn, đo bằng depth (mm)
    if depth is not None and not np.all(np.isnan(sig["height_mm"])):
        held = np.where(np.isnan(sig["height_mm"]), False,
                        sig["height_mm"] > H.ON_TABLE_MM)
    else:
        held = np.zeros(n, bool)      # không có depth -> không xác định được

    episodes = []

    if depth is not None:
        for oid in obj_ids:
            # TÍN HIỆU RIÊNG CỦA TỪNG VẬT — không dùng chung "vật đang thao tác",
            # nếu không thì cái cốc đứng yên bên cạnh cũng bị gán là đang được
            # cầm chỉ vì tay đang cầm chai ở gần đó.
            ov_o = np.array([per_obj[oid][i]["overlap"] if per_obj[oid][i] else 0
                             for i in range(n)], dtype=float)
            contact_o = H.deglitch_contact(ov_o >= H.CONTACT_MIN_OVERLAP_PX)
            if not contact_o.any():
                continue                      # vật này không hề bị chạm tới
            gap = compute_gap(per_frame, depth, hand_ids, oid)
            vcx, vcy, hold, sp = object_motion(per_obj, oid, fps)
            cdist = np.array([per_obj[oid][i]["cdist"] if per_obj[oid][i]
                              and per_obj[oid][i]["cdist"] is not None else np.nan
                              for i in range(n)])
            vcdist = H._grad(H._smooth(cdist, H.SMOOTH_WINDOW), fps, H.VEL_WINDOW)

            for s, e in find_grasp_episodes(contact_o, gap, vcdist, hold, sp, fps):
                subs = split_grasp_subskills(s, e, gap, fps)
                episodes.append({
                    "skill_en": "Grasp", "skill_vi": H.VI_NAMES.get("Grasp", "Grasp"),
                    "start": round(s / fps, 3), "end": round(e / fps, 3),
                    "object_id": oid, "object_label": id2label.get(oid, ""),
                    "subskills": [
                        {"skill": name, "skill_vi": SUB_VI.get(name, name),
                         "start": round(a / fps, 3), "end": round(b / fps, 3)}
                        for a, b, name in subs],
                })
    elif verbose:
        print("  LƯU Ý: không có --depth -> BỎ QUA nhánh Grasp, vì không có\n"
              "  cách nào đo khoảng cách tay-vật theo chiều sâu (xem docstring).")

    if depth is not None:
        contact = H.deglitch_contact(sig["overlap"] >= H.CONTACT_MIN_OVERLAP_PX)
        for s, e in find_pour_episodes(sig, dev, contact, held, fps):
            subs = split_pour_subskills(s, e, sig, fps)
            oid = active[s] if s < len(active) else None
            episodes.append({
                "skill_en": "Pour", "skill_vi": H.VI_NAMES.get("Pour", "Pour"),
                "start": round(s / fps, 3), "end": round(e / fps, 3),
                "object_id": oid,
                "object_label": id2label.get(oid, "") if oid is not None else "",
                "subskills": [
                    {"skill": name, "skill_vi": SUB_VI.get(name, name),
                     "start": round(a / fps, 3), "end": round(b / fps, 3)}
                    for a, b, name in subs],
            })

    episodes.sort(key=lambda x: x["start"])

    result = {
        "source_masks": mask_json,
        "source_depth": depth_path,
        "fps": fps,
        "n_frames": n,
        "upright_angle_deg": None if np.isnan(ref_ang) else round(ref_ang, 1),
        "gap_align_mm": GAP_ALIGN_MM,
        "tilt_min_deg": TILT_MIN_DEG,
        "episodes": episodes,
    }

    if verbose:
        print(f"Mask  : {mask_json}")
        if depth_path:
            print(f"Depth : {depth_path}")
        print(f"Video : {data.get('video')}  {fps:.2f} fps  {n} frame")
        print(f"Tư thế thẳng của vật: {result['upright_angle_deg']}°")
        print(f"\nSub-skill theo skill cấp cao:\n{render(result)}")
    return result


def render(result):
    lines = []
    for ep in result["episodes"]:
        lines.append(f"  {ep['skill_en']} ({ep['skill_vi']})  "
                     f"{ep['start']:.2f}s - {ep['end']:.2f}s"
                     + (f"  [{ep['object_label']}]" if ep["object_label"] else ""))
        for s in ep["subskills"]:
            lines.append(f"      {s['skill']:<17} {s['skill_vi']:<16} "
                         f"{s['start']:5.2f}s - {s['end']:5.2f}s  "
                         f"({s['end'] - s['start']:.2f}s)")
    return "\n".join(lines) if lines else "  (không tìm thấy episode nào)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks_json")
    ap.add_argument("--depth", default=None,
                    help="BẮT BUỘC để có kết quả đầy đủ — xem docstring")
    ap.add_argument("--out", default="subskills.json")
    args = ap.parse_args()

    result = infer(args.masks_json, args.depth)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi: {args.out}")


if __name__ == "__main__":
    main()
