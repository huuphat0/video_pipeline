"""
skill_params.py

Kết xuất **THAM SỐ SKILL** — bản ghi mà Task Planner của robot dùng được để
thực hiện lại skill, KHÁC với `skills_lowlevel.json` (chỉ có nhãn + thời gian).

VÌ SAO CẦN RIÊNG: nhãn "Grasp" và "Pour" chưa đủ để robot làm. Robot cần biết
nắm vật NÀO, nắm CHỖ NÀO, tới TỪ HƯỚNG NÀO, và trạng thái đổi ra sao. Bản ghi ở
đây gom đúng những thứ đó.

NGUYÊN TẮC QUAN TRỌNG — học SKILL, không học CHUYỂN ĐỘNG:
  Toạ độ TUYỆT ĐỐI của tay trong ảnh KHÔNG được đưa vào bản ghi. Camera quay
  người khác camera của robot, nên con số đó vô nghĩa khi chuyển sang. Mọi thứ
  ở đây đều là TƯƠNG ĐỐI VỚI VẬT:

    - điểm chạm   : vị trí trên chính vật thể (0..1 dọc trục + lệch ngang)
    - hướng tới   : tay đi tới vật từ hướng nào, tính bằng góc so với vật
    - độ cao      : mm so với mặt bàn (đã tự hiệu chuẩn)
    - góc vật     : hướng trục chính của vật

  Tâm tay chỉ dùng để suy ra HƯỚNG, và chỉ đo TRƯỚC khi chạm — sau khi đã chạm
  thì vị trí tương tác được neo vào vật nên không cần tâm tay nữa. Điều này
  cũng né được việc mask `hand` của SAM2 gồm cả cẳng tay (không có cách hình
  học nào tách lòng bàn tay ra một cách đáng tin — đã đo, cách "điểm dày nhất"
  hỏng hẳn ở một frame).

Cách dùng:
    python3 skill_params.py masks_gsam2.json --skills skills_lowlevel.json \\
        --depth demo_depth.npz --out skill_params.json
"""

import argparse
import json

import numpy as np

import hoi_skill_inference as H
import contact_point as CP
import subskill_inference as S

# dùng chung ngưỡng gap với tầng sub-skill để hai bên nhất quán
GAP_ALIGN_MM = S.GAP_ALIGN_MM


def CP_import_compute_gap(per_frame, depth, hand_ids, oid):
    """Bọc lại `contact_point`/`subskill_inference` để chỗ gọi gọn hơn."""
    return S.compute_gap(per_frame, depth, hand_ids, oid)

# Cửa sổ lấy hướng tiếp cận: các frame NGAY TRƯỚC khi chạm
APPROACH_LOOKBACK_SEC = 0.30
MIN_CONTACT_FRAMES = 3      # đoạn có ít frame chạm hơn -> không kết xuất điểm chạm


def _median_of(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.median(v)), 3) if v else None


def build(masks_json, skills_json, depth_path=None, verbose=True):
    data, per_frame = H.load_mask_series(masks_json)
    fps = float(data["fps"])
    hand_ids, obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}
    with open(skills_json, "r", encoding="utf-8") as f:
        skills = json.load(f)

    depth = H.load_depth(depth_path)[0] if depth_path else None
    n = len(per_frame)

    # Tính feature + mốc tham chiếu MỘT LẦN cho cả video. Nếu gọi trong vòng
    # lặp từng frame thì thành O(n²) — đã bị treo thật khi chạy lần đầu.
    per_obj = H.frame_features(per_frame, hand_ids, obj_ids, depth=depth)
    refs = H._object_references(per_obj)

    def masks_at(i):
        m = per_frame[i] if i < n else {}
        hm = None
        for h in hand_ids:
            if h in m:
                hm = m[h] if hm is None else (hm | m[h])
        return m, hm

    # CHẠM ĐẦU TIÊN của từng vật: hướng tiếp cận chỉ có nghĩa ở đó. Dùng
    # `gap` (khoảng cách theo chiều sâu) khi có depth, vì mask 2D nới rộng báo
    # chạm cả khi tay còn ở TRÊN CAO; không có depth thì đành dùng overlap 2D.
    import cv2
    first_contact = {}
    for oid in obj_ids:
        ov = np.array([per_obj[oid][i]["overlap"] if per_obj[oid][i] else 0
                       for i in range(n)], dtype=float)
        if depth is not None:
            gap = CP_import_compute_gap(per_frame, depth, hand_ids, oid)
            near = np.array([(not np.isnan(g)) and g <= GAP_ALIGN_MM for g in gap])
        else:
            near = ov >= H.CONTACT_MIN_OVERLAP_PX
        idxs = np.nonzero(near)[0]
        first_contact[oid] = int(idxs[0]) if idxs.size else None

    out_segments = []
    for seg in skills["segments"]:
        oid = seg.get("object_id")
        t0, t1 = seg["time_sec"], seg["end_sec"]
        i0, i1 = int(t0 * fps), min(int(t1 * fps), n)
        if oid is None or i1 <= i0:
            continue

        contacts, alongs, parts, sides = [], [], [], []
        heights = []
        for i in range(i0, i1):
            m, hm = masks_at(i)
            om = m.get(oid)
            if om is None or hm is None:
                continue
            ci = CP.contact_info(om, hm)
            if ci is None or not ci.get("exact"):
                continue
            contacts.append((ci["x"], ci["y"]))
            if "along" in ci:
                alongs.append(ci["along"])
                parts.append(ci["label"])
                sides.append(ci.get("side", ""))
            if depth is not None:
                df = depth[i] if i < len(depth) else None
                if df is not None:
                    mm = om
                    if mm.shape[:2] != df.shape[:2]:
                        import cv2
                        mm = cv2.resize(mm.astype(np.uint8), (df.shape[1], df.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    d = H._median_depth_in_mask(df, mm)
                    dref = refs.get(oid, (np.nan, np.nan))[1]
                    if not np.isnan(d) and not np.isnan(dref):
                        heights.append(max(0.0, dref - d))

        rec = {
            "skill": seg["skill_en"], "skill_vi": seg["skill_vi"],
            "object": id2label.get(oid, ""),
            "start": t0, "end": t1,
        }

        # --- ĐIỂM CHẠM (chỉ khi thật sự có tiếp xúc) ---
        if len(contacts) >= MIN_CONTACT_FRAMES:
            cx = _median_of([p[0] for p in contacts])
            cy = _median_of([p[1] for p in contacts])
            rec["contact_on_object"] = {
                "x_px": cx, "y_px": cy,
                "along": _median_of(alongs),               # 0 = đáy, 1 = đầu trên
                "part": max(set(parts), key=parts.count) if parts else None,
                "side": max(set(sides), key=sides.count) if any(sides) else "",
                "n_frames": len(contacts),
            }

        # --- HƯỚNG TIẾP CẬN: chỉ có nghĩa ở đoạn CHẠM ĐẦU TIÊN của một lần
        # tương tác. Với các đoạn sau (Lift, Pour, Place...) tay đã ở trên vật
        # từ trước, nên "hướng tay so với vật" chỉ là vị trí hiện tại — không
        # phải hướng đi tới. Đã thấy thật: đoạn Pour bị ghi "tới từ trên-trái"
        # trong khi thực ra không hề có động tác tới nào ở đó.
        # Hướng tiếp cận CHỈ gắn với đoạn chứa frame CHẠM ĐẦU TIÊN của vật.
        fc = first_contact.get(oid)
        is_first_contact_seg = (fc is not None and i0 <= fc < i1)
        if rec.get("contact_on_object") and is_first_contact_seg:
            degs, labels = [], []
            for i in range(max(0, i0 - int(APPROACH_LOOKBACK_SEC * fps)), i0 + 1):
                m, hm = masks_at(i)
                om = m.get(oid)
                if om is None or hm is None:
                    continue
                # trước khi chạm thì neo vào TÂM VẬT (chưa có điểm chạm)
                d, lb = CP.approach_direction(om, hm, contact_xy=None)
                if d is not None:
                    degs.append(d); labels.append(lb)
            if degs:
                rec["approach"] = {"deg": _median_of(degs),
                                   "label": max(set(labels), key=labels.count)}

        if heights:
            rec["height_mm"] = {"at_start": round(heights[0], 1),
                                "max": round(max(heights), 1),
                                "at_end": round(heights[-1], 1)}

        out_segments.append(rec)

    result = {
        "source_masks": masks_json, "source_skills": skills_json,
        "source_depth": depth_path, "fps": fps,
        "note": ("Tham số để robot thực hiện lại skill. Mọi vị trí đều TƯƠNG ĐỐI "
                 "VỚI VẬT, không dùng toạ độ tuyệt đối của tay trong ảnh — camera "
                 "người quay khác camera robot nên toạ độ tuyệt đối không chuyển "
                 "được. Xem docstring skill_params.py."),
        "skills": out_segments,
    }

    if verbose:
        print(f"Mask  : {masks_json}")
        print(f"Depth : {depth_path or '(không có — bỏ độ cao)'}")
        print(f"\nTham số skill:\n{render(result)}")
    return result


def grasp_points_per_object(masks_json, depth_path=None):
    """BẢNG ĐIỂM GẮP CHO MỌI VẬT trong video — không chỉ vật đang thao tác.

    Vì sao cần riêng: `build()` chỉ kết xuất theo các ĐOẠN SKILL, mà đoạn skill
    chỉ có một vật "đang thao tác". Muốn mở rộng ra nhiều loại vật thì cần biết
    vật nào có được chạm không, chạm ở đâu, và mô tả bằng trục nào.
    """
    data, per_frame = H.load_mask_series(masks_json)
    fps = float(data["fps"])
    hand_ids, obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}
    depth = H.load_depth(depth_path)[0] if depth_path else None
    intr = H.load_intrinsics(depth_path) if depth_path else None
    per_obj = H.frame_features(per_frame, hand_ids, obj_ids, depth=depth)
    refs = H._object_references(per_obj)

    rows = []
    for oid in obj_ids:
        contacts, alongs, labels, sides, axes = [], [], [], [], []
        elongs, pts, xyzs = [], [], []
        # CỔNG XÁC THỰC BẰNG DEPTH: `find_contact` dựa trên mask 2D nới rộng nên
        # báo "chạm" cả khi bàn tay ở TRÊN CAO (chiếu vuông góc xuống trùng chỗ
        # vật). Có depth thì phải lọc thêm bằng `gap` giống hệt video đang vẽ —
        # nếu không, bảng tổng hợp và video sẽ nói hai chuyện khác nhau.
        gap = (S.compute_gap(per_frame, depth, hand_ids, oid)
               if depth is not None else None)
        for i, m in enumerate(per_frame):
            om = m.get(oid)
            if om is None or not om.any():
                continue
            hm = None
            for h in hand_ids:
                if h in m:
                    hm = m[h] if hm is None else (hm | m[h])
            if hm is None:
                continue
            ci = CP.contact_info(om, hm)
            if ci is None or not ci.get("exact"):
                continue
            if gap is not None:
                g = gap[i] if i < len(gap) else np.nan
                if np.isnan(g) or g > GAP_ALIGN_MM:
                    continue              # tay còn ở trên cao -> chưa thật chạm
            pts.append((round(ci["x"], 1), round(ci["y"], 1)))
            if "along" in ci:
                alongs.append(ci["along"]); labels.append(ci["label"])
                sides.append(ci.get("side", "")); axes.append(ci.get("axis"))
                elongs.append(ci.get("elongation"))
            if depth is not None and intr is not None and i < len(depth):
                xyz = CP.contact_point_xyz_mm(ci, om.shape, depth[i], intr)
                if xyz is not None:
                    xyzs.append(xyz)

        if not pts:
            rows.append({"object": id2label.get(oid, ""), "contact_frames": 0})
            continue
        dref = refs.get(oid, (np.nan, np.nan))[1]
        heights = []
        for i, m in enumerate(per_frame):
            om = m.get(oid)
            if om is None or not om.any() or depth is None or i >= len(depth):
                continue
            mm = om
            df = depth[i]
            if mm.shape[:2] != df.shape[:2]:
                import cv2
                mm = cv2.resize(mm.astype(np.uint8), (df.shape[1], df.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            d = H._median_depth_in_mask(df, mm)
            if not np.isnan(d) and not np.isnan(dref):
                heights.append(max(0.0, dref - d))

        rows.append({
            "object": id2label.get(oid, ""),
            "contact_frames": len(pts),
            "grasp_point_px": {"x": _median_of([p[0] for p in pts]),
                               "y": _median_of([p[1] for p in pts])},
            # Toạ độ 3D THẬT (mm) trong HỆ CAMERA — chỉ có khi --depth trỏ tới
            # 1 file .npz CÓ nội tham số (fx/fy/cx/cy, xem depth_from_mcap.py).
            # Đây là hệ CAMERA, không phải hệ robot/hệ vật — xem cảnh báo
            # trong `contact_point.py::contact_point_xyz_mm`.
            "grasp_point_xyz_mm": (None if not xyzs else {
                "x": _median_of([p["x_mm"] for p in xyzs]),
                "y": _median_of([p["y_mm"] for p in xyzs]),
                "z": _median_of([p["z_mm"] for p in xyzs]),
            }),
            "part": max(set(labels), key=labels.count) if labels else None,
            "along": _median_of(alongs),
            "side": max(set(sides), key=sides.count) if any(sides) else "",
            "axis_used": max(set(axes), key=axes.count) if axes else None,
            "elongation": _median_of(elongs),
            "height_range_mm": (None if not heights else
                                {"min": round(min(heights), 1),
                                 "max": round(max(heights), 1)}),
        })
    return {"source_masks": masks_json, "fps": fps, "objects": rows}


def validated_contacts(per_frame, hand_ids, oid, depth=None, intr=None):
    """Chuỗi tiếp xúc ĐÃ XÁC THỰC của vật `oid` -> (cflags, info, xyz).

    cflags[i] = True khi tay chạm vật thật ở frame i (mask chồng lấn VÀ, nếu
    có depth, tay cùng tầm sâu với vật — cổng `gap` giống video đang vẽ).
    info[i] = dict của `CP.contact_info`, xyz[i] = toạ độ 3D hệ camera (mm).
    Dùng chung cho `grasp_episodes` và `grasp_card.py` để hai bên luôn cùng
    định nghĩa "đang nắm"."""
    n = len(per_frame)
    gap = (S.compute_gap(per_frame, depth, hand_ids, oid)
           if depth is not None else None)
    cflags = np.zeros(n, bool)
    per_frame_info = [None] * n
    per_frame_xyz = [None] * n
    for i, m in enumerate(per_frame):
        om = m.get(oid)
        if om is None or not om.any():
            continue
        hm = None
        for h in hand_ids:
            if h in m:
                hm = m[h] if hm is None else (hm | m[h])
        if hm is None:
            continue
        ci = CP.contact_info(om, hm)
        if ci is None or not ci.get("exact"):
            continue
        if gap is not None:
            g = gap[i] if i < len(gap) else np.nan
            if np.isnan(g) or g > GAP_ALIGN_MM:
                continue
        cflags[i] = True
        per_frame_info[i] = ci
        if depth is not None and intr is not None and i < len(depth):
            per_frame_xyz[i] = CP.contact_point_xyz_mm(ci, om.shape, depth[i], intr)
    return cflags, per_frame_info, per_frame_xyz


def grasp_episodes(masks_json, depth_path=None, min_frames=5):
    """GOM MỌI VỊ TRÍ GẮP trên một vật từ MỘT video.

    Mỗi LẦN CẦM (một đoạn tiếp xúc liên tục) cho một điểm gắp, quy về HỆ CỦA
    VẬT. Nhờ vậy một video người cầm vật nhiều lần sẽ cho một TẬP điểm gắp
    thay vì chỉ một — đây là cách mở rộng ra nhiều vị trí gắp.

    Vì sao phải quy về hệ của vật chứ không dùng toạ độ ảnh: trong lúc cầm,
    vật có thể xoay (chai nghiêng để rót) nhưng tay KHÔNG trượt — nên vị trí
    gắp tính theo vật phải đứng yên. Đo được trên video demo: dùng trọng tâm
    vùng chạm chiếu lên trục chính thì trôi 0.097 (thang 0-1 theo chiều dài
    vật), trong khi toạ độ tâm tay trong ảnh trôi 44 px.
    """
    data, per_frame = H.load_mask_series(masks_json)
    fps = float(data["fps"])
    hand_ids, obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}
    depth = H.load_depth(depth_path)[0] if depth_path else None
    intr = H.load_intrinsics(depth_path) if depth_path else None
    out = []

    for oid in obj_ids:
        cflags, per_frame_info, per_frame_xyz = validated_contacts(
            per_frame, hand_ids, oid, depth, intr)

        for a, b in S.runs_of(cflags):
            if b - a < min_frames:
                continue
            al = [per_frame_info[i]["along"] for i in range(a, b)
                  if per_frame_info[i] and "along" in per_frame_info[i]]
            sd = [per_frame_info[i]["side"] for i in range(a, b)
                  if per_frame_info[i]]
            lb = [per_frame_info[i]["label"] for i in range(a, b)
                  if per_frame_info[i] and "label" in per_frame_info[i]]
            xyzs = [per_frame_xyz[i] for i in range(a, b) if per_frame_xyz[i]]
            if not al:
                continue
            out.append({
                "object": id2label.get(oid, ""),
                "start": round(a / fps, 2), "end": round(b / fps, 2),
                "duration_sec": round((b - a) / fps, 2),
                "grasp_point": {
                    "along": _median_of(al),                      # 0..1 theo chiều dài vật
                    "part": max(set(lb), key=lb.count) if lb else None,
                    "side": max(set(sd), key=sd.count) if any(sd) else "",
                },
                # Hệ CAMERA (mm) — xem cảnh báo hệ quy chiếu ở contact_point.py
                "grasp_point_xyz_mm": (None if not xyzs else {
                    "x": _median_of([p["x_mm"] for p in xyzs]),
                    "y": _median_of([p["y_mm"] for p in xyzs]),
                    "z": _median_of([p["z_mm"] for p in xyzs]),
                }),
                "n_frames": int(b - a),
            })
    out.sort(key=lambda x: x["start"])
    return {"source_masks": masks_json, "source_depth": depth_path, "fps": fps,
            "grasp_episodes": out}


def render(result):
    lines = []
    for s in result["skills"]:
        lines.append(f"  {s['start']:5.2f}-{s['end']:5.2f}s  {s['skill']:<13} [{s['object']}]")
        c = s.get("contact_on_object")
        if c:
            side = f" · {c['side']}" if c["side"] else ""
            lines.append(f"      chạm: {c['part']}{side}  (along={c['along']}, "
                         f"{c['n_frames']} frame)")
        a = s.get("approach")
        if a:
            lines.append(f"      tới từ: {a['label']}  ({a['deg']}°)")
        h = s.get("height_mm")
        if h:
            lines.append(f"      độ cao: {h['at_start']} → {h['max']} → {h['at_end']} mm")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks_json")
    ap.add_argument("--skills", default=None,
                    help="skills_lowlevel.json — bắt buộc với --mode segments (mặc định)")
    ap.add_argument("--depth", default=None)
    ap.add_argument("--out", default="skill_params.json")
    ap.add_argument("--mode", choices=["segments", "grasp_points", "grasp_episodes"],
                    default="segments",
                    help="segments: tham số theo đoạn skill (cần --skills, 1 vật/đoạn). "
                         "grasp_points: MỘT điểm gắp gộp cho MỖI vật trong video. "
                         "grasp_episodes: MỘT điểm gắp cho MỖI LẦN CẦM (nhiều lần/vật).")
    args = ap.parse_args()

    if args.mode == "segments":
        if not args.skills:
            ap.error("--mode segments cần --skills skills_lowlevel.json")
        result = build(args.masks_json, args.skills, args.depth)
    elif args.mode == "grasp_points":
        result = grasp_points_per_object(args.masks_json, args.depth)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = grasp_episodes(args.masks_json, args.depth)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi: {args.out}")


if __name__ == "__main__":
    main()
