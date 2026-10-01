#!/usr/bin/env python3
"""
merge_object_ids.py

HẬU XỬ LÝ: gộp lại các ID vật bị SAM2/Grounding DINO phân mảnh trong một
`masks_gsam2.json` ĐÃ CÓ SẴN — không cần chạy lại `gsam2_video.py` (không cần
GPU/model).

VÌ SAO CẦN: `Sam2ChunkTracker._assign_ids` (sam_dino/common.py) nối ID giữa
2 chunk bằng nhãn + IoU bbox. Grounding DINO là open-vocab nên CÙNG MỘT vật có
thể được gọi bằng chữ khác nhau ở keyframe sau ("glass cup" -> "glass" ->
"cup"), khiến vòng nối theo nhãn thất bại và tạo ID MỚI dù bbox gần như không
đổi. Đo được thật trên `video_test3.mp4`: 2 ly thuỷ tinh thật ra 4 ID
(`glass cup`×2, `glass`, `cup`), và chỉ 1/4 ID nhận đủ frame chạm để tính điểm
gắp — 3 ID kia bị `grasp_points_per_object`/`grasp_episodes` bỏ sót vì mỗi ID
chỉ sống một đoạn ngắn.

`common.py::Sam2ChunkTracker._assign_ids` đã được sửa (dùng VAI TRÒ tay/vật
thay vì so khớp chữ nhãn) để các lần CHẠY MỚI không bị lỗi này nữa. Script này
sửa các file ĐÃ SINH RA TRƯỚC ĐÓ bằng đúng logic tương tự, áp dụng hậu kỳ trên
toàn bộ timeline thay vì chỉ 2 chunk liền kề.

THUẬT TOÁN: với mỗi cặp ID (a, b) cùng VAI TRÒ (tay/vật — không bao giờ gộp
tay với vật), a kết thúc ở frame F_a và b bắt đầu ở frame F_b:
  - không được overlap thời gian (mọi frame a từng xuất hiện phải khác mọi
    frame b từng xuất hiện) — đây là điều kiện bắt buộc để không gộp nhầm HAI
    VẬT THẬT đang cùng có mặt trong khung hình;
  - khoảng cách 0 < F_b - F_a <= --max-gap-frames (mặc định 10 ~ 1/3 giây ở
    30fps — đủ để nối qua đúng 1 lần đổi chunk, hầu hết case fragment thật
    có gap = 1 frame);
  - IoU bbox tại điểm nối (bbox cuối của a, bbox đầu của b) >= --min-iou.
Ghép bằng union-find nên nối được CHUỖI dài hơn 2 (a -> b -> c...), đúng như
trường hợp id3 -> id5 -> id3 (nhãn quay lại "glass cup" nên tự nối lại, không
cần script này xử lý, nhưng id2 -> id4 thì không tự nối được vì hết video).

Dùng:
    python3 merge_object_ids.py masks_gsam2.json --out masks_gsam2_merged.json
"""
import argparse
import json


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def role_of(label):
    """Giống hệt heuristic `pick_roles` (hoi_skill_inference.py) và `_role`
    (sam_dino/common.py::Sam2ChunkTracker) — chỉ tách tay/vật, không phân biệt
    sâu hơn vì nhãn open-vocab không đáng tin để làm căn cứ."""
    return "hand" if "hand" in (label or "").lower() else "object"


RUN_SPLIT_GAP = 2   # gap frame lớn hơn -> tách RUN mới (dù cùng 1 id)


def _split_runs(frame_bbox_sorted):
    """Tách 1 id thành các RUN liên tục. CẦN THIẾT vì một id có thể quay lại
    sau khi đã "nhường chỗ" cho id khác — đo được thật trên test3: id2 sống
    frame 0-209, biến mất (id4 thế chỗ 210-254), rồi QUAY LẠI làm id2 ở frame
    255-263 vì tại keyframe 255 Grounding DINO tình cờ trả đúng nhãn cũ nên
    `_assign_ids` tự nối lại được. Coi id2 là MỘT khối 0-263 sẽ làm sai hướng
    gộp: đoạn 210-254 (id4) nằm ở GIỮA hai đoạn của id2, không phải sau nó."""
    runs = [[frame_bbox_sorted[0]]]
    for item in frame_bbox_sorted[1:]:
        if item[0] - runs[-1][-1][0] <= RUN_SPLIT_GAP:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def find_merge_groups(data, max_gap_frames=10, min_iou=0.5, verbose=True):
    """Trả dict {id_cũ: id_gốc_sau_gộp} cho MỌI id trong `data`."""
    objs_meta = {o["id"]: o for o in data["objects"]}
    timeline = {}                      # id -> [(frame, bbox), ...] tăng dần
    for fr in data["frames"]:
        for o in fr["objects"]:
            timeline.setdefault(o["id"], []).append((fr["frame"], o["bbox"]))
    for tl in timeline.values():
        tl.sort(key=lambda x: x[0])

    ids = sorted(timeline)
    frame_sets = {i: {f for f, _ in tl} for i, tl in timeline.items()}
    # runs: list of dict {id, start_frame, end_frame, start_box, end_box}
    runs = []
    for i in ids:
        for run in _split_runs(timeline[i]):
            runs.append({"id": i, "start_frame": run[0][0], "end_frame": run[-1][0],
                        "start_box": run[0][1], "end_box": run[-1][1]})

    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # giữ id có first_frame sớm hơn làm gốc, để "objects" giữ đúng lịch sử
        if objs_meta[ra]["first_frame"] <= objs_meta[rb]["first_frame"]:
            parent[rb] = ra
        else:
            parent[ra] = rb

    merges = []     # log để in ra, kiểm chứng bằng mắt được
    for run_a in runs:
        role_a = role_of(objs_meta[run_a["id"]]["label"])
        best_b, best_iou = None, min_iou
        for run_b in runs:
            if run_b["id"] == run_a["id"]:
                continue           # cùng id thì khỏi cần gộp với chính nó
            if role_of(objs_meta[run_b["id"]]["label"]) != role_a:
                continue
            gap = run_b["start_frame"] - run_a["end_frame"]
            if gap <= 0 or gap > max_gap_frames:
                continue
            iou = box_iou(run_a["end_box"], run_b["start_box"])
            if iou >= best_iou:
                best_b, best_iou = run_b, iou
        if best_b is not None:
            merges.append((run_a["id"], best_b["id"], round(best_iou, 3),
                           best_b["start_frame"] - run_a["end_frame"]))
            union(run_a["id"], best_b["id"])

    # kiểm tra an toàn: 2 id đã gộp chung nhóm không được cùng có mặt ở 1 frame
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    for root, members in groups.items():
        for k in range(len(members)):
            for j in range(k + 1, len(members)):
                if frame_sets[members[k]] & frame_sets[members[j]]:
                    print(f"  CẢNH BÁO: id{members[k]} và id{members[j]} bị gộp "
                          f"chung nhưng từng CÙNG xuất hiện — kiểm tra lại "
                          f"--min-iou/--max-gap-frames.")

    if verbose:
        if merges:
            print(f"Gộp {len(merges)} cặp RUN:")
            for a, b, iou, gap in merges:
                la, lb = objs_meta[a]["label"], objs_meta[b]["label"]
                print(f"  id{a} ({la!r}) -> id{b} ({lb!r})  IoU={iou}  gap={gap} frame")
        else:
            print("Không tìm thấy ID nào để gộp.")

    return {i: find(i) for i in ids}


def apply_merge(data, id_map):
    """Trả bản sao `data` đã remap id theo `id_map` và gộp lại danh sách
    `objects`. Không đụng tới bbox/rle từng detection — chỉ đổi nhãn id."""
    groups = {}
    for old_id, root in id_map.items():
        groups.setdefault(root, []).append(old_id)

    new_objects = []
    for root, members in sorted(groups.items()):
        metas = [next(o for o in data["objects"] if o["id"] == m) for m in members]
        # nhãn CANONICAL = nhãn xuất hiện ở NHIỀU FRAME nhất trong nhóm, không
        # phải nhãn của id gốc — id gốc chỉ được chọn vì first_frame sớm nhất,
        # có thể chỉ sống một đoạn ngắn với nhãn hiếm gặp.
        frame_count = {m["id"]: 0 for m in metas}
        for fr in data["frames"]:
            for o in fr["objects"]:
                if o["id"] in frame_count:
                    frame_count[o["id"]] += 1
        canon_label = max(metas, key=lambda m: frame_count[m["id"]])["label"]
        new_objects.append({
            "id": root,
            "label": canon_label,
            "first_frame": min(m["first_frame"] for m in metas),
            "score": max(m["score"] for m in metas),
            "merged_from": sorted(members) if len(members) > 1 else None,
        })

    new_frames = []
    for fr in data["frames"]:
        objs = [{**o, "id": id_map[o["id"]]} for o in fr["objects"]]
        seen = {}
        for o in objs:                 # phòng thủ: 2 detection cùng frame bị
            seen[o["id"]] = o          # remap về cùng id -> giữ cái sau cùng
        new_frames.append({**fr, "objects": list(seen.values())})

    out = dict(data)
    out["objects"] = new_objects
    out["frames"] = new_frames
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap-frames", type=int, default=10,
                    help="khoảng cách frame tối đa giữa lúc ID cũ biến mất và ID mới xuất hiện")
    ap.add_argument("--min-iou", type=float, default=0.5,
                    help="IoU bbox tối thiểu tại điểm nối để coi là cùng một vật")
    args = ap.parse_args()

    with open(args.masks_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_before = len(data["objects"])
    id_map = find_merge_groups(data, args.max_gap_frames, args.min_iou)
    merged = apply_merge(data, id_map)
    n_after = len(merged["objects"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    print(f"\n{n_before} ID -> {n_after} ID sau gộp. Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
