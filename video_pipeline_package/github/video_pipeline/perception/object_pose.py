"""
object_pose.py

POSE của vật theo TỪNG FRAME: vị trí thật (X,Y,Z mm, hệ CAMERA) + hướng trục
chính trong KHÔNG GIAN THẬT (PCA 3D trên đám mây điểm dựng từ mask+depth),
KHÁC với góc trong `hoi_skill_inference._principal_angle_deg` — đó là góc
CHIẾU LÊN ẢNH, bị rút ngắn phối cảnh khi vật nghiêng ra xa/lại gần camera.

CHỈ CÓ 2/3 TRỤC XOAY (hướng chỉ của trục dài — tilt bao nhiêu độ, nghiêng về
đâu). THIẾU trục thứ 3 (ROLL — xoay quanh chính trục dài vật, vd chai xoay tại
chỗ quanh thân nó): với vật ĐỐI XỨNG TRỤC (chai, ly, cốc tròn) roll KHÔNG có
ý nghĩa vật lý để gắp (nắm chỗ nào quanh thân cũng như nhau). Vật KHÔNG đối
xứng (kéo, khoan...) cần pose estimator riêng (vd FoundationPose) — xem
tongket.md mục P3.

Cách dùng:
    python3 object_pose.py masks_gsam2.json --depth demo_depth.npz --out pose3d.json
"""
import argparse
import json

import os
import sys

# cho phép chạy trực tiếp `python <thư mục>/<file>.py`: đưa gốc video_pipeline/ vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skill_inference import hoi_skill_inference as H


def object_pose_series(masks_json, depth_path):
    """Trả {object_label: [{frame, t, center_mm, axis, tilt_deg, ...}, ...]}
    — 1 bản ghi cho mỗi frame vật đó CÓ MẶT trong khung hình (không chỉ lúc
    bị chạm), nên dùng được để xem cả pose lúc vật đứng yên trên bàn."""
    data, per_frame = H.load_mask_series(masks_json)
    fps = float(data["fps"])
    _, obj_ids = H.pick_roles(data)
    id2label = {o["id"]: o["label"] for o in data["objects"]}
    depth, _ = H.load_depth(depth_path)
    intr = H.load_intrinsics(depth_path)
    if intr is None:
        raise RuntimeError("File depth không có nội tham số camera (fx/fy/cx/cy) "
                           "— chạy lại depth_from_mcap.py bản mới để có pose 3D")

    out = {oid: [] for oid in obj_ids}
    for i, m in enumerate(per_frame):
        if i >= len(depth):
            break
        for oid in obj_ids:
            om = m.get(oid)
            if om is None or not om.any():
                continue
            pts = H.backproject_mask_xyz(om, depth[i], intr)
            pose = H.principal_axis_3d(pts) if pts is not None else None
            if pose is None:
                continue
            confident = pose["elongation"] >= H.ELONG_MIN
            row = {
                "frame": i, "t": round(i / fps, 3),
                "center_mm": {"x": round(float(pose["center_mm"][0]), 1),
                             "y": round(float(pose["center_mm"][1]), 1),
                             "z": round(float(pose["center_mm"][2]), 1)},
                "elongation": round(pose["elongation"], 2),
                "axis_confident": bool(confident),
            }
            # axis/tilt chỉ đáng tin khi elongation >= ELONG_MIN (vật đủ dài
            # để trục chính có nghĩa) — giống hệt điều kiện bên contact_point.py
            if confident:
                axis = pose["axis"]
                row["axis"] = [round(float(v), 4) for v in axis]
                row["tilt_deg"] = round(H.axis_tilt_deg(axis), 1)
            out[oid].append(row)

    return {
        "source_masks": masks_json, "source_depth": depth_path, "fps": fps,
        "note": ("Pose có 2/3 trục xoay (hướng trục chính trong KHÔNG GIAN "
                "THẬT, không phải góc chiếu ảnh) + vị trí X,Y,Z hệ CAMERA "
                "(không phải hệ robot — cần hiệu chuẩn camera->robot riêng, "
                "xem cảnh báo ở hoi_skill_inference.pixel_to_xyz_mm). THIẾU "
                "roll quanh chính trục dài — vô nghĩa vật lý với vật đối "
                "xứng trục (chai/ly/cốc). `axis_confident=false` khi vật quá "
                "tròn (elongation < 2.0) — trục chính không đáng tin, chỉ nên "
                "dùng center_mm lúc đó."),
        "objects": {id2label.get(oid, str(oid)): rows
                   for oid, rows in out.items() if rows},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks_json")
    ap.add_argument("--depth", required=True,
                    help="file .npz của depth_from_mcap.py, CẦN có fx/fy/cx/cy")
    ap.add_argument("--out", default="object_pose.json")
    args = ap.parse_args()

    result = object_pose_series(args.masks_json, args.depth)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    n = sum(len(v) for v in result["objects"].values())
    print(f"Đã ghi {args.out} — {len(result['objects'])} vật, {n} bản ghi pose")
    for label, rows in result["objects"].items():
        n_conf = sum(1 for r in rows if r["axis_confident"])
        print(f"  {label}: {len(rows)} frame, {n_conf} frame trục đáng tin "
              f"(elongation >= {H.ELONG_MIN})")


if __name__ == "__main__":
    main()
