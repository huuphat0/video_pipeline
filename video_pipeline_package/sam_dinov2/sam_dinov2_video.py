#!/usr/bin/env python3
"""Nhánh B — SAM 2 + DINOv2.

SAM 2 sinh mask tự động ở keyframe (lưới điểm, KHÔNG có text prompt) -> crop từng
mask -> DINOv2 nhúng thành vector 768 chiều -> gán nhãn bằng ảnh mẫu (one-shot)
hoặc gom cụm KMeans -> bbox đưa vào cùng tracker SAM2 của nhánh A.

Ví dụ:
    # chế độ gom cụm (không cần chuẩn bị gì)
    python3 sam_dinov2_video.py ../video_test2.mp4 --n-clusters 6 --outdir out_test2

    # chế độ one-shot (refs/<tên lớp>/*.jpg)
    python3 sam_dinov2_video.py ../video_test2.mp4 --refs refs --outdir out_test2
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import (AutoImageProcessor, AutoModel, Sam2Model, Sam2Processor,
                          Sam2VideoModel, Sam2VideoProcessor)

import amg
import common

SAM2_ID = "facebook/sam2.1-hiera-large"
DINO_ID = "facebook/dinov2-base"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--interval", type=float, default=1.5)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--max-side", type=int, default=720)
    p.add_argument("--outdir", default="out_sam_dinov2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # sinh mask
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--pred-iou-thresh", type=float, default=0.80)
    p.add_argument("--min-area-frac", type=float, default=0.0015)
    p.add_argument("--max-area-frac", type=float, default=0.15)
    p.add_argument("--nms-iou", type=float, default=0.75)
    p.add_argument("--max-objects", type=int, default=8,
                   help="số mask giữ lại mỗi keyframe để đưa vào tracker")
    p.add_argument("--rank", choices=["iou", "area"], default="iou",
                   help="tiêu chí chọn mask: iou = điểm chất lượng của SAM (mặc định, "
                        "ưu tiên vật gọn rõ), area = diện tích (dễ trúng tường/sàn)")
    # gán nhãn
    p.add_argument("--refs", default=None,
                   help="thư mục refs/<tên lớp>/*.jpg -> gán nhãn one-shot")
    p.add_argument("--sim-threshold", type=float, default=0.55,
                   help="cosine tối thiểu để nhận nhãn, dưới ngưỡng -> unknown")
    p.add_argument("--keep-unknown", action="store_true",
                   help="chế độ --refs: giữ cả mask không khớp lớp nào (mặc định bỏ, "
                        "vì phần lớn là nền: tường, sàn, dây cáp)")
    p.add_argument("--n-clusters", type=int, default=6,
                   help="số cụm KMeans khi không có --refs")
    p.add_argument("--save-crops", action="store_true",
                   help="lưu từng crop ra crops/<nhãn>/ để chọn làm ảnh mẫu cho --refs")
    return p.parse_args()


# ------------------------------------------------------------------ DINOv2
@torch.inference_mode()
def embed_crops(dino, dino_proc, crops, device):
    """crops: list ảnh RGB. Trả (N, 768) đã L2-normalize."""
    if not crops:
        return np.zeros((0, 768), dtype=np.float32)
    inputs = dino_proc(images=crops, return_tensors="pt").to(device)
    out = dino(**inputs)
    emb = out.last_hidden_state[:, 0]                  # CLS token
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.float().cpu().numpy()


def crop_from_mask(image, mask, bbox, pad=8, bg_dim=0.5):
    """Crop theo bbox, làm mờ nền ngoài mask để DINOv2 tập trung vào vật."""
    H, W = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1][..., None]
    crop = crop * m + crop * (1 - m) * bg_dim
    return crop.astype(np.uint8)


def load_refs(path, dino, dino_proc, device):
    """refs/<tên lớp>/*.jpg -> (labels, prototype (C, 768))."""
    labels, protos = [], []
    for d in sorted(os.listdir(path)):
        full = os.path.join(path, d)
        if not os.path.isdir(full):
            continue
        files = sorted(sum([glob.glob(os.path.join(full, e))
                            for e in ("*.jpg", "*.jpeg", "*.png")], []))
        if not files:
            continue
        imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
        emb = embed_crops(dino, dino_proc, imgs, device)
        v = emb.mean(0)
        protos.append(v / (np.linalg.norm(v) + 1e-8))
        labels.append(d)
        print(f"      lớp {d!r}: {len(files)} ảnh mẫu")
    if not labels:
        raise RuntimeError(f"không tìm thấy lớp nào trong {path}")
    return labels, np.stack(protos)


# ------------------------------------------------------------------- main
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[1/6] đọc video {args.video}")
    frames, fps, (H, W) = common.read_video(args.video, args.max_seconds, args.max_side)
    keyframes = common.keyframe_indices(len(frames), fps, args.interval)
    print(f"      {len(frames)} frame, {fps:.1f} fps, {W}x{H}, {len(keyframes)} keyframe")

    print(f"[2/6] nạp model lên {device}")
    t0 = time.perf_counter()
    sam_img_proc = Sam2Processor.from_pretrained(SAM2_ID)
    sam_img = Sam2Model.from_pretrained(SAM2_ID, dtype=dtype).to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
    dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
    sam_vid_proc = Sam2VideoProcessor.from_pretrained(SAM2_ID)
    sam_vid = Sam2VideoModel.from_pretrained(SAM2_ID, dtype=dtype).to(device).eval()
    print(f"      xong sau {time.perf_counter() - t0:.1f}s")

    ref_labels, ref_protos = (None, None)
    if args.refs:
        print(f"[3/6] nạp ảnh mẫu từ {args.refs}")
        ref_labels, ref_protos = load_refs(args.refs, dino, dino_proc, device)
    else:
        print(f"[3/6] không có --refs -> sẽ gom cụm KMeans (k={args.n_clusters})")

    # --- lượt 1: sinh mask + nhúng ở mọi keyframe -------------------------
    print("[4/6] SAM2 sinh mask + DINOv2 nhúng ở keyframe")
    kf_data, all_emb, t_amg, t_emb = {}, [], [], []
    for kf in keyframes:
        img = frames[kf]
        t0 = time.perf_counter()
        cands = amg.generate_masks(
            sam_img, sam_img_proc, img, device,
            points_per_side=args.points_per_side, pred_iou_thresh=args.pred_iou_thresh,
            min_area_frac=args.min_area_frac, max_area_frac=args.max_area_frac,
            nms_iou=args.nms_iou)
        t_amg.append(time.perf_counter() - t0)

        for c in cands:
            c["bbox"] = common.mask_bbox(c["mask"])
        cands = [c for c in cands if c["bbox"] is not None]
        cands.sort(key=lambda c: -(c["iou"] if args.rank == "iou" else c["area"]))
        cands = cands[:args.max_objects]

        t0 = time.perf_counter()
        crops = [crop_from_mask(img, c["mask"], c["bbox"]) for c in cands]
        emb = embed_crops(dino, dino_proc, crops, device)
        t_emb.append(time.perf_counter() - t0)

        for i, c in enumerate(cands):
            c["emb_idx"] = len(all_emb) + i
            c["crop"] = crops[i]
            c["kf"] = kf
        all_emb.extend(emb)
        kf_data[kf] = cands
        print(f"      keyframe {kf}: {len(cands)} mask")

    all_emb = np.stack(all_emb) if all_emb else np.zeros((0, 768), np.float32)

    # --- gán nhãn ---------------------------------------------------------
    print("[5/6] gán nhãn")
    if ref_protos is not None:
        sims = all_emb @ ref_protos.T                    # (N, C)
        best = sims.argmax(1)
        names = [ref_labels[b] if sims[i, b] >= args.sim_threshold else "unknown"
                 for i, b in enumerate(best)]
        conf = [float(sims[i, b]) for i, b in enumerate(best)]
    else:
        k = min(args.n_clusters, len(all_emb))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(all_emb)
        names = [f"cluster_{c}" for c in km.labels_]
        centers = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1,
                                                        keepdims=True) + 1e-8)
        conf = [float(all_emb[i] @ centers[c]) for i, c in enumerate(km.labels_)]
        save_cluster_grid(os.path.join(args.outdir, "clusters.png"), kf_data, names)

    for cands in kf_data.values():
        for c in cands:
            c["label"] = names[c["emb_idx"]]
            c["score"] = conf[c["emb_idx"]]

    if args.save_crops:
        for cands in kf_data.values():
            for c in cands:
                d = os.path.join(args.outdir, "crops", c["label"])
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f'kf{c["kf"]:04d}_{c["emb_idx"]:03d}.png'),
                            cv2.cvtColor(c["crop"], cv2.COLOR_RGB2BGR))
        print(f"      crop đã lưu vào {os.path.join(args.outdir, 'crops')}/")

    np.savez_compressed(
        os.path.join(args.outdir, "embeddings.npz"),
        emb=all_emb,
        label=np.array([names[i] for i in range(len(all_emb))]),
        conf=np.array(conf, dtype=np.float32),
        frame_idx=np.array([kf for kf, cs in kf_data.items() for _ in cs]),
        bbox=np.array([c["bbox"] for cs in kf_data.values() for c in cs], dtype=np.int32),
    )

    if ref_protos is not None and not args.keep_unknown:
        for kf in kf_data:
            n0 = len(kf_data[kf])
            kf_data[kf] = [c for c in kf_data[kf] if c["label"] != "unknown"]
            print(f"      keyframe {kf}: giữ {len(kf_data[kf])}/{n0} mask "
                  f"({', '.join(c['label'] for c in kf_data[kf]) or '-'})")

    # --- lượt 2: tracking (cùng tracker với nhánh A) ----------------------
    print("[6/6] SAM2 track + ghi kết quả")
    detections = {kf: [{"box": [float(v) for v in c["bbox"]], "label": c["label"],
                        "score": c["score"]} for c in cs]
                  for kf, cs in kf_data.items()}
    tracker = common.Sam2ChunkTracker(sam_vid, sam_vid_proc, device, dtype,
                                      link_by_label=ref_protos is not None)
    t0 = time.perf_counter()
    per_frame, timing = tracker.run(frames, keyframes, lambda f, i: detections.get(i, []))
    total = time.perf_counter() - t0

    vid_out = os.path.join(args.outdir, "annotated_sam_dinov2.mp4")
    common.write_video(vid_out, [common.draw_frame(f, o)
                                 for f, o in zip(frames, per_frame)], fps)
    common.dump_masks_json(
        os.path.join(args.outdir, "masks_sam_dinov2.json"),
        {"video": args.video, "fps": fps, "size": [H, W],
         "mode": "refs" if ref_protos is not None else "kmeans",
         "points_per_side": args.points_per_side, "max_objects": args.max_objects},
        per_frame, fps, tracker.objects)

    timing.update({
        "amg_ms_per_keyframe": round(1000 * float(np.mean(t_amg)), 1),
        "dinov2_ms_per_keyframe": round(1000 * float(np.mean(t_emb)), 1),
        "track_total_s": round(total, 1),
        "n_masks_total": int(len(all_emb)),
    })
    with open(os.path.join(args.outdir, "timing_sam_dinov2.json"), "w") as f:
        json.dump(timing, f, indent=2)

    n_hit = sum(1 for o in per_frame if o)
    print(f"\n  video     : {vid_out}")
    print(f"  object    : {len(tracker.objects)} — " +
          ", ".join(f'#{o["id"]} {o["label"]}' for o in tracker.objects.values()))
    print(f"  phủ frame : {n_hit}/{len(frames)} ({100 * n_hit / len(frames):.0f}%)")
    print(f"  thời gian : AMG {timing['amg_ms_per_keyframe']} ms/keyframe, "
          f"DINOv2 {timing['dinov2_ms_per_keyframe']} ms/keyframe, "
          f"track {timing['track_ms_per_frame']} ms/frame")


def save_cluster_grid(path, kf_data, names, thumb=96, per_row=10):
    """Lưới ảnh crop nhóm theo cụm, để người xem đặt tên cho từng cụm."""
    groups = {}
    for cands in kf_data.values():
        for c in cands:
            groups.setdefault(names[c["emb_idx"]], []).append(c["crop"])
    if not groups:
        return
    rows = []
    for name in sorted(groups, key=lambda n: int(n.split("_")[1])):
        crops = groups[name][:per_row]
        cells = [cv2.resize(c, (thumb, thumb)) for c in crops]
        cells += [np.zeros((thumb, thumb, 3), np.uint8)] * (per_row - len(cells))
        row = np.hstack(cells)
        band = np.zeros((22, row.shape[1], 3), np.uint8)
        cv2.putText(band, f"{name}  (n={len(groups[name])})", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([band, row]))
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
