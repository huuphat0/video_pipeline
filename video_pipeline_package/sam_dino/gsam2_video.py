#!/usr/bin/env python3
"""Nhánh A — Grounded-SAM: Grounding DINO (text -> box) ở keyframe,
SAM 2.1 lan truyền mask cho các frame còn lại.

Ví dụ:
    python3 sam_dino/gsam2_video.py video_test2.mp4 \
        --prompt "hand . plastic water bottle . glass cup ." \
        --interval 1.5 --outdir out_gsam2
"""
import argparse
import json
import os
import time

import torch
from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                          Sam2VideoModel, Sam2VideoProcessor)

import common

GDINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_ID = "facebook/sam2.1-hiera-large"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--prompt", required=True,
                   help='danh từ viết thường, ngăn bằng dấu chấm: "hand . cup . knife ."')
    p.add_argument("--interval", type=float, default=1.5, help="giây giữa 2 keyframe")
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--dedup-iou", type=float, default=0.55,
                   help="cùng nhãn, IoU trên ngưỡng này -> coi là trùng, giữ score cao hơn")
    p.add_argument("--dedup-contain", type=float, default=0.80,
                   help="cùng nhãn, box nằm trong box đã giữ quá tỉ lệ này -> bỏ")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--max-side", type=int, default=720, help="thu nhỏ cạnh dài nhất")
    p.add_argument("--outdir", default="out_gsam2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[1/4] đọc video {args.video}")
    frames, fps, (H, W) = common.read_video(args.video, args.max_seconds, args.max_side)
    keyframes = common.keyframe_indices(len(frames), fps, args.interval)
    print(f"      {len(frames)} frame, {fps:.1f} fps, {W}x{H}, {len(keyframes)} keyframe")

    print(f"[2/4] nạp model lên {device}")
    t0 = time.perf_counter()
    gdino_proc = AutoProcessor.from_pretrained(GDINO_ID)
    gdino = GroundingDinoForObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
    sam_proc = Sam2VideoProcessor.from_pretrained(SAM2_ID)
    sam = Sam2VideoModel.from_pretrained(SAM2_ID, dtype=dtype).to(device).eval()
    print(f"      xong sau {time.perf_counter() - t0:.1f}s")

    def detect(frame_rgb, frame_idx):
        """Grounding DINO trên 1 keyframe -> list {box xyxy, label, score}."""
        inputs = gdino_proc(images=frame_rgb, text=args.prompt,
                            return_tensors="pt").to(device)
        with torch.inference_mode():
            out = gdino(**inputs)
        res = gdino_proc.post_process_grounded_object_detection(
            out, inputs["input_ids"],
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(frame_rgb.shape[0], frame_rgb.shape[1])],
        )[0]

        dets = []
        for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
            label = label.strip()
            if not label:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            dets.append({"box": [x0, y0, x1, y1], "label": label,
                         "score": float(score)})

        # khử box trùng, cùng nhãn: xét cả IoU lẫn mức độ bao hàm, giữ score cao hơn
        dets.sort(key=lambda d: -d["score"])
        kept, dropped = [], []
        for d in dets:
            dup = next((k for k in kept
                        if d["label"] == k["label"]
                        and (common.box_iou(d["box"], k["box"]) > args.dedup_iou
                             or common.box_containment(d["box"], k["box"]) > args.dedup_contain)),
                       None)
            if dup is not None:
                dropped.append((d, dup))
                continue
            kept.append(d)
        for d, k in dropped:
            print(f"      bỏ box trùng: {d['label']} {d['score']:.2f} "
                  f"(IoU {common.box_iou(d['box'], k['box']):.2f}, "
                  f"bao hàm {common.box_containment(d['box'], k['box']):.2f} "
                  f"với {k['label']} {k['score']:.2f})")
        return kept

    print(f"[3/4] detect + track, prompt = {args.prompt!r}")
    tracker = common.Sam2ChunkTracker(sam, sam_proc, device, dtype)
    t0 = time.perf_counter()
    per_frame, timing = tracker.run(frames, keyframes, detect)
    timing["total_s"] = round(time.perf_counter() - t0, 1)
    timing["fps_end_to_end"] = round(len(frames) / timing["total_s"], 2)

    print("[4/4] ghi kết quả")
    vid_out = os.path.join(args.outdir, "annotated_gsam2.mp4")
    common.write_video(vid_out, [common.draw_frame(f, o)
                                 for f, o in zip(frames, per_frame)], fps)
    common.dump_masks_json(
        os.path.join(args.outdir, "masks_gsam2.json"),
        {"video": args.video, "fps": fps, "size": [H, W], "prompt": args.prompt,
         "box_threshold": args.box_threshold, "text_threshold": args.text_threshold},
        per_frame, fps, tracker.objects)
    with open(os.path.join(args.outdir, "timing_gsam2.json"), "w") as f:
        json.dump(timing, f, indent=2)

    n_hit = sum(1 for o in per_frame if o)
    print(f"\n  video     : {vid_out}")
    print(f"  object    : {len(tracker.objects)} — " +
          ", ".join(f'#{o["id"]} {o["label"]}' for o in tracker.objects.values()))
    print(f"  phủ frame : {n_hit}/{len(frames)} ({100 * n_hit / len(frames):.0f}%)")
    print(f"  thời gian : detect {timing['detect_ms_per_keyframe']} ms/keyframe, "
          f"track {timing['track_ms_per_frame']} ms/frame, "
          f"tổng {timing['total_s']}s ({timing['fps_end_to_end']} fps)")


if __name__ == "__main__":
    main()
