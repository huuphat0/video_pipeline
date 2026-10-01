#!/usr/bin/env python3
"""Nhánh A, bản 1 FRAME — áp điểm gắp ĐÃ HỌC từ video (along/part/side, xem
`contact_point.py::apply_grasp_point`) lên MỘT ẢNH DUY NHẤT chụp lúc robot
chuẩn bị gắp một vật MỚI. Nhẹ hơn `gsam2_video.py` vì không cần SAM2 VIDEO
SESSION lẫn tracking ID giữa các chunk — chỉ 1 lần Grounding DINO detect +
1 lần SAM2 (ảnh, không phải video) segment.

Đây là MẢNH CÒN THIẾU để nối dữ liệu học từ video sang lệnh cho robot:
    video huấn luyện --(skill_params.py grasp_episodes)--> along, part, side
    ảnh lúc gắp thật --(script này)--------------------> pixel + X,Y,Z (hệ camera)

Đã kiểm chứng bằng thật (không đoán API): chạy trên frame 150 của
`demo_color.mp4` với prompt "plastic water bottle .", SAM2 ra mask có bbox
[211,129]-[426,268], `apply_grasp_point(along=0.351)` cho (288.1, 212.2) —
khớp với con số đo được từ nhánh video-tracking đầy đủ ở cùng frame đó
((286.6, 213.8), lệch <2px, hai đường tính khác nhau nên khó trùng tuyệt đối).

Ví dụ:
    python3 gsam2_frame.py anh_robot.jpg \\
        --prompt "plastic water bottle ." --object "water bottle" \\
        --along 0.351 --side "" \\
        --depth-npz ../out/demo_depth.npz --depth-frame-idx 150

    # hoặc lấy nhanh 1 frame từ video có sẵn để thử:
    python3 gsam2_frame.py ../out/demo_color.mp4 --frame-idx 150 \\
        --prompt "plastic water bottle ." --object "water bottle" --along 0.351
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                          Sam2Model, Sam2Processor)

import common

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "github", "video_pipeline"))
import contact_point as CP     # noqa: E402
import hoi_skill_inference as H  # noqa: E402

GDINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_ID = "facebook/sam2.1-hiera-large"


def read_one_frame(path, frame_idx):
    """Ảnh tĩnh (jpg/png) -> đọc thẳng. Video -> lấy đúng frame_idx."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"không đọc được ảnh: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"không mở được video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx or 0)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"không đọc được frame {frame_idx} của {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect(gdino, gdino_proc, frame_rgb, prompt, device,
          box_threshold=0.30, text_threshold=0.25):
    """Giống hệt `detect()` trong gsam2_video.py, bớt phần khử trùng lặp theo
    chunk (1 frame thì không cần) — giữ để nếu 1 nhãn ra 2 box thì báo lỗi rõ
    ràng thay vì âm thầm lấy nhầm."""
    inputs = gdino_proc(images=frame_rgb, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = gdino(**inputs)
    res = gdino_proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold,
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
        dets.append({"box": [x0, y0, x1, y1], "label": label, "score": float(score)})
    return dets


def segment(sam, sam_proc, frame_rgb, box, device):
    """1 box -> mask bool cùng cỡ frame_rgb. SAM2 trả 3 mask ứng viên
    (multimask_output), chọn theo `iou_scores` (điểm SAM2 tự chấm cho mask
    của chính nó) cao nhất — CÙNG quy ước với `Sam2ChunkTracker` bên
    `common.py` (ở đó SAM2 video cũng tự chọn 1 mask/track, không phải 3)."""
    inputs = sam_proc(images=Image.fromarray(frame_rgb), input_boxes=[[box]],
                      return_tensors="pt").to(device)
    with torch.inference_mode():
        out = sam(**inputs, multimask_output=True)
    masks = sam_proc.post_process_masks(out.pred_masks, inputs["original_sizes"])
    best = int(torch.argmax(out.iou_scores[0, 0]))
    return masks[0][0, best].cpu().numpy().astype(bool), float(out.iou_scores[0, 0, best])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="ảnh tĩnh (jpg/png) hoặc video (dùng cùng --frame-idx)")
    ap.add_argument("--frame-idx", type=int, default=0, help="chỉ khi `image` là video")
    ap.add_argument("--prompt", required=True,
                   help='danh từ Grounding DINO, vd "plastic water bottle ."')
    ap.add_argument("--object", default=None,
                   help="nhãn CHÍNH XÁC cần lấy điểm gắp (khớp chuỗi con, không "
                        "phân biệt hoa thường) — bỏ qua nếu prompt chỉ có 1 vật")
    ap.add_argument("--along", type=float, required=True,
                   help="giá trị ĐÃ HỌC từ video khác, xem grasp_episodes()")
    ap.add_argument("--side", default="",
                   help='đã học: số 0..1, hoặc nhãn "bên trái"/"bên phải"/"" (mặc định giữa)')
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--depth-npz", default=None,
                   help="file .npz của depth_from_mcap.py (cần có fx/fy/cx/cy "
                        "để ra X,Y,Z — chạy lại bản mới nếu file cũ chưa có)")
    ap.add_argument("--depth-frame-idx", type=int, default=0,
                   help="frame nào trong --depth-npz khớp với ảnh này")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="ghi kết quả JSON ra file này")
    args = ap.parse_args()

    device = torch.device(args.device)
    frame_rgb = read_one_frame(args.image, args.frame_idx)
    print(f"[1/3] ảnh {frame_rgb.shape[1]}x{frame_rgb.shape[0]}")

    print(f"[2/3] Grounding DINO detect, prompt={args.prompt!r}")
    gdino_proc = AutoProcessor.from_pretrained(GDINO_ID)
    gdino = GroundingDinoForObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
    dets = detect(gdino, gdino_proc, frame_rgb, args.prompt, device,
                  args.box_threshold, args.text_threshold)
    if not dets:
        raise SystemExit("Không phát hiện được vật nào — hạ --box-threshold hoặc sửa --prompt")
    for d in dets:
        print(f"      {d['label']!r}  score={d['score']:.2f}  box={[round(v) for v in d['box']]}")

    target = dets[0]
    if args.object:
        cands = [d for d in dets if args.object.lower() in d["label"].lower()]
        if not cands:
            raise SystemExit(f"Không có detection nào khớp --object {args.object!r} "
                             f"— các nhãn phát hiện được: {[d['label'] for d in dets]}")
        target = max(cands, key=lambda d: d["score"])
    elif len(dets) > 1:
        print(f"      CẢNH BÁO: {len(dets)} vật được phát hiện, mặc định lấy "
              f"{target['label']!r} (score cao nhất) — dùng --object để chọn đúng vật")

    print(f"[3/3] SAM2 segment vật {target['label']!r}")
    sam_proc = Sam2Processor.from_pretrained(SAM2_ID)
    sam = Sam2Model.from_pretrained(SAM2_ID).to(device).eval()
    mask, seg_score = segment(sam, sam_proc, frame_rgb, target["box"], device)
    print(f"      mask {int(mask.sum())} px, điểm SAM2 tự chấm={seg_score:.3f}")

    depth_frame, intr = None, None
    if args.depth_npz:
        depth_stack, _ = H.load_depth(args.depth_npz)
        intr = H.load_intrinsics(args.depth_npz)
        if args.depth_frame_idx < len(depth_stack):
            depth_frame = depth_stack[args.depth_frame_idx]
        if intr is None:
            print("      CẢNH BÁO: file depth không có fx/fy/cx/cy — chỉ ra được pixel")

    result = CP.apply_grasp_point(mask, args.along, side=args.side,
                                  depth_frame=depth_frame, intr=intr)
    if result is None:
        raise SystemExit("Vật quá nhỏ hoặc mask suy biến — không tính được điểm gắp")

    result.update({"object": target["label"], "detect_score": target["score"],
                   "seg_score": seg_score, "learned_along": args.along,
                   "learned_side": args.side})
    print("\nĐiểm gắp áp được:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nĐã ghi: {args.out}")


if __name__ == "__main__":
    main()
