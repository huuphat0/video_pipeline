"""
dexycb_to_video.py

Chuyển MỘT sequence DexYCB (RGB-D) thành cặp file mà pipeline đang dùng:
    <out>.mp4        — ảnh màu, để đưa vào gsam2_video.py
    <out>_depth.npz  — depth uint16 mm, CÙNG chỉ số frame với mp4

Nhờ xuất đúng định dạng của depth_from_mcap.py (cùng khoá `depth`/`fps`/`size`,
cùng đơn vị mm, cùng chỉ số frame), pipeline chạy thẳng không cần sửa gì:
    hoi_skill_inference.py masks.json --depth <out>_depth.npz

VÌ SAO CẦN: `depth_from_mcap.py` chỉ đọc ROS2 bag. DexYCB lưu mỗi frame một file
rời (color_######.jpg + aligned_depth_to_color_######.png) chứ không phải bag,
nên phải có bộ chuyển riêng.

LƯU Ý VỀ SỐ FRAME LỆCH: trong thực tế có camera thiếu frame (đo được một camera
có 49 color nhưng 59 depth). Vì vậy script lấy GIAO của hai tập chỉ số frame
chứ không ghép theo thứ tự — nếu ghép theo thứ tự thì ảnh màu và depth sẽ lệch
nhau ngay từ frame thiếu đầu tiên, và mọi phép đo khoảng cách tay-vật sẽ sai.

Cách dùng:
    python3 dexycb_to_video.py <thư_mục_sequence> --camera 840412060917 \\
        --out dexycb_demo.mp4
    python3 dexycb_to_video.py <thư_mục_sequence> --list   # xem các camera
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

DEXYCB_FPS = 30.0


def list_cameras(seq_dir):
    """Các thư mục con là số = serial camera."""
    out = []
    for d in sorted(os.listdir(seq_dir)):
        p = os.path.join(seq_dir, d)
        if os.path.isdir(p) and d.isdigit():
            n_c = len(glob.glob(os.path.join(p, "color_*.jpg")))
            n_d = len(glob.glob(os.path.join(p, "aligned_depth_to_color_*.png")))
            out.append((d, n_c, n_d))
    return out


def _indexed(paths, pattern):
    """{chỉ số frame: đường dẫn}. Bắt chỉ số từ tên file, KHÔNG dùng thứ tự
    sắp xếp chuỗi (color_000010.jpg đứng trước color_000009.jpg nếu so chuỗi)."""
    out = {}
    rx = re.compile(pattern)
    for p in paths:
        m = rx.search(os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def convert(seq_dir, camera, out_mp4, out_npz, max_frames=None):
    cam_dir = os.path.join(seq_dir, camera)
    if not os.path.isdir(cam_dir):
        raise RuntimeError(f"Không thấy camera {camera} trong {seq_dir}")

    colors = _indexed(glob.glob(os.path.join(cam_dir, "color_*.jpg")), r"color_(\d+)\.jpg$")
    depths = _indexed(glob.glob(os.path.join(cam_dir, "aligned_depth_to_color_*.png")),
                      r"aligned_depth_to_color_(\d+)\.png$")
    if not colors:
        raise RuntimeError(f"Không có ảnh màu trong {cam_dir}")
    if not depths:
        raise RuntimeError(f"Không có depth trong {cam_dir}")

    # GIAO của hai tập chỉ số — xem docstring phần "số frame lệch"
    idx = sorted(set(colors) & set(depths))
    if max_frames:
        idx = idx[:max_frames]
    if not idx:
        raise RuntimeError("Ảnh màu và depth không có chỉ số frame nào trùng nhau")

    first = cv2.imread(colors[idx[0]])
    if first is None:
        raise RuntimeError(f"Không đọc được {colors[idx[0]]}")
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             DEXYCB_FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter: {out_mp4}")

    stack = np.zeros((len(idx), h, w), dtype=np.uint16)
    n_ok = 0
    for i, k in enumerate(idx):
        img = cv2.imread(colors[k])
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        d = cv2.imread(depths[k], cv2.IMREAD_UNCHANGED)
        if d is not None:
            if d.shape[:2] != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
            stack[i] = d.astype(np.uint16)
            n_ok += 1
    writer.release()

    np.savez_compressed(out_npz, depth=stack,
                        t=np.arange(len(idx)) / DEXYCB_FPS,
                        fps=DEXYCB_FPS, size=np.array([h, w]),
                        size_color=np.array([h, w]))

    valid = stack[stack > 0]
    print(f"Camera    : {camera}")
    print(f"Frame     : {len(idx)} (color {len(colors)}, depth {len(depths)} "
          f"-> giao {len(idx)})")
    print(f"Thời lượng: {len(idx) / DEXYCB_FPS:.1f}s ở {DEXYCB_FPS:.0f} fps  ({w}x{h})")
    if valid.size:
        print(f"Depth     : {100 * valid.size / stack.size:.1f}% pixel hợp lệ, "
              f"{valid.min()}–{valid.max()} mm, trung vị {np.median(valid):.0f} mm")
    print(f"Đã ghi    : {out_mp4}\n            {out_npz}")
    return out_mp4, out_npz


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq_dir", help="Thư mục sequence, vd .../20200928_155212")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--out", default=None, help="File .mp4 đầu ra")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"Các camera trong {args.seq_dir}:")
        for serial, nc, nd in list_cameras(args.seq_dir):
            print(f"  {serial}  color {nc:>3}  depth {nd:>3}"
                  + ("   <- thiếu frame" if nc != nd else ""))
        return

    if not args.camera:
        raise SystemExit("Cần --camera (xem --list để chọn)")
    out = args.out or os.path.join(os.path.dirname(args.seq_dir.rstrip("/")),
                                   f"{os.path.basename(args.seq_dir.rstrip('/'))}_{args.camera}.mp4")
    base = os.path.splitext(out)[0]
    convert(args.seq_dir, args.camera, out, base + "_depth.npz", args.max_frames)
    print("\nBước tiếp theo:")
    print(f"  cd ../sam_dino && ../../venv/bin/python gsam2_video.py {out} \\")
    print("      --prompt \"hand . scissors . \" --interval 0.8 --outdir ../out/gsam2_dexycb")


if __name__ == "__main__":
    main()
