"""
extract_frames.py

Trích keyframe từ 1 video clip theo khoảng thời gian cố định, lưu thành
ảnh JPEG để đưa vào VLM. Không cần trích quá dày — 1 frame mỗi 1-1.5s
là đủ cho action ở tốc độ bình thường.

Cách dùng:
    python3 extract_frames.py test_clip.mp4 --interval 1.5 --out frames/
"""

import cv2
import os
import argparse


def extract_keyframes(video_path, interval_sec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback nếu metadata thiếu

    frame_interval = max(1, int(fps * interval_sec))
    idx = 0
    saved = 0
    saved_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            timestamp = idx / fps
            out_path = os.path.join(out_dir, f"frame_{saved:03d}_t{timestamp:.1f}s.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            saved += 1
        idx += 1

    cap.release()
    print(f"Đã trích {saved} keyframe từ {video_path} -> {out_dir}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--interval", type=float, default=1.5, help="Giây giữa mỗi keyframe")
    parser.add_argument("--out", default="frames", help="Thư mục lưu keyframe")
    args = parser.parse_args()

    extract_keyframes(args.video_path, args.interval, args.out)


if __name__ == "__main__":
    main()
