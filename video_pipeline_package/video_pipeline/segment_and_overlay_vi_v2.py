"""
segment_and_overlay_vi_v2.py

BẢN CẢI TIẾN của segment_and_overlay_vi.py — KHÔNG sửa file cũ.

VẤN ĐỀ ĐÃ PHÁT HIỆN: phân loại dựa trên 1 frame tĩnh duy nhất khiến VLM
không thấy được HƯỚNG CHUYỂN ĐỘNG, dễ nhầm giữa các skill trông giống
nhau ở trạng thái tĩnh (vd tay đang "cầm" trông giống "đang nhấc" giống
"đang giữ để đặt xuống"). Frame đầu tiên (chưa có hành động) cũng bị ép
phải chọn bừa 1 skill.

CÁCH SỬA:
1. Dùng cửa sổ trượt 3 frame liên tiếp (trước - hiện tại - sau) mỗi lần
   hỏi VLM, để model thấy được chuyển động thay vì ảnh tĩnh.
2. Cho phép model trả lời "Unknown" khi chưa có hành động rõ ràng
   (frame đầu/cuối khi tay chưa vào khung), thay vì ép chọn bừa.

Cách dùng:
    python3 segment_and_overlay_vi_v2.py video_test0.mp4 \\
        --skills ../skill_ontology/skills.yaml \\
        --names-vi ../skill_ontology/skill_names_vi.yaml \\
        --interval 1.5
"""

import base64
import json
import argparse
import cv2
import requests
import yaml
import numpy as np
from PIL import ImageFont, ImageDraw, Image

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"


def load_skill_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [s["name"] for s in data["skills"]]


def load_vi_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def build_window_prompt(skill_names):
    skill_list = ", ".join(skill_names)
    return f"""Bạn xem 3 khung hình LIÊN TIẾP theo đúng thứ tự thời gian
(khung 1 = trước, khung 2 = hiện tại, khung 3 = sau), trích từ video một
người đang thao tác với vật thể.

Dựa vào SỰ THAY ĐỔI giữa 3 khung hình (không chỉ nhìn 1 khung riêng lẻ),
xác định skill nào đang diễn ra ở khung 2 (khung giữa), CHỈ chọn MỘT
skill trong danh sách sau:

{skill_list}

Nếu cả 3 khung hình không cho thấy hành động rõ ràng (ví dụ tay chưa
xuất hiện, hoặc mọi thứ đứng yên không đổi), trả lời: Unknown

Chỉ trả lời đúng 1 từ là tên skill (tiếng Anh, đúng như trong danh sách),
không giải thích, không thêm chữ nào khác."""


def classify_window(frames_window, skill_names):
    """frames_window: list gồm 1-3 frame (ít hơn 3 ở đầu/cuối video)."""
    prompt = build_window_prompt(skill_names)
    images_b64 = [encode_frame(f) for f in frames_window]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
        },
        timeout=150,
    )
    if response.status_code != 200:
        print(f"LỖI HTTP {response.status_code}: {response.text}")
        return "Unknown"

    raw = response.json()["response"].strip()
    cleaned = raw.strip(".*` \n").split()[0] if raw else "Unknown"

    for name in skill_names:
        if name.lower() == cleaned.lower():
            return name
    return "Unknown"


def put_vietnamese_text(frame, text, org, font_scale=1.0, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, int(30 * font_scale))
    except OSError:
        font = ImageFont.load_default()
    draw.text(org, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_video(video_path, skill_names, vi_names, interval_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(fps * interval_sec))

    # Đọc trước toàn bộ frame ở các mốc sample (video ngắn nên OK về RAM)
    sample_frames = []  # [(timestamp, frame), ...]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            sample_frames.append((idx / fps, frame))
        idx += 1
    cap.release()

    print(f"Đã lấy {len(sample_frames)} mốc thời gian để phân loại "
          f"(cửa sổ 3 frame liên tiếp mỗi mốc).\n")

    segments = []
    labels_at_sample = []
    for i, (ts, frame) in enumerate(sample_frames):
        window = []
        if i > 0:
            window.append(sample_frames[i - 1][1])
        window.append(frame)
        if i < len(sample_frames) - 1:
            window.append(sample_frames[i + 1][1])

        label = classify_window(window, skill_names)
        label_vi = vi_names.get(label, label)
        labels_at_sample.append(label)
        segments.append({"time_sec": round(ts, 1), "skill_en": label, "skill_vi": label_vi})
        print(f"  t={ts:5.1f}s  ->  {label} ({label_vi})")

    # Ghi video overlay: gán nhãn gần nhất theo thời gian cho mỗi frame gốc
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    idx = 0
    sample_idx = 0
    current_label = labels_at_sample[0] if labels_at_sample else "Unknown"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        while (sample_idx + 1 < len(sample_frames)
               and t >= sample_frames[sample_idx + 1][0]):
            sample_idx += 1
            current_label = labels_at_sample[sample_idx]

        label_vi = vi_names.get(current_label, current_label)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        frame = put_vietnamese_text(frame, f"Skill: {label_vi}", (10, 10))
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"\nĐã ghi video overlay vào: {out_path}")
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--skills", default="../skill_ontology/skills.yaml")
    parser.add_argument("--names-vi", default="../skill_ontology/skill_names_vi.yaml")
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--out", default="annotated_output_vi_v2.mp4")
    parser.add_argument("--log", default="segments_log_vi_v2.json")
    args = parser.parse_args()

    skill_names = load_skill_names(args.skills)
    vi_names = load_vi_names(args.names_vi)
    print(f"Skill ontology: {skill_names}\n")

    segments = process_video(args.video_path, skill_names, vi_names, args.interval, args.out)

    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu timeline vào: {args.log}")

    print("\nLƯU Ý: nếu video output không phát được, convert bằng ffmpeg:")
    print(f"  ffmpeg -i {args.out} -c:v libx264 -pix_fmt yuv420p annotated_output_vi_v2_h264.mp4")


if __name__ == "__main__":
    main()
