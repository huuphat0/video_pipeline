"""
segment_and_overlay.py

Khác với vlm_recognize.py (chỉ trả về 1 list skill chung cho cả chuỗi),
script này gán skill cho TỪNG keyframe riêng biệt kèm timestamp, rồi ghi
đè lên video gốc một dòng chữ hiển thị skill đang diễn ra tại mỗi thời
điểm — để xem trực quan trên màn hình, đối chiếu bằng mắt dễ hơn nhiều
so với đọc JSON.

Cách dùng:
    python3 segment_and_overlay.py test_clip.mp4 \
        --skills ../skill_ontology/skills.yaml \
        --interval 1.5 \
        --out annotated_output.mp4

Yêu cầu: Ollama đang chạy, model qwen2.5vl:7b đã pull.
"""

import base64
import json
import argparse
import os
import cv2
import requests
import yaml

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"


def load_skill_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [s["name"] for s in data["skills"]]


def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def build_single_frame_prompt(skill_names):
    skill_list = ", ".join(skill_names)
    return f"""Đây là MỘT khung hình (frame) trích từ video quay một người đang
thao tác với vật thể. Hãy xác định skill (hành động) đang diễn ra trong
khung hình này, CHỈ chọn MỘT skill duy nhất trong danh sách sau:

{skill_list}

Nếu không rõ hoặc không khớp skill nào, trả lời: Unknown

Chỉ trả lời đúng 1 từ là tên skill, không giải thích, không thêm chữ nào khác."""


def classify_frame(frame, skill_names):
    prompt = build_single_frame_prompt(skill_names)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [encode_frame(frame)],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=120,
    )
    if response.status_code != 200:
        print(f"LỖI HTTP {response.status_code}: {response.text}")
        return "Unknown"

    raw = response.json()["response"].strip()
    # Chuẩn hóa: model có thể trả về "Grasp." hoặc "**Grasp**" v.v.
    cleaned = raw.strip(".*` \n").split()[0] if raw else "Unknown"

    for name in skill_names:
        if name.lower() == cleaned.lower():
            return name
    return "Unknown"


def process_video(video_path, skill_names, interval_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(fps * interval_sec))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    idx = 0
    current_label = "..."
    segments = []  # để lưu lại làm log timeline

    print(f"Đang xử lý video ({fps:.1f} fps, {width}x{height}), "
          f"phân loại lại mỗi {interval_sec}s ...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            timestamp = idx / fps
            current_label = classify_frame(frame, skill_names)
            segments.append({"time_sec": round(timestamp, 1), "skill": current_label})
            print(f"  t={timestamp:5.1f}s  ->  {current_label}")

        # Overlay: nền đen mờ + chữ trắng to, góc trên trái
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(
            frame, f"Skill: {current_label}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

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
    parser.add_argument("--interval", type=float, default=1.5,
                         help="Giây giữa mỗi lần phân loại lại (giữ nhãn cũ giữa 2 lần)")
    parser.add_argument("--out", default="annotated_output.mp4")
    parser.add_argument("--log", default="segments_log.json")
    args = parser.parse_args()

    skill_names = load_skill_names(args.skills)
    print(f"Skill ontology: {skill_names}\n")

    segments = process_video(args.video_path, skill_names, args.interval, args.out)

    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu timeline vào: {args.log}")

    print("\nLƯU Ý: video output dùng codec mp4v (OpenCV mặc định), một số trình")
    print("phát có thể không đọc được. Nếu vậy, convert lại bằng ffmpeg:")
    print(f"  ffmpeg -i {args.out} -c:v libx264 -pix_fmt yuv420p annotated_output_h264.mp4")


if __name__ == "__main__":
    main()
