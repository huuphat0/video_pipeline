"""
segment_and_overlay_vi.py

BẢN SAO của segment_and_overlay.py — KHÔNG sửa file gốc. File này thêm
tính năng hiển thị tên skill bằng tiếng Việt trên overlay, dùng mapping
riêng trong skill_names_vi.yaml (skill_ontology/skills.yaml vẫn giữ tên
tiếng Anh làm chuẩn nội bộ, chỉ đổi tên lúc HIỂN THỊ).

Cách dùng:
    python3 segment_and_overlay_vi.py video_test0.mp4 \
        --skills ../skill_ontology/skills.yaml \
        --names-vi ../skill_ontology/skill_names_vi.yaml \
        --interval 1.5

Yêu cầu: Ollama đang chạy, model qwen2.5vl:7b đã pull.
"""

import base64
import json
import argparse
import cv2
import requests
import yaml

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


def build_single_frame_prompt(skill_names):
    skill_list = ", ".join(skill_names)
    return f"""Đây là MỘT khung hình (frame) trích từ video quay một người đang
thao tác với vật thể. Hãy xác định skill (hành động) đang diễn ra trong
khung hình này, CHỈ chọn MỘT skill duy nhất trong danh sách sau:

{skill_list}

Nếu không rõ hoặc không khớp skill nào, trả lời: Unknown

Chỉ trả lời đúng 1 từ là tên skill (tiếng Anh, đúng như trong danh sách),
không giải thích, không thêm chữ nào khác."""


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
    cleaned = raw.strip(".*` \n").split()[0] if raw else "Unknown"

    for name in skill_names:
        if name.lower() == cleaned.lower():
            return name
    return "Unknown"


def put_vietnamese_text(frame, text, org, font_scale=1.0, color=(255, 255, 255)):
    """
    OpenCV cv2.putText mặc định KHÔNG hiển thị đúng dấu tiếng Việt (font
    Hershey không có glyph tiếng Việt). Dùng Pillow để render text tiếng
    Việt chính xác, sau đó ghép ngược lại vào frame OpenCV.
    """
    from PIL import ImageFont, ImageDraw, Image
    import numpy as np

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # DejaVuSans hỗ trợ tiếng Việt và có sẵn trên hầu hết distro Linux.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, int(30 * font_scale))
    except OSError:
        font = ImageFont.load_default()
        print("CẢNH BÁO: không tìm thấy DejaVuSans-Bold.ttf, dùng font mặc định "
              "(có thể hiển thị sai dấu tiếng Việt). Cài bằng: "
              "sudo apt install -y fonts-dejavu-core")

    draw.text(org, text, font=font, fill=color[::-1])  # PIL dùng RGB, color đang là BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_video(video_path, skill_names, vi_names, interval_sec, out_path):
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
    current_label_en = "..."
    segments = []

    print(f"Đang xử lý video ({fps:.1f} fps, {width}x{height}), "
          f"phân loại lại mỗi {interval_sec}s ...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            timestamp = idx / fps
            current_label_en = classify_frame(frame, skill_names)
            label_vi = vi_names.get(current_label_en, current_label_en)
            segments.append({
                "time_sec": round(timestamp, 1),
                "skill_en": current_label_en,
                "skill_vi": label_vi,
            })
            print(f"  t={timestamp:5.1f}s  ->  {current_label_en} ({label_vi})")

        label_vi = vi_names.get(current_label_en, current_label_en)

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
    parser.add_argument("--out", default="annotated_output_vi.mp4")
    parser.add_argument("--log", default="segments_log_vi.json")
    args = parser.parse_args()

    skill_names = load_skill_names(args.skills)
    vi_names = load_vi_names(args.names_vi)
    print(f"Skill ontology: {skill_names}\n")

    segments = process_video(args.video_path, skill_names, vi_names, args.interval, args.out)

    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu timeline vào: {args.log}")

    print("\nLƯU Ý: nếu video output không phát được, convert bằng ffmpeg:")
    print(f"  ffmpeg -i {args.out} -c:v libx264 -pix_fmt yuv420p annotated_output_vi_h264.mp4")


if __name__ == "__main__":
    main()
