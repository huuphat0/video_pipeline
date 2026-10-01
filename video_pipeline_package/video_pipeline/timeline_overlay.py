"""
timeline_overlay.py

HẬU XỬ LÝ (chỉ đọc file, không gọi VLM) — thêm DẢI TIMELINE skill vào bên
dưới video đã annotate của segment_and_overlay_vi_v3.py.

VÌ SAO CẦN: overlay gốc chỉ hiện ĐÚNG 1 skill tại thời điểm đang xem
("Skill: Cầm nắm - chai nước"). Muốn biết cả video gồm những skill nào,
thứ tự ra sao, đoạn nào dài bao lâu thì phải tua hết video. Dải timeline vẽ
TOÀN BỘ chuỗi skill thành các khối màu nằm cạnh nhau, khối đang phát được
tô sáng + có con trỏ chạy theo thời gian — nhìn 1 khung hình là nắm được cả
tiến trình, đúng kiểu dữ liệu tham khảo cho Task Planner.

Script này KHÔNG sửa segment_and_overlay_vi_v3.py: nó đọc lại file log JSON
mà pipeline đã xuất (--log) rồi vẽ thêm, nên chạy lại nhiều lần rất nhanh
(không tốn request VLM nào).

Không có mốc thời gian ở CUỐI video trong log (pipeline chỉ ghi mốc lấy mẫu),
nên mốc cuối cùng được kéo dài tới hết video.

Cách dùng:
    python3 timeline_overlay.py annotated.mp4 --log segments_log.json \\
        --task-log video_task_log.json --out annotated_timeline.mp4 --scale 2
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Màu riêng cho từng skill — nhóm cùng họ hành động dùng tông gần nhau để
# nhìn dải timeline đoán được "đang cầm vật" hay "đang di chuyển" mà không
# cần đọc chữ.
SKILL_COLORS = {
    "Reach":       (120, 144, 156),
    "Grasp":       (46, 125, 50),
    "Release":     (165, 214, 167),
    "Lift":        (2, 136, 209),
    "Place":       (129, 212, 250),
    "MoveToTarget": (25, 118, 210),
    "Pour":        (230, 81, 0),
    "Push":        (255, 143, 0),
    "Pull":        (255, 183, 77),
    "Open":        (106, 27, 154),
    "Close":       (149, 117, 205),
    "Rotate":      (0, 137, 123),
    "Insert":      (216, 27, 96),
    "Remove":      (240, 98, 146),
    "Press":       (255, 193, 7),
    "GoHome":      (84, 110, 122),
    "Unknown":     (97, 97, 97),
}
DEFAULT_COLOR = (97, 97, 97)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def build_intervals(segments, video_duration):
    """Chuyển list mốc rời rạc [(time_sec, skill), ...] thành các KHOẢNG liên
    tục [(start, end, seg), ...] phủ kín cả video — mốc cuối kéo dài tới hết
    video vì log không có mốc kết thúc."""
    out = []
    n = len(segments)
    for i, seg in enumerate(segments):
        start = seg["time_sec"]
        end = segments[i + 1]["time_sec"] if i + 1 < n else video_duration
        if end <= start:      # mốc trùng thời gian -> bỏ qua khối rỗng
            continue
        out.append((start, end, seg))
    return out


def pick_task_name(task_log_path, log_path):
    """Lấy task_name ứng với đúng file log này từ video_task_log.json (file
    tích luỹ nhiều video). Trả "" nếu không tìm thấy."""
    if not task_log_path or not os.path.isfile(task_log_path):
        return ""
    try:
        with open(task_log_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    for rec in reversed(records):   # bản ghi mới nhất của video này
        if os.path.basename(rec.get("source_log", "")) == os.path.basename(log_path):
            return rec.get("task_name", "")
    return records[-1].get("task_name", "") if records else ""


def render_timeline_panel(intervals, duration, cur_time, task_name, width, panel_h,
                          scale, font_scale=1.0):
    """Vẽ panel timeline bằng PIL (cần PIL vì OpenCV không vẽ được tiếng Việt
    có dấu). Trả về ảnh BGR ndarray kích thước (panel_h, width, 3)."""
    img = Image.new("RGB", (width, panel_h), (28, 28, 32))
    draw = ImageDraw.Draw(img)

    f_task = _font(FONT_BOLD, int(19 * scale * font_scale))
    f_skill = _font(FONT_BOLD, int(14 * scale * font_scale))
    f_small = _font(FONT_REG, int(12 * scale * font_scale))

    y_text = int(8 * scale)
    y_bar = int(38 * scale)
    bar_h = int(46 * scale)
    x_pad = int(10 * scale)
    bar_w = width - 2 * x_pad

    # Dòng trên: tên task suy luận được + thời gian hiện tại
    if task_name:
        draw.text((x_pad, y_text), f"Task: {task_name}", font=f_task, fill=(255, 255, 255))
    draw.text((width - x_pad, y_text), f"{cur_time:5.1f}s / {duration:.1f}s",
              font=f_small, fill=(180, 180, 190), anchor="ra")

    if duration <= 0:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def x_of(t):
        return x_pad + int(bar_w * max(0.0, min(t, duration)) / duration)

    # Nền dải + từng khối skill
    draw.rectangle([x_pad, y_bar, x_pad + bar_w, y_bar + bar_h], fill=(55, 55, 62))
    for start, end, seg in intervals:
        x1, x2 = x_of(start), x_of(end)
        if x2 <= x1:
            x2 = x1 + 1
        color = SKILL_COLORS.get(seg["skill_en"], DEFAULT_COLOR)
        playing = start <= cur_time < end
        # Khối đang phát: sáng hơn hẳn để mắt bắt ngay được vị trí hiện tại
        fill = tuple(min(255, int(c * 1.45) + 30) for c in color) if playing else color
        draw.rectangle([x1, y_bar, x2, y_bar + bar_h], fill=fill)

        label = seg.get("skill_vi") or seg["skill_en"]
        # Chỉ ghi chữ khi khối đủ rộng, nếu không chữ sẽ tràn sang khối bên
        # cạnh và gây hiểu sai về ranh giới giữa 2 skill.
        if x2 - x1 >= draw.textlength(label, font=f_skill) + int(10 * scale):
            draw.text(((x1 + x2) // 2, y_bar + bar_h // 2), label, font=f_skill,
                      fill=(255, 255, 255), anchor="mm")

    # Viền khối đang phát cho rõ ràng
    for start, end, seg in intervals:
        if start <= cur_time < end:
            draw.rectangle([x_of(start), y_bar, x_of(end), y_bar + bar_h],
                           outline=(255, 255, 255), width=max(1, int(2 * scale)))
            break

    # Con trỏ thời gian
    xc = x_of(cur_time)
    draw.line([xc, y_bar - int(4 * scale), xc, y_bar + bar_h + int(4 * scale)],
              fill=(255, 235, 59), width=max(1, int(2 * scale)))

    # Trục thời gian
    y_tick = y_bar + bar_h + int(6 * scale)
    for t in range(0, int(duration) + 1, 2):
        xt = x_of(t)
        draw.line([xt, y_tick, xt, y_tick + int(4 * scale)], fill=(150, 150, 160),
                  width=max(1, int(scale)))
        draw.text((xt, y_tick + int(5 * scale)), f"{t}s", font=f_small,
                  fill=(150, 150, 160), anchor="ma")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render(video_path, log_path, out_path, task_log_path=None, scale=2.0):
    with open(log_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not segments:
        raise RuntimeError(f"Log rỗng, không có mốc skill nào: {log_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    task_name = pick_task_name(task_log_path, log_path)
    intervals = build_intervals(segments, duration)

    # 112 = chiều cao tối thiểu để không cắt chữ: dòng task (8->46) + dải
    # skill (38->84) + vạch chia (90->94) + nhãn giây (95->107) + lề dưới.
    panel_h = int(round(112 * scale))
    out_w, out_h = int(round(w * scale)), int(round(h * scale)) + panel_h
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter cho: {out_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        # Video đã có overlay chữ của pipeline; chỉ cần phóng to cho dễ đọc
        # rồi ghép panel timeline xuống dưới.
        big = cv2.resize(frame, (out_w, int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
        panel = render_timeline_panel(intervals, duration, t, task_name, out_w, panel_h, scale)
        writer.write(np.vstack([big, panel]))
        idx += 1

    cap.release()
    writer.release()
    print(f"  fps={fps:.2f}  {total} frame  {duration:.2f}s  ->  {out_w}x{out_h}")
    print(f"  Task: {task_name or '(không có)'}")
    print(f"  Đã ghi: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Thêm dải timeline skill vào dưới video đã annotate")
    parser.add_argument("video_path", help="Video ĐÃ annotate của segment_and_overlay_vi_v3.py")
    parser.add_argument("--log", required=True, help="File log JSON của pipeline (--log)")
    parser.add_argument("--task-log", default=None, help="video_task_log.json (để lấy tên task)")
    parser.add_argument("--out", default="annotated_timeline.mp4")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Hệ số phóng to video gốc (mặc định 2 — video 640x480 thành 1280x960)")
    args = parser.parse_args()

    render(args.video_path, args.log, args.out, args.task_log, args.scale)
    print("\nNếu không phát được, convert bằng ffmpeg:")
    print(f"  ffmpeg -i {args.out} -c:v libx264 -pix_fmt yuv420p {os.path.splitext(args.out)[0]}_h264.mp4")


if __name__ == "__main__":
    main()
