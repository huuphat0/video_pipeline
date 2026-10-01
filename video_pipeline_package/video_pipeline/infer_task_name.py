"""
infer_task_name.py

Sau khi Skill Recognition (segment_and_overlay_vi_v2.py) đã tạo ra file
segments_log_vi_v2.json (chuỗi skill theo thời gian), script này đọc
file đó và hỏi LLM suy luận TÊN TASK tổng quát của cả video — vd chuỗi
[Reach, Grasp, Lift, MoveToTarget, Pour] -> "Rót nước từ chai vào cốc".

MỤC ĐÍCH (đã thống nhất — Cách 3/Hướng kết hợp): KHÔNG thay thế Task
Planner luật hiện tại. Tên task + chuỗi skill quan sát được từ video
CHỈ được lưu lại làm DỮ LIỆU THAM KHẢO, để sau này con người xem xét,
phát hiện pattern lặp lại hoặc case chưa được skills.yaml mô tả đúng
(vd task nhiều vật lồng nhau), rồi CHỦ ĐỘNG bổ sung luật thủ công.
Không có cơ chế tự động nào ở đây thay đổi skills.yaml hay Task Planner.

Cách dùng:
    python3 infer_task_name.py segments_log_vi_v2.json \\
        --out video_task_log.json
"""

import argparse
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"


def infer_task_name(skill_sequence, model_name=MODEL_NAME):
    """
    skill_sequence: list các skill theo thời gian, có thể chứa lặp liên
    tiếp (vd cùng 1 skill xuất hiện ở nhiều mốc thời gian sample) — nên
    rút gọn (dedupe liên tiếp) trước khi đưa cho LLM để prompt gọn hơn.
    model_name: cho phép gọi bằng model khác (vd script khác đang dùng
    qwen2.5vl:32b thay vì 7b mặc định) mà không cần sửa module-level.
    """
    deduped = []
    for s in skill_sequence:
        if not deduped or deduped[-1] != s:
            deduped.append(s)

    prompt = f"""Đây là chuỗi skill (hành động) được nhận diện từ 1 video
quay người thao tác vật thể, theo đúng thứ tự thời gian:

{" -> ".join(deduped)}

Nhiệm vụ: đặt 1 TÊN TASK ngắn gọn (1 câu, tiếng Việt) mô tả tổng quát
việc người trong video đang làm — giống cách bạn sẽ ra lệnh cho robot
làm lại đúng việc đó. Ví dụ: "Rót nước từ chai vào cốc", "Mở nắp lọ".

Trả lời DUY NHẤT bằng JSON: {{"task_name": "<tên task>"}}"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        },
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    start, end = raw.find("{"), raw.rfind("}")
    parsed = json.loads(raw[start : end + 1])
    return parsed.get("task_name", "(không xác định)"), deduped


def append_task_log(source_log, skill_sequence, task_name, out_path="video_task_log.json"):
    """Tích luỹ (không ghi đè) 1 bản ghi mới vào out_path — mỗi lần chạy
    trên 1 video mới sẽ thêm 1 record, giữ nguyên lịch sử các video trước."""
    record = {
        "source_log": source_log,
        "skill_sequence": skill_sequence,
        "task_name": task_name,
        "note": ("Dữ liệu tham khảo từ video quan sát được — KHÔNG tự động "
                 "đưa vào Task Planner hay skills.yaml. Con người xem xét "
                 "thủ công để quyết định có cần bổ sung luật hay không."),
    }
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_records = []
    all_records.append(record)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    return len(all_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_log", help="File JSON từ segment_and_overlay_vi_v2.py "
                                              "(vd segments_log_vi_v2.json)")
    parser.add_argument("--out", default="video_task_log.json")
    args = parser.parse_args()

    with open(args.segments_log, "r", encoding="utf-8") as f:
        segments = json.load(f)

    skill_sequence = [seg["skill_en"] for seg in segments]

    task_name, deduped_sequence = infer_task_name(skill_sequence)

    print(f"Chuỗi skill (đã rút gọn): {' -> '.join(deduped_sequence)}")
    print(f"Tên task suy luận được: {task_name}")

    total = append_task_log(args.segments_log, deduped_sequence, task_name, args.out)
    print(f"\nĐã lưu vào: {args.out} (tổng {total} bản ghi tích lũy)")


if __name__ == "__main__":
    main()