"""
vlm_recognize.py

Đưa các keyframe đã trích vào Qwen2.5-VL (chạy local qua Ollama) để nhận
diện chuỗi skill xuất hiện trong video, dựa trên skill ontology đã định
nghĩa trong skills.yaml.

Yêu cầu: Ollama đang chạy (ollama serve, thường tự chạy nền sau khi cài),
và đã pull model: ollama pull qwen2.5vl:7b

Cách dùng:
    python3 vlm_recognize.py frames/ --skills ../skill_ontology/skills.yaml
"""

import base64
import json
import argparse
import glob
import logging
import os
import sys
import time

import requests
import yaml

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "qwen2.5vl:7b"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 5  # giây, nhân đôi sau mỗi lần thử lại

logger = logging.getLogger("vlm_recognize")


class VLMRecognizeError(Exception):
    """Lỗi không thể phục hồi trong quá trình nhận diện skill."""


def load_skill_names(yaml_path):
    if not os.path.isfile(yaml_path):
        raise VLMRecognizeError(f"Không tìm thấy file skill ontology: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    skills = data.get("skills") if data else None
    if not skills:
        raise VLMRecognizeError(f"File ontology {yaml_path} không có skill nào (key 'skills' rỗng/thiếu).")
    names = [s["name"] for s in skills]
    if len(names) != len(set(names)):
        dup = {n for n in names if names.count(n) > 1}
        logger.warning("Skill ontology có tên trùng lặp: %s", dup)
    return names


def encode_image(path):
    if not os.path.isfile(path):
        raise VLMRecognizeError(f"Không tìm thấy keyframe: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_prompt(skill_names):
    skill_list = ", ".join(skill_names)
    return f"""Bạn đang xem một chuỗi keyframe được trích từ video, theo đúng thứ tự thời gian.
Video này ghi lại một người đang thao tác với vật thể (con người, không phải robot).

Nhiệm vụ: xác định chuỗi skill (hành động) mà người đó thực hiện, CHỈ dùng
các skill trong danh sách sau (không được tạo skill mới, không giải thích thêm):

{skill_list}

Trả lời DUY NHẤT một JSON list theo đúng thứ tự thời gian xuất hiện, ví dụ:
["Reach", "Grasp", "Lift", "MoveToTarget", "Pour"]

Nếu một keyframe không rõ ràng hoặc không khớp skill nào trong danh sách, bỏ qua nó.
Không lặp lại skill liên tiếp nếu nó là cùng một hành động đang diễn ra (chỉ ghi 1 lần)."""


def check_ollama_ready(ollama_url, model_name, timeout=5):
    """Kiểm tra nhanh Ollama đang chạy và model đã được pull, trước khi tốn
    thời gian gửi request nhận diện (fail-fast thay vì đợi hết timeout)."""
    base_url = ollama_url.rsplit("/api/", 1)[0]
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise VLMRecognizeError(
            f"Không kết nối được tới Ollama tại {base_url} ({exc}). "
            f"Hãy chắc chắn đã chạy 'ollama serve'."
        ) from exc

    tags = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(t == model_name or t.startswith(model_name.split(":")[0]) for t in tags):
        logger.warning(
            "Model '%s' chưa thấy trong 'ollama list' (có: %s). "
            "Nếu chưa pull, chạy: ollama pull %s",
            model_name, tags, model_name,
        )


def _post_with_retry(url, payload, timeout, max_retries, backoff):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code != 200:
                logger.error("LỖI HTTP %s từ Ollama. Nội dung phản hồi:\n%s",
                             response.status_code, response.text)
                response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff * attempt
                logger.warning("Request tới Ollama lỗi (lần %d/%d): %s. Thử lại sau %ds...",
                               attempt, max_retries, exc, wait)
                time.sleep(wait)
    raise VLMRecognizeError(f"Gọi Ollama thất bại sau {max_retries} lần thử: {last_exc}") from last_exc


def parse_skill_list(raw_text):
    """Trích JSON list skill từ output thô của VLM (có thể bọc trong markdown
    hoặc kèm text thừa)."""
    raw_text = raw_text.strip()
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1:
        return [], "Không tìm thấy JSON list trong output."

    json_str = raw_text[start : end + 1]
    try:
        skills = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return [], f"Parse JSON lỗi: {exc}"

    if not isinstance(skills, list):
        return [], "JSON trả về không phải list."

    return skills, None


def validate_skills(skills, known_skill_names):
    """Lọc bỏ skill lạ (không có trong ontology) và dedupe skill lặp liên
    tiếp. Trả về (skills_hợp_lệ, danh_sách_skill_lạ_bị_loại)."""
    known_set = set(known_skill_names)
    valid, unknown = [], []
    for s in skills:
        if not isinstance(s, str):
            unknown.append(s)
            continue
        if s not in known_set:
            unknown.append(s)
            continue
        if not valid or valid[-1] != s:
            valid.append(s)
    return valid, unknown


def recognize_skills(frame_paths, skill_names, ollama_url=DEFAULT_OLLAMA_URL,
                      model_name=DEFAULT_MODEL_NAME, timeout=DEFAULT_TIMEOUT,
                      max_retries=DEFAULT_MAX_RETRIES, retry_backoff=DEFAULT_RETRY_BACKOFF):
    images_b64 = [encode_image(p) for p in frame_paths]
    prompt = build_prompt(skill_names)

    response = _post_with_retry(
        ollama_url,
        {
            "model": model_name,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 16384},
        },
        timeout=timeout,
        max_retries=max_retries,
        backoff=retry_backoff,
    )
    raw_text = response.json()["response"]

    skills, error = parse_skill_list(raw_text)
    if error:
        logger.warning("%s Output thô:\n%s", error, raw_text)
        return [], [], raw_text

    valid_skills, unknown_skills = validate_skills(skills, skill_names)
    if unknown_skills:
        logger.warning("VLM trả về %d skill không có trong ontology, đã loại bỏ: %s",
                       len(unknown_skills), unknown_skills)

    return valid_skills, unknown_skills, raw_text


def sample_frames(frame_paths, max_frames):
    if len(frame_paths) <= max_frames:
        return frame_paths
    step = len(frame_paths) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    original_count = len(frame_paths)
    sampled = [frame_paths[i] for i in indices]
    logger.warning("%d keyframe vượt giới hạn %d, đã lấy mẫu đều còn %d frame.",
                   original_count, max_frames, len(sampled))
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", help="Thư mục chứa keyframe (từ extract_frames.py)")
    parser.add_argument("--skills", default="../skill_ontology/skills.yaml")
    parser.add_argument("--out", default="recognition_result.json")
    parser.add_argument("--max-frames", type=int, default=8,
                         help="Số keyframe tối đa đưa vào 1 request (tránh lỗi 400/timeout/OOM)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--skip-health-check", action="store_true",
                         help="Bỏ qua kiểm tra Ollama/model trước khi chạy")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        skill_names = load_skill_names(args.skills)
        logger.info("Skill ontology có %d skill: %s", len(skill_names), skill_names)

        frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
        if not frame_paths:
            raise VLMRecognizeError(f"Không tìm thấy keyframe nào trong {args.frames_dir}")

        frame_paths = sample_frames(frame_paths, args.max_frames)

        if not args.skip_health_check:
            check_ollama_ready(args.ollama_url, args.model)

        logger.info("Đưa %d keyframe vào VLM (model: %s) ...", len(frame_paths), args.model)

        skills, unknown_skills, raw_output = recognize_skills(
            frame_paths, skill_names,
            ollama_url=args.ollama_url, model_name=args.model,
            timeout=args.timeout, max_retries=args.max_retries,
        )
    except VLMRecognizeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print("=" * 50)
    print("Kết quả nhận diện:")
    print(json.dumps(skills, ensure_ascii=False, indent=2))
    print("=" * 50)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "frames": frame_paths,
                "predicted_skills": skills,
                "unknown_skills_filtered": unknown_skills,
                "raw_output": raw_output,
                "model": args.model,
            },
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Đã lưu kết quả vào: %s", args.out)


if __name__ == "__main__":
    main()
