"""
llm_client.py

Bước 2 của phần suy luận constraint (xem out/bao_cao/suy_luan_constraint_skill.md):
MỘT chỗ duy nhất gọi model qua Ollama, dùng chung cho bước A (VLM nhìn ảnh)
và bước B (LLM suy luận).

    call(model, prompt, schema=..., images=..., think=...) -> dict kết quả

Những gì hàm này lo thay cho nơi gọi:
  - Ép đầu ra đúng JSON Schema (tham số `format` của Ollama) rồi kiểm tra lại
    bằng constraint_schema.validate().
  - Kết quả ổn định giữa các lần chạy: temperature=0, seed cố định.
  - num_ctx mặc định 16384: với ngữ cảnh mặc định của Ollama, thinking mode
    nghĩ hết chỗ (done_reason=length) và trả về nội dung RỖNG (xem time_line.md).
  - Lỗi mạng -> thử lại có giãn cách. Đầu ra sai schema -> gửi lại kèm danh
    sách lỗi để model tự sửa (thử lại y nguyên vô ích vì temperature=0 cho ra
    đúng đầu ra cũ).
  - Cache theo hash đầu vào: chạy lại khi chỉ sửa code bước sau thì đọc kết
    quả cũ, không gọi model lại.
  - Ghi thời gian chạy và số token để so sánh model ở bước 6.

Yêu cầu: Ollama đang chạy (ollama serve) và đã pull model.

Cách dùng (tự kiểm tra với 2 model mặc định):
    python3 llm_client.py --selftest
    python3 llm_client.py --model qwen3.5:9b --prompt "..." --schema constraint --think
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import time

import requests

import constraint_schema as CS

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_VLM = "qwen3-vl:8b-instruct"
DEFAULT_LLM = "qwen3.5:9b"
DEFAULT_NUM_CTX = 16384
DEFAULT_TIMEOUT = 900          # giây; thinking mode mất ~90 s/đoạn skill trên RTX 5060 Ti
DEFAULT_MAX_RETRIES = 3        # lỗi mạng / HTTP
DEFAULT_MAX_REPAIRS = 1        # số lần gửi lại để model tự sửa đầu ra sai schema
DEFAULT_RETRY_BACKOFF = 5      # giây, nhân theo số lần thử
DEFAULT_KEEP_ALIVE = "30m"     # giữ model trong VRAM giữa các đoạn skill
SEED = 0

logger = logging.getLogger("llm_client")


class LLMError(Exception):
    """Lỗi không thể phục hồi khi gọi model."""


def check_ready(model, base_url=DEFAULT_BASE_URL, timeout=5):
    """Fail-fast: Ollama phải đang chạy và model phải có ĐÚNG tên (kể cả tag).

    Khác check_ollama_ready() của vlm_recognize.py ở chỗ so khớp cả tag:
    'qwen3-vl:8b-instruct' không được coi là có chỉ vì máy có 'qwen3-vl:30b'."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"Không kết nối được tới Ollama tại {base_url} ({exc}). "
                       f"Hãy chắc chắn đã chạy 'ollama serve'.") from exc
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in names:
        raise LLMError(f"Chưa có model '{model}' trong Ollama (có: {sorted(names)}). "
                       f"Chạy: ollama pull {model}")


def encode_image(image):
    """Nhận đường dẫn file ảnh hoặc bytes (ảnh đã mã hoá JPEG/PNG) -> base64."""
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        if not os.path.isfile(image):
            raise LLMError(f"Không tìm thấy ảnh: {image}")
        with open(image, "rb") as f:
            data = f.read()
    return base64.b64encode(data).decode()


def _cache_key(payload):
    """Hash toàn bộ những gì ảnh hưởng tới đầu ra (model, prompt, ảnh, schema,
    options). Ảnh được hash theo nội dung nên đổi ảnh là đổi khoá."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()[:32]


def _post(url, payload, timeout, max_retries, backoff):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                # lỗi do request sai (vd model không hỗ trợ think) -> không thử lại
                if 400 <= resp.status_code < 500:
                    raise LLMError(f"Ollama từ chối request (HTTP {resp.status_code}): {resp.text}")
                resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff * attempt
                logger.warning("Gọi Ollama lỗi (lần %d/%d): %s. Thử lại sau %ds...",
                               attempt, max_retries, exc, wait)
                time.sleep(wait)
    raise LLMError(f"Gọi Ollama thất bại sau {max_retries} lần thử: {last_exc}") from last_exc


def _check_output(content, done_reason, schema):
    """Trả về (data, danh_sách_lỗi). Lỗi rỗng nghĩa là dùng được."""
    if done_reason == "length":
        return None, ["Đầu ra bị cắt vì hết ngữ cảnh (done_reason=length); "
                      "tăng num_ctx hoặc rút ngắn prompt"]
    if not content.strip():
        return None, ["Model trả về nội dung rỗng"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        # không yêu cầu JSON -> text thường là hợp lệ
        return None, [f"Không parse được JSON: {exc}"] if schema else []
    return data, CS.validate(data, schema) if schema else []


def call(model, prompt, schema=None, images=None, think=False, system=None,
         base_url=DEFAULT_BASE_URL, num_ctx=DEFAULT_NUM_CTX, temperature=0.0,
         timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES,
         max_repairs=DEFAULT_MAX_REPAIRS, retry_backoff=DEFAULT_RETRY_BACKOFF,
         cache_dir=None):
    """Gọi model một lần (có thể kèm vài lượt tự sửa) và trả về dict:

        data          dict đã parse, đúng schema (None nếu schema=None và không phải JSON)
        raw           nội dung thô cuối cùng model trả
        thinking      phần suy nghĩ (chỉ có khi think=True)
        model, think
        duration_s    tổng thời gian các lượt gọi
        prompt_tokens, eval_tokens   cộng dồn các lượt
        repairs       số lượt phải gửi lại để sửa
        cached        True nếu đọc từ cache

    Ném LLMError nếu sau max_repairs lượt vẫn sai schema."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_msg = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = [encode_image(img) for img in images]
    messages.append(user_msg)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "options": {"temperature": temperature, "seed": SEED, "num_ctx": num_ctx},
    }
    if schema:
        payload["format"] = schema

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{_cache_key(payload)}.json")
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                result = json.load(f)
            result["cached"] = True
            logger.info("Cache hit %s (%s)", os.path.basename(cache_path), model)
            return result

    url = f"{base_url}/api/chat"
    totals = {"duration_s": 0.0, "prompt_tokens": 0, "eval_tokens": 0}
    for repair in range(max_repairs + 1):
        resp = _post(url, payload, timeout, max_retries, retry_backoff)
        msg = resp.get("message", {})
        content = msg.get("content", "")
        totals["duration_s"] += resp.get("total_duration", 0) / 1e9
        totals["prompt_tokens"] += resp.get("prompt_eval_count", 0)
        totals["eval_tokens"] += resp.get("eval_count", 0)

        data, errors = _check_output(content, resp.get("done_reason"), schema)
        if not errors:
            break
        logger.warning("%s: đầu ra lỗi (lượt %d/%d): %s", model, repair + 1,
                       max_repairs + 1, errors[:5])
        if repair == max_repairs or data is None:
            # rỗng / bị cắt / không parse được: gửi lại kèm lỗi cũng không giúp gì,
            # chỉ JSON đúng cú pháp nhưng sai schema mới đáng cho model tự sửa
            raise LLMError(f"{model} trả đầu ra không hợp lệ sau {repair + 1} lượt: {errors[:5]}")
        # cho model xem lại đầu ra của chính nó + danh sách lỗi để sửa
        payload["messages"] = payload["messages"] + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Your JSON has these errors:\n- " + "\n- ".join(errors[:20])
                                        + "\nReturn the corrected JSON only."},
        ]

    result = {
        "data": data,
        "raw": content,
        "thinking": msg.get("thinking") or "",
        "model": model,
        "think": think,
        "duration_s": round(totals["duration_s"], 2),
        "prompt_tokens": totals["prompt_tokens"],
        "eval_tokens": totals["eval_tokens"],
        "repairs": repair,
        "cached": False,
    }
    logger.info("%s%s: %.1fs, %d token vào, %d token ra%s", model, " (think)" if think else "",
                result["duration_s"], result["prompt_tokens"], result["eval_tokens"],
                f", sửa {repair} lượt" if repair else "")
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ---------------------------------------------------------------------------
# Tự kiểm tra
# ---------------------------------------------------------------------------

def _selftest(vlm, llm, image, cache_dir):
    """Chạy 4 ca: VLM có ảnh + schema scene, LLM + schema constraint, gọi lại
    để thử cache, và model không tồn tại phải bị chặn sớm. Trả về số ca hỏng."""
    failed = 0

    def report(name, ok, detail=""):
        nonlocal failed
        failed += 0 if ok else 1
        print(f"[{'OK' if ok else 'LỖI'}] {name} {detail}")

    for m in (vlm, llm):
        check_ready(m)
    report("Ollama sẵn sàng", True, f"({vlm}, {llm})")

    try:
        check_ready("khong-co-model:1b")
        report("chặn model không tồn tại", False)
    except LLMError:
        report("chặn model không tồn tại", True)

    if image:
        r = call(vlm, "Describe the objects in this image as JSON. Give each distinct object an "
                      "integer id starting at 0. Use 'unknown' when you cannot see something.",
                 schema=CS.SCENE_VLM_SCHEMA, images=[image], cache_dir=cache_dir)
        labels = [o["label"] for o in r["data"]["objects"]]
        report("VLM + ảnh + schema scene", True,
               f"{r['duration_s']}s, vật: {labels}, nắp chai: "
               f"{[o['state']['lid'] for o in r['data']['objects']]}")

    dims = "\n".join(f"- {d['name']}: {d['question']}" for d in CS.DIMENSIONS)
    prompt = ("Robot UR3 with 2-finger gripper. Skill: MoveToTarget, holding object 1. "
              "Previous skill: Lift. Next skill: Pour.\nScene: " + json.dumps(CS.EXAMPLE_SCENE_VLM)
              + "\n\nDimensions (answer every one):\n" + dims + "\n\n" + CS.FIELD_GUIDE)
    r1 = call(llm, prompt, schema=CS.CONSTRAINT_SCHEMA, cache_dir=cache_dir)
    st = {k: v["status"] for k, v in r1["data"]["dimensions"].items()}
    report("LLM + schema constraint", True,
           f"{r1['duration_s']}s, target_id={r1['data']['target_id']}, "
           f"{sum(s == 'constraint' for s in st.values())} constraint / "
           f"{sum(s == 'flexible' for s in st.values())} flexible")

    t0 = time.time()
    r2 = call(llm, prompt, schema=CS.CONSTRAINT_SCHEMA, cache_dir=cache_dir)
    report("cache: gọi lại không chạy model", r2["cached"] and r2["data"] == r1["data"],
           f"({time.time() - t0:.3f}s)")
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="chạy các ca tự kiểm tra")
    ap.add_argument("--vlm", default=DEFAULT_VLM)
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--image", help="ảnh dùng cho --selftest hoặc gửi kèm --prompt")
    ap.add_argument("--model", help="model cho --prompt")
    ap.add_argument("--prompt")
    ap.add_argument("--schema", choices=["scene", "constraint"])
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--cache-dir", help="thư mục cache (mặc định không cache)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.selftest:
            return 1 if _selftest(args.vlm, args.llm, args.image, args.cache_dir) else 0
        if not (args.model and args.prompt):
            ap.error("cần --selftest, hoặc --model và --prompt")
        check_ready(args.model)
        schema = {"scene": CS.SCENE_VLM_SCHEMA, "constraint": CS.CONSTRAINT_SCHEMA,
                  None: None}[args.schema]
        r = call(args.model, args.prompt, schema=schema, think=args.think,
                 images=[args.image] if args.image else None, cache_dir=args.cache_dir)
        print(json.dumps(r["data"], ensure_ascii=False, indent=2) if r["data"] is not None
              else r["raw"])
        return 0
    except LLMError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
