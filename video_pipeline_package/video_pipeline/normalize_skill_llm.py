"""
normalize_skill_llm.py

Sau khi thử nghiệm với sentence embedding (all-MiniLM-L6-v2 và
BAAI/bge-small-en-v1.5), cả hai đều thất bại ở việc phân biệt HÀNH ĐỘNG
khi 2 skill khác nhau chia sẻ cùng DANH TỪ. Ví dụ: "spinning the jar lid"
(đúng: Rotate) bị nhầm thành Close/Open chỉ vì cả 3 đều liên quan tới
"lid" — embedding nhỏ thiên về khớp từ khóa danh từ, không hiểu sâu ngữ
nghĩa động từ. Cả hai model cũng không phân biệt được câu vô nghĩa
("the robot is dancing") với câu có nghĩa.

CÁCH TIẾP CẬN MỚI: dùng chính qwen2.5vl (đã có sẵn, chạy local) ở chế
độ text-only để suy luận ngữ nghĩa, thay vì tính cosine similarity.
LLM 7B có khả năng phân biệt ngữ cảnh hành động tốt hơn nhiều so với
embedding 22-33M tham số.

Cách dùng CLI:
    python3 normalize_skill_llm.py "grab the bottle"
    python3 normalize_skill_llm.py --batch raw_labels.txt
"""

import argparse
import json
import yaml
import requests
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"


@dataclass
class NormalizationResult:
    raw_label: str
    matched_skill: str | None
    confidence: str  # "high" | "medium" | "low"
    reasoning: str
    status: str  # "known" | "review" | "unknown"


class LLMSkillNormalizer:
    def __init__(self, skills_yaml_path):
        with open(skills_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.skills = data["skills"]
        self.skill_names = [s["name"] for s in self.skills]

        # Khớp CHÍNH XÁC vẫn nên làm trước — rẻ, nhanh, không cần gọi LLM
        self.exact_lookup = {}
        for s in self.skills:
            self.exact_lookup[s["name"].lower()] = s["name"]
            for syn in s.get("synonyms", []):
                self.exact_lookup[syn.lower()] = s["name"]

        # Mô tả đầy đủ từng skill để đưa vào prompt cho LLM suy luận
        self._skill_descriptions = "\n".join(
            f"- {s['name']}: {s['description']} (còn gọi là: "
            f"{', '.join(s.get('synonyms', [])) or 'không có'})"
            for s in self.skills
        )

    def normalize(self, raw_label: str) -> NormalizationResult:
        cleaned = raw_label.strip().lower()

        if cleaned in self.exact_lookup:
            return NormalizationResult(
                raw_label=raw_label,
                matched_skill=self.exact_lookup[cleaned],
                confidence="high",
                reasoning="Khớp chính xác với tên hoặc synonym.",
                status="known",
            )

        prompt = f"""Đây là danh sách skill chuẩn trong hệ thống robot:

{self._skill_descriptions}

Một mô tả hành động THÔ (có thể từ video, không chắc dùng đúng thuật ngữ)
là: "{raw_label}"

Nhiệm vụ: xác định mô tả này khớp với skill CHUẨN nào ở trên, dựa vào
Ý NGHĨA HÀNH ĐỘNG (động từ), không chỉ dựa vào từ khóa trùng lặp (ví dụ
câu nhắc tới "lid" không tự động có nghĩa là Open/Close — cần xét xem
hành động THỰC SỰ là gì: xoay, mở, hay đóng).

Nếu mô tả không khớp rõ ràng với skill nào (ví dụ nó không mô tả một
hành động thao tác vật lý thực tế), hoặc quá mơ hồ/vô nghĩa, hãy trả
lời matched_skill là null.

Trả lời DUY NHẤT bằng JSON theo đúng format sau, không thêm chữ nào khác:
{{"matched_skill": "<tên skill hoặc null>", "confidence": "high|medium|low", "reasoning": "<1 câu ngắn giải thích>"}}"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            },
            timeout=120,
        )
        if response.status_code != 200:
            return NormalizationResult(
                raw_label=raw_label, matched_skill=None, confidence="low",
                reasoning=f"Lỗi HTTP {response.status_code}: {response.text}",
                status="unknown",
            )

        raw_text = response.json()["response"].strip()
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            return NormalizationResult(
                raw_label=raw_label, matched_skill=None, confidence="low",
                reasoning=f"Không parse được JSON. Output thô: {raw_text}",
                status="unknown",
            )

        try:
            parsed = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return NormalizationResult(
                raw_label=raw_label, matched_skill=None, confidence="low",
                reasoning=f"JSON lỗi cú pháp. Output thô: {raw_text}",
                status="unknown",
            )

        matched = parsed.get("matched_skill")
        if matched and matched.lower() == "null":
            matched = None
        confidence = parsed.get("confidence", "low")
        reasoning = parsed.get("reasoning", "")

        # Xác thực skill trả về có thực sự nằm trong ontology không
        # (LLM đôi khi "sáng tạo" tên gần đúng nhưng không khớp 100%)
        if matched:
            matched_normalized = None
            for name in self.skill_names:
                if name.lower() == matched.lower():
                    matched_normalized = name
                    break
            matched = matched_normalized

        if matched is None:
            status = "unknown"
        elif confidence == "high":
            status = "known"
        else:
            status = "review"

        return NormalizationResult(
            raw_label=raw_label, matched_skill=matched, confidence=confidence,
            reasoning=reasoning, status=status,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?")
    parser.add_argument("--skills", default="../skill_ontology/skills.yaml")
    parser.add_argument("--batch", help="File .txt, mỗi dòng 1 nhãn thô")
    args = parser.parse_args()

    normalizer = LLMSkillNormalizer(args.skills)

    labels = []
    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
    elif args.label:
        labels = [args.label]
    else:
        print("Cần cung cấp 1 nhãn hoặc dùng --batch file.txt")
        return

    for label in labels:
        result = normalizer.normalize(label)
        status_display = {
            "known": "✓ KNOWN",
            "review": "? REVIEW (khả nghi, cần người duyệt)",
            "unknown": "✗ UNKNOWN (skill mới)",
        }[result.status]
        print(f"'{result.raw_label}' -> {result.matched_skill or '(không rõ)'} "
              f"(confidence={result.confidence})  [{status_display}]")
        print(f"      Lý do: {result.reasoning}")


if __name__ == "__main__":
    main()
