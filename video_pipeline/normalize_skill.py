"""
normalize_skill.py

Cơ chế Skill Normalization: so khớp một nhãn skill THÔ (từ VLM, có thể là
tên chuẩn, synonym, hoặc câu tự do như "grabbing the cup") với skill
CHUẨN trong skill_ontology/skills.yaml.

Nguyên tắc thiết kế (đã bàn trong thiết kế tổng thể):
1. Ưu tiên khớp CHÍNH XÁC (case-insensitive) với skill name hoặc synonym
   trước — rẻ, nhanh, không cần model, và đáng tin cậy nhất.
2. Nếu không khớp chính xác, dùng embedding similarity so với
   (description + synonyms) của từng skill.
3. Có 2 ngưỡng similarity:
   - >= HIGH_THRESHOLD  -> tự động chấp nhận, coi là Known Skill.
   - MID_THRESHOLD..HIGH_THRESHOLD -> "khả nghi", cần người duyệt
     (Mục 4 trong thiết kế: tránh tạo skill trùng chỉ vì diễn đạt khác).
   - < MID_THRESHOLD -> Unknown Skill, không tự động gán, không tự
     động tạo skill mới (việc tạo skill mới là bước RIÊNG, có review).

Cách dùng làm module:
    from normalize_skill import SkillNormalizer
    norm = SkillNormalizer("skill_ontology/skills.yaml")
    result = norm.normalize("grabbing the cup")
    # result.status in {"known", "review", "unknown"}

Cách dùng CLI để test nhanh:
    python3 normalize_skill.py "grab the bottle"
    python3 normalize_skill.py --batch raw_labels.txt
"""

import argparse
import yaml
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util

HIGH_THRESHOLD = 0.60   # >= mức này: tự động chấp nhận là Known Skill
MID_THRESHOLD = 0.35    # dưới mức này: Unknown Skill

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # nâng cấp từ all-MiniLM-L6-v2


@dataclass
class NormalizationResult:
    raw_label: str
    matched_skill: str | None
    similarity: float
    status: str  # "known" | "review" | "unknown"


class SkillNormalizer:
    def __init__(self, skills_yaml_path, model_name=EMBED_MODEL_NAME):
        with open(skills_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.skills = data["skills"]
        self.skill_names = [s["name"] for s in self.skills]

        # Bảng tra cứu nhanh cho khớp CHÍNH XÁC: name + mọi synonym -> skill name chuẩn
        self.exact_lookup = {}
        for s in self.skills:
            self.exact_lookup[s["name"].lower()] = s["name"]
            for syn in s.get("synonyms", []):
                self.exact_lookup[syn.lower()] = s["name"]

        # Chuẩn bị embedding: MỖI skill có NHIỀU câu ứng viên ngắn riêng biệt
        # (tên, từng synonym, description) thay vì gộp chung 1 câu dài.
        # Gộp chung làm loãng tín hiệu embedding khi so với nhãn thô ngắn
        # gọn (vd "grab the bottle" so với 1 câu dài chứa cả description
        # + 5 synonym sẽ cho similarity thấp giả tạo). Ở đây ta lấy
        # similarity CAO NHẤT trong số các câu ứng viên của mỗi skill.
        self.model = SentenceTransformer(model_name)

        self._candidate_texts = []      # danh sách phẳng mọi câu ứng viên
        self._candidate_skill_idx = []  # skill index tương ứng mỗi câu ứng viên

        for i, s in enumerate(self.skills):
            candidates = [s["name"], s["description"]]
            candidates.extend(s.get("synonyms", []))
            for c in candidates:
                self._candidate_texts.append(c)
                self._candidate_skill_idx.append(i)

        self._candidate_embeddings = self.model.encode(
            self._candidate_texts, convert_to_tensor=True
        )

    def normalize(self, raw_label: str) -> NormalizationResult:
        cleaned = raw_label.strip().lower()

        # Bước 1: khớp chính xác (name hoặc synonym) — rẻ và đáng tin nhất
        if cleaned in self.exact_lookup:
            return NormalizationResult(
                raw_label=raw_label,
                matched_skill=self.exact_lookup[cleaned],
                similarity=1.0,
                status="known",
            )

        # Bước 2: embedding similarity — so với TỪNG câu ứng viên, lấy
        # điểm cao nhất theo mỗi skill (vd nếu nhãn thô khớp tốt với
        # 1 synonym cụ thể, không bị pha loãng bởi description dài).
        query_emb = self.model.encode(raw_label, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, self._candidate_embeddings)[0]

        best_per_skill = {}  # skill_idx -> best similarity
        for cand_idx, sim in enumerate(sims.tolist()):
            skill_idx = self._candidate_skill_idx[cand_idx]
            if skill_idx not in best_per_skill or sim > best_per_skill[skill_idx]:
                best_per_skill[skill_idx] = sim

        best_idx = max(best_per_skill, key=best_per_skill.get)
        best_sim = best_per_skill[best_idx]
        best_skill = self.skill_names[best_idx]

        if best_sim >= HIGH_THRESHOLD:
            status = "known"
        elif best_sim >= MID_THRESHOLD:
            status = "review"  # khả nghi -> cần người duyệt, không tự tạo skill mới
        else:
            status = "unknown"

        return NormalizationResult(
            raw_label=raw_label,
            matched_skill=best_skill if status != "unknown" else None,
            similarity=round(best_sim, 3),
            status=status,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", help="Nhãn thô cần chuẩn hóa")
    parser.add_argument("--skills", default="../skill_ontology/skills.yaml")
    parser.add_argument("--batch", help="File .txt, mỗi dòng 1 nhãn thô cần test")
    parser.add_argument("--model", default=EMBED_MODEL_NAME,
                         help=f"Tên model embedding (mặc định: {EMBED_MODEL_NAME})")
    parser.add_argument("--debug", action="store_true",
                         help="Hiện top-3 skill có điểm cao nhất, để chẩn đoán ngưỡng")
    args = parser.parse_args()

    normalizer = SkillNormalizer(args.skills, model_name=args.model)

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
              f"(similarity={result.similarity})  [{status_display}]")

        if args.debug:
            query_emb = normalizer.model.encode(label, convert_to_tensor=True)
            sims = util.cos_sim(query_emb, normalizer._candidate_embeddings)[0]
            ranked = sorted(
                zip(normalizer._candidate_texts, normalizer._candidate_skill_idx, sims.tolist()),
                key=lambda x: -x[2],
            )[:5]
            for text, skill_idx, sim in ranked:
                print(f"      [{normalizer.skill_names[skill_idx]}] '{text}' -> {sim:.3f}")


if __name__ == "__main__":
    main()
