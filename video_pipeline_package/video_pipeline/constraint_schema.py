"""
constraint_schema.py

Bước 1 của phần suy luận constraint (xem out/bao_cao/suy_luan_constraint_skill.md):
định nghĩa KHUÔN DỮ LIỆU dùng chung cho các bước sau.

    SCENE_VLM_SCHEMA   phần VLM điền ở bước A (chỉ MÔ TẢ cảnh, không suy luận)
    MEASURED_FIELDS    khối số đo từ depth do code ghép vào scene (không qua model)
    DIMENSIONS         checklist 11 chiều bước B BẮT BUỘC xét hết cho mỗi skill
    CONSTRAINT_SCHEMA  đầu ra của bước B cho MỘT đoạn skill
    validate()         kiểm tra một dict theo schema (dùng ở bước 2 và bước 5)

Các schema là JSON Schema dạng dict Python thuần, chỉ dùng tập con mà Ollama
hỗ trợ cho tham số `format` (type, properties, required, enum, items,
additionalProperties) — truyền thẳng vào Ollama để ép model trả đúng khuôn,
và dùng lại chính nó để kiểm tra ở bước 5. Không cần cài jsonschema/pydantic.

Mọi trường có thể không nhìn thấy đều có giá trị "unknown": model được phép
nói "không biết" thay vì bị khuôn ép phải đoán.

QUY ƯỚC `evidence` (bước B trích dẫn, bước 5 tra ngược vào scene):
    obj<ID>.<trường>=<giá trị>      obj1.state.contents=liquid, obj1.symmetry=rotational_vertical
    env.<trường>=<giá trị>          env.clutter=low, env.obstacles_near_path=[]
    rel:<ID> <quan hệ> <ID>         rel:1 left_of 0
    measured.<trường>               measured.tilt_deg.max   (giá trị lấy từ scene khi kiểm tra)
    segment.<trường>=<giá trị>      segment.next=Pour
Dùng ID vật (khớp mask GSAM2), KHÔNG dùng chỉ số mảng, để không nhầm khi
model liệt kê vật theo thứ tự khác.

Chạy trực tiếp để in schema và tự kiểm tra ví dụ mẫu:
    python3 constraint_schema.py
    python3 constraint_schema.py --print constraint
"""

import argparse
import json
import re
import sys

UNKNOWN = "unknown"

# skill không có thao tác -> bỏ qua, không suy luận constraint
SKIP_SKILLS = {"Idle"}

# ---------------------------------------------------------------------------
# Bước A: mô tả cảnh do VLM điền
# ---------------------------------------------------------------------------

LID_STATES = ["open", "closed", "none", UNKNOWN]            # none = vật không có nắp
CONTENTS = ["liquid", "solid", "granular", "empty", UNKNOWN]
FILL_LEVELS = ["full", "half", "low", "empty", UNKNOWN]
MATERIALS = ["plastic", "glass", "ceramic", "metal", "paper", "wood",
             "fabric", "food", "other", UNKNOWN]
YES_NO = ["yes", "no", UNKNOWN]
DEFORMABLE = ["rigid", "slightly", "highly", UNKNOWN]
SHAPES = ["cylinder", "box", "sphere", "bowl", "flat", "irregular", UNKNOWN]
# rotational_vertical: xoay quanh trục dọc không đổi hình dạng (chai, cốc không quai)
SYMMETRY = ["rotational_vertical", "mirror", "none", UNKNOWN]
SUPPORT_SURFACES = ["table", "shelf", "counter", "floor", "other", UNKNOWN]
CLUTTER = ["low", "medium", "high"]
RELATIONS = ["left_of", "right_of", "in_front_of", "behind", "above", "below",
             "inside", "on_top_of", "next_to"]


def _enum(values):
    return {"type": "string", "enum": list(values)}


OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "label": {"type": "string"},
        "state": {
            "type": "object",
            "properties": {
                "lid": _enum(LID_STATES),
                "contents": _enum(CONTENTS),
                "fill_level": _enum(FILL_LEVELS),
            },
            "required": ["lid", "contents", "fill_level"],
            "additionalProperties": False,
        },
        "material": _enum(MATERIALS),
        "fragile": _enum(YES_NO),
        "deformable": _enum(DEFORMABLE),
        "shape": _enum(SHAPES),
        "symmetry": _enum(SYMMETRY),
        # bộ phận cầm được / bộ phận chức năng, vd ["body", "handle"], ["opening", "spout"]
        "graspable_parts": {"type": "array", "items": {"type": "string"}},
        "functional_parts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "label", "state", "material", "fragile", "deformable",
                 "shape", "symmetry", "graspable_parts", "functional_parts"],
    "additionalProperties": False,
}

SCENE_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": OBJECT_SCHEMA},
        "environment": {
            "type": "object",
            "properties": {
                "support_surface": _enum(SUPPORT_SURFACES),
                # ID các vật nằm giữa vật đang cầm và đích
                "obstacles_near_path": {"type": "array", "items": {"type": "integer"}},
                "clutter": _enum(CLUTTER),
            },
            "required": ["support_surface", "obstacles_near_path", "clutter"],
            "additionalProperties": False,
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "integer"},
                    "relation": _enum(RELATIONS),
                    "object": {"type": "integer"},
                },
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["objects", "environment", "relations"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Khối số đo do CODE ghép vào scene ở bước A (nguồn: skill_params.json,
# demo_pose3d.json). Không đưa vào schema của model — model không được điền.
# ---------------------------------------------------------------------------

MEASURED_FIELDS = {
    "height_mm": "độ cao vật so với mặt bàn {start, max, end} (mm)",
    "tilt_deg": "góc trục vật trong hệ camera {min, max, mean, n_frames}; "
                "chỉ tính các frame axis_confident=true",
    "travel_mm": "quãng đường tâm vật đi được trong đoạn (mm, hệ camera)",
    "grasp_part": "phần thân bị cầm, vd 'thân', 'phần dưới'",
    "grasp_along": "vị trí cầm dọc trục vật, 0 = đáy, 1 = đỉnh",
    "approach": "hướng tay tiếp cận {deg, label}",
}

# ---------------------------------------------------------------------------
# Checklist 11 chiều cho bước B
# ---------------------------------------------------------------------------
# question: câu hỏi PHẢN CHỨNG đưa vào prompt (tiếng Anh — model hiểu ổn định hơn)
# measure : khoá trong `measured` dùng để đối chiếu ở bước 5 (None = không đo được)

DIMENSIONS = [
    {"name": "grasp_region",
     "desc_vi": "Cầm vào phần nào của vật",
     "question": "If the gripper grasped a different part of the object, would the task fail or cause harm?",
     "measure": "grasp_along"},
    {"name": "grasp_orientation",
     "desc_vi": "Hướng tiếp cận khi cầm",
     "question": "If the gripper approached the object from a different direction, would the task fail or cause harm?",
     "measure": "approach"},
    {"name": "object_orientation_during_motion",
     "desc_vi": "Tư thế vật khi di chuyển",
     "question": "If the held object were tilted arbitrarily while moving, would the task fail or cause harm?",
     "measure": "tilt_deg"},
    {"name": "path_shape",
     "desc_vi": "Hình dạng quỹ đạo",
     "question": "If the path between start and goal took any shape, would the task fail or cause harm?",
     "measure": None},
    {"name": "path_clearance",
     "desc_vi": "Khoảng cách an toàn với vật khác",
     "question": "If the object passed arbitrarily close to other objects, would the task fail or cause harm?",
     "measure": "height_mm"},
    {"name": "speed",
     "desc_vi": "Tốc độ, gia tốc",
     "question": "If the motion were much faster or slower, would the task fail or cause harm?",
     "measure": None},
    {"name": "end_position",
     "desc_vi": "Vị trí đích",
     "question": "If the object ended at a different position, would the task fail?",
     "measure": "height_mm"},
    {"name": "end_orientation",
     "desc_vi": "Tư thế ở đích",
     "question": "If the object ended in a different orientation, would the task fail?",
     "measure": "tilt_deg"},
    {"name": "contact",
     "desc_vi": "Kiểu tiếp xúc với môi trường (Li & Brock 2026)",
     "question": "Must the object keep a specific contact with the environment (slide on a plane, "
                 "move along a line, rotate about a hinge)? Would breaking that contact make the task fail?",
     "measure": None},
    {"name": "rotation_about_symmetry_axis",
     "desc_vi": "Xoay quanh trục đối xứng của vật",
     "question": "If the object were rotated about its own symmetry axis, would the task fail?",
     "measure": None},
    {"name": "ordering",
     "desc_vi": "Thứ tự với skill trước, sau",
     "question": "Must this skill happen strictly after the previous skill and before the next one?",
     "measure": None},
]
DIMENSION_NAMES = [d["name"] for d in DIMENSIONS]

STATUSES = ["constraint", "flexible", UNKNOWN]
# path: giữ trong suốt skill; subgoal: phải đạt khi kết thúc skill (ReKep, Huang et al. 2024)
PHASES = ["path", "subgoal", "none"]
# 4 kiểu ràng buộc tiếp xúc (Li & Brock 2026); none = chiều không phải `contact`
CONTACT_TYPES = ["free_space", "plane", "prismatic", "revolute", "none", UNKNOWN]

# THỨ TỰ TRƯỜNG CÓ Ý NGHĨA: Ollama sinh JSON đúng theo thứ tự này, nên model
# viết lý do và bằng chứng TRƯỚC rồi mới kết luận `status`. Chạy thử với thứ tự
# ngược lại (status đứng đầu), qwen3.5:9b chốt `constraint` cho gần như mọi chiều
# rồi mới tìm lý do bào chữa, và nhét chuỗi evidence vào `rule`.
DIMENSION_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "status": _enum(STATUSES),
        "phase": _enum(PHASES),
        # luật dạng gần code, có ngưỡng số, vd "tilt(obj1) <= 15deg"; "" khi không phải constraint
        "rule": {"type": "string"},
        "contact_type": _enum(CONTACT_TYPES),
    },
    "required": ["reason", "evidence", "status", "phase", "rule", "contact_type"],
    "additionalProperties": False,
}

# Tất cả 11 chiều đều `required` -> Ollama ép model trả lời đủ checklist.
CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        # ID vật đích (vd cốc khi MoveToTarget/Pour); -1 nếu skill không có đích
        "target_id": {"type": "integer"},
        "dimensions": {
            "type": "object",
            "properties": {name: DIMENSION_RESULT_SCHEMA for name in DIMENSION_NAMES},
            "required": list(DIMENSION_NAMES),
            "additionalProperties": False,
        },
    },
    "required": ["target_id", "dimensions"],
    "additionalProperties": False,
}

NO_TARGET = -1

# Định nghĩa từng trường cho model. Schema chỉ ép KIỂU dữ liệu, không truyền
# được Ý NGHĨA: chạy thử không có đoạn này, model chép câu hỏi vào `rule`, để
# `phase` toàn "none" và lấy vật đang cầm làm `target_id`. Bước 4 đưa nguyên
# văn đoạn này vào prompt.
FIELD_GUIDE = f"""\
Output fields:
- target_id: ID of the object the held object is brought to or acts on (e.g. the cup when
  moving a bottle to pour). NOT the held object. Use {NO_TARGET} if the skill has no target.
- For each dimension, fill the fields in this order:
  - reason: one short sentence answering the dimension's question for THIS scene.
  - evidence: facts from the scene that support the reason, each in one of these forms:
      obj<ID>.<field>=<value>     e.g. obj1.state.contents=liquid
      env.<field>=<value>         e.g. env.obstacles_near_path=[]
      rel:<ID> <relation> <ID>    e.g. rel:1 left_of 0
      measured.<field>            e.g. measured.tilt_deg.max
      segment.<field>=<value>     e.g. segment.next=Pour
    Only cite facts that are actually present in the scene.
  - status: "constraint" if changing this dimension freely would make the task fail or cause
    harm; "flexible" if the robot may choose freely; "unknown" if the scene does not tell.
    Free-space motion (path shape, speed) is flexible unless the scene gives a concrete reason.
  - phase: for a constraint, "path" if it must hold DURING the whole skill, "subgoal" if it
    must hold at the END of the skill. "none" for flexible or unknown.
  - rule: for a constraint, a short checkable expression with a number when possible, e.g.
    "tilt(obj1) <= 15deg", "above(obj1.opening, obj0.opening_top)", "after(Lift)".
    Empty string "" for flexible or unknown. Do not repeat the question here.
  - contact_type: only for the "contact" dimension: free_space, plane (slides on a surface),
    prismatic (moves along a line, e.g. drawer), revolute (rotates about a hinge, e.g. door)
    or unknown. Use "none" for every other dimension.
"""

EVIDENCE_PATTERN = re.compile(
    r"^(obj\d+\.[\w.]+=.+|env\.\w+=.+|rel:\d+ \w+ \d+|measured\.[\w.]+|segment\.\w+=.+)$")

# ---------------------------------------------------------------------------
# Kiểm tra theo schema
# ---------------------------------------------------------------------------

_PY_TYPES = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool}


def validate(value, schema, path="$"):
    """Kiểm tra `value` theo tập con JSON Schema ở trên.

    Trả về danh sách lỗi dạng "đường_dẫn: mô tả"; rỗng nghĩa là hợp lệ.
    Không dừng ở lỗi đầu tiên để bước 5 ghi đủ lý do vào rejected.json."""
    errors = []
    expected = schema.get("type")
    if expected:
        py_type = _PY_TYPES[expected]
        # bool là lớp con của int trong Python -> loại riêng
        if not isinstance(value, py_type) or (expected in ("integer", "number")
                                              and isinstance(value, bool)):
            return [f"{path}: cần kiểu {expected}, nhận {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: '{value}' không thuộc {schema['enum']}")
    if expected == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: thiếu trường '{key}'")
        for key, sub in value.items():
            if key in props:
                errors.extend(validate(sub, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: trường lạ '{key}'")
    if expected == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


# ---------------------------------------------------------------------------
# Ví dụ mẫu (đoạn MoveToTarget 3.39–4.09 s của video demo_20260912) để tự kiểm tra
# ---------------------------------------------------------------------------

def _dim(status, phase="none", rule="", contact_type="none", evidence=(), reason=""):
    return {"reason": reason, "evidence": list(evidence), "status": status,
            "phase": phase, "rule": rule, "contact_type": contact_type}


EXAMPLE_SCENE_VLM = {
    "objects": [
        {"id": 1, "label": "plastic water bottle",
         "state": {"lid": UNKNOWN, "contents": "liquid", "fill_level": "full"},
         "material": "plastic", "fragile": "no", "deformable": "slightly",
         "shape": "cylinder", "symmetry": "rotational_vertical",
         "graspable_parts": ["body"], "functional_parts": ["cap", "opening"]},
        {"id": 0, "label": "white cup",
         "state": {"lid": "none", "contents": "empty", "fill_level": "empty"},
         "material": "ceramic", "fragile": "yes", "deformable": "rigid",
         "shape": "cylinder", "symmetry": "none",
         "graspable_parts": ["handle", "body"], "functional_parts": ["opening_top", "handle"]},
    ],
    "environment": {"support_surface": "table", "obstacles_near_path": [], "clutter": "low"},
    "relations": [{"subject": 1, "relation": "left_of", "object": 0}],
}

EXAMPLE_CONSTRAINT = {
    "target_id": 0,
    "dimensions": {
        "grasp_region": _dim("flexible", evidence=["obj1.graspable_parts=body"],
                             reason="Thân chai trụ đều, cầm chỗ nào trên thân cũng được"),
        "grasp_orientation": _dim("flexible", reason="Đã cầm xong ở skill trước"),
        "object_orientation_during_motion": _dim(
            "constraint", "path", "tilt(obj1) <= 15deg",
            evidence=["obj1.state.contents=liquid", "segment.next=Pour", "measured.tilt_deg.max"],
            reason="Chai có nước và sắp rót nên nhiều khả năng đã mở nắp"),
        "path_shape": _dim("flexible", evidence=["env.obstacles_near_path=[]"],
                           reason="Không có vật cản"),
        "path_clearance": _dim(UNKNOWN, reason="Chưa đủ thông tin khoảng cách"),
        "speed": _dim("flexible", reason="Chai đầy nhưng di chuyển ngắn"),
        "end_position": _dim("constraint", "subgoal", "above(obj1.opening, obj0.opening_top)",
                             evidence=["segment.next=Pour", "rel:1 left_of 0"],
                             reason="Skill sau là rót nên miệng chai phải tới trên miệng cốc"),
        "end_orientation": _dim("flexible", reason="Góc rót thuộc skill Pour"),
        "contact": _dim("flexible", contact_type="free_space",
                        reason="Chai đi trong không gian tự do"),
        "rotation_about_symmetry_axis": _dim("flexible",
                                             evidence=["obj1.symmetry=rotational_vertical"]),
        "ordering": _dim("constraint", "none", "after(Lift) and before(Pour)",
                         evidence=["segment.next=Pour"], reason="Phải nhấc chai lên trước khi mang đi"),
    },
}


def self_test():
    """Kiểm tra schema tự nhất quán và ví dụ mẫu hợp lệ. Trả về số lỗi."""
    n_err = 0
    for name, (value, schema) in {"scene": (EXAMPLE_SCENE_VLM, SCENE_VLM_SCHEMA),
                                  "constraint": (EXAMPLE_CONSTRAINT, CONSTRAINT_SCHEMA)}.items():
        errs = validate(value, schema)
        print(f"[{'OK' if not errs else 'LỖI'}] ví dụ {name}")
        for e in errs:
            print("    ", e)
        n_err += len(errs)
    bad_ev = [ev for d in EXAMPLE_CONSTRAINT["dimensions"].values()
              for ev in d["evidence"] if not EVIDENCE_PATTERN.match(ev)]
    print(f"[{'OK' if not bad_ev else 'LỖI'}] định dạng evidence", bad_ev or "")
    n_err += len(bad_ev)
    # schema phải phát hiện được lỗi thật, không chỉ luôn trả "hợp lệ"
    broken = json.loads(json.dumps(EXAMPLE_CONSTRAINT))
    del broken["dimensions"]["speed"]
    broken["dimensions"]["contact"]["status"] = "maybe"
    caught = validate(broken, CONSTRAINT_SCHEMA)
    ok = len(caught) == 2
    print(f"[{'OK' if ok else 'LỖI'}] bắt được lỗi cố ý: {caught}")
    n_err += 0 if ok else 1
    print(f"{len(DIMENSIONS)} chiều: {', '.join(DIMENSION_NAMES)}")
    return n_err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", choices=["scene", "constraint", "dimensions"],
                    help="in schema/checklist ra màn hình thay vì tự kiểm tra")
    args = ap.parse_args()
    if args.print:
        obj = {"scene": SCENE_VLM_SCHEMA, "constraint": CONSTRAINT_SCHEMA,
               "dimensions": DIMENSIONS}[args.print]
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return 0
    return 1 if self_test() else 0


if __name__ == "__main__":
    sys.exit(main())
