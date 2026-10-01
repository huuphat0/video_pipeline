"""
segment_and_overlay_vi_v3.py

BẢN CẢI TIẾN của segment_and_overlay_vi_v2.py — KHÔNG sửa file cũ.

VẤN ĐỀ ĐÃ PHÁT HIỆN: v2 chỉ in tên skill chung chung (vd "Grasp"), không
biết đang cầm/chạm vào VẬT THỂ NÀO, nên overlay không đủ chi tiết để
theo dõi (vd không phân biệt được "Grasp - quai cốc" với "Grasp - thìa").

CÁCH SỬA: mỗi cửa sổ 3 frame, hỏi VLM ĐỒNG THỜI 2 việc trong 1 request
(để không tốn thêm 1 lần gọi model):
  1. Skill nào đang diễn ra (như v2).
  2. Tên vật thể / vị trí trên vật thể mà tay đang tương tác (tiếng Việt,
     ngắn gọn, tự do — không giới hạn danh sách cố định, vì vật thể trong
     nhà bếp rất đa dạng).
VLM trả JSON {"skill": "...", "object": "..."} thay vì chỉ 1 từ.

Overlay hiển thị: "Skill: Cầm nắm - quai cốc" thay vì chỉ "Skill: Cầm nắm".

CẬP NHẬT: vị trí chạm/nắm (chỉ hiện lúc skill = Grasp) giờ dùng kiến trúc
hybrid VLM + hand-object interaction thay vì chỉ hỏi VLM đoán bằng chữ
(hay hallucination ra vị trí "điển hình" như "nắp chai" dù tay chưa chạm
tới):
  1. MediaPipe Hands (chạy local, CPU) tìm toạ độ THẬT của 5 đầu ngón tay.
  2. Qwen2.5-VL định vị bounding box của vật thể đang thao tác.
  3. Đo khoảng cách thật (code, không qua VLM) từ ngón tay gần nhất tới
     bbox đó — nếu quá xa, coi là CHƯA chạm và không hiển thị gì cả.
  4. Chỉ khi xác nhận có tiếp xúc, mới cắt cận cảnh + hỏi VLM mô tả bộ
     phận đang chạm (thân/nắp/quai/...).

Yêu cầu thêm: cài mediapipe và tải model hand_landmarker.task (chạy 1 lần):
    pip install mediapipe
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Cách dùng:
    python3 segment_and_overlay_vi_v3.py video_test1.mp4 \\
        --skills ../skill_ontology/skills.yaml \\
        --names-vi ../skill_ontology/skill_names_vi.yaml \\
        --interval 1.5
"""

import base64
import json
import argparse
import os
import re
import time

import cv2
import requests
import yaml
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from infer_task_name import infer_task_name, append_task_log

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

HAND_LANDMARKER_MODEL = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
# Toàn bộ 21 landmark/tay theo chuẩn MediaPipe Hands — không chỉ 5 đầu ngón
# (4,8,12,16,20), mà cả các khớp/đốt ngón (vd 6,7,10,11...) và cổ tay (0).
# Lý do: khi NẮM CHẶT một vật (Grasp), đầu ngón tay thường cong vào che
# khuất/áp sát mặt sau vật thể, camera không thấy được đầu ngón — nhưng
# các khớp ngón/lòng bàn tay vẫn thấy rõ và nằm sát cạnh vật thể. Nếu chỉ
# xét đầu ngón, dễ tính ra khoảng cách lớn giả tạo dù tay đang cầm chắc.
HAND_CONTACT_LANDMARK_IDS = list(range(21))

# Các skill mà việc hiển thị vị trí chạm là có ý nghĩa (tay đang thao tác
# trực tiếp lên 1 điểm cụ thể của vật thể) — không chỉ riêng "Grasp". Vd
# "Close" (vặn đóng nắp) hay "Press" cũng cần biết đang chạm vào đâu,
# nhất là khi model phân loại kỹ hơn (Grasp -> Close khi tay bắt đầu vặn)
# thì vẫn phải tiếp tục hiện đúng vị trí thay vì im lặng bỏ qua.
#
# Bao gồm cả các skill DI CHUYỂN VẬT ĐANG CẦM (Lift, Place, Pour, Push,
# Pull, MoveToTarget): trong suốt các skill này bàn tay VẪN đang nắm vật ở
# đúng 1 điểm cụ thể, nên vị trí chạm vừa có ý nghĩa vừa cần được kiểm
# chứng bằng MediaPipe. Trước đây chúng bị loại khỏi tập này, gây ra 2 lỗi
# thật cùng lúc (đã kiểm chứng trên video_test4, mốc t=9.0s/10.5s "Place
# chai nước"): (1) tầng grounding không chạy nên chuỗi ghi vào log chỉ là
# mô tả tự do chưa kiểm chứng của VLM, lẫn lộn với các mốc đã verify;
# (2) overlay không vẽ vị trí chạm, khiến thao tác chạm nắp chai có thật
# ở cuối video trông như VLM hoàn toàn không nhận ra.
TOUCH_DISPLAY_SKILLS = {"Grasp", "Close", "Open", "Press", "Release", "Insert", "Remove", "Rotate",
                        "Lift", "Place", "Pour", "Push", "Pull", "MoveToTarget"}

_hand_detector = None


def get_hand_detector():
    """Khởi tạo HandLandmarker (MediaPipe) 1 lần, dùng lại cho cả video —
    chạy local trên CPU, không tốn request VLM, độ trễ ~vài chục ms/frame."""
    global _hand_detector
    if _hand_detector is None:
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "Chưa cài mediapipe. Cài bằng: pip install mediapipe, và tải model:\n"
                "curl -L -o hand_landmarker.task https://storage.googleapis.com/"
                "mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        if not os.path.isfile(HAND_LANDMARKER_MODEL):
            raise RuntimeError(f"Thiếu model {HAND_LANDMARKER_MODEL}, xem hướng dẫn tải ở trên.")
        base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL)
        # min_hand_detection_confidence mặc định 0.5 hay MISS hoàn toàn khi
        # tay đang NẮM CHẶT quanh vật thể (hình dạng bàn tay khác nhiều so
        # với tay xoè bình thường mà model quen nhận) — đã kiểm chứng thực
        # tế: 1 tư thế nắm chai thật, conf=0.5 và 0.3 đều trả 0 tay phát
        # hiện được, conf=0.1 mới bắt được, landmark vẫn chính xác. Hạ
        # ngưỡng để ưu tiên KHÔNG bỏ sót tay đang cầm chặt vật, thà chấp
        # nhận thêm vài false positive còn hơn liên tục fallback về mô tả
        # tự do không kiểm chứng của VLM (hay hallucination) mỗi khi tay
        # nắm chặt.
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2,
            min_hand_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        _hand_detector = mp_vision.HandLandmarker.create_from_options(options)
    return _hand_detector


def detect_fingertips_px(frame):
    """Trả về list toạ độ pixel (x, y) của TOÀN BỘ điểm landmark bàn tay
    phát hiện được trong frame (đầu ngón + khớp ngón + lòng bàn tay), dùng
    MediaPipe Hands — chính xác hơn nhiều so với để VLM tự đoán 1 điểm
    chạm, vì đây là model chuyên dụng cho bàn tay."""
    detector = get_hand_detector()
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    points = []
    for hand_landmarks in result.hand_landmarks:
        for idx in HAND_CONTACT_LANDMARK_IDS:
            lm = hand_landmarks[idx]
            points.append((lm.x * w, lm.y * h))
    return points


def detect_fingertips_px_multi(frames):
    """Thử phát hiện tay lần lượt trên NHIỀU frame liền kề (chỉ cách nhau
    vài chục mili-giây, tư thế tay gần như không đổi) thay vì chỉ 1 frame.
    Lý do: đã kiểm chứng thực tế MediaPipe không ổn định 100% — cùng một tư
    thế nắm gần giống hệt nhau, có lúc phát hiện được tay (21 điểm), có lúc
    lại trả về 0 tuỳ frame chính xác nào được đưa vào. Trả về điểm phát
    hiện được ở frame ĐẦU TIÊN thành công, hoặc [] nếu không frame nào ra
    kết quả."""
    for frame in frames:
        points = detect_fingertips_px(frame)
        if points:
            return points
    return []


OLLAMA_URL = "http://localhost:11434/api/generate"
# Model mặc định. Đổi được lúc chạy bằng cờ --model (vd --model qwen3-vl:8b-instruct)
# để so sánh A/B giữa các model mà không phải sửa code.
MODEL_NAME = "qwen2.5vl:32b"

# Đếm số request VLM + tổng thời gian chờ, để báo cáo giây/request ở cuối
# lần chạy (dùng khi so sánh tốc độ giữa các model).
VLM_STATS = {"requests": 0, "seconds": 0.0}


def _post_vlm(payload, timeout):
    """Gọi Ollama và ghi lại thời gian, dùng chung cho mọi request VLM."""
    t0 = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    VLM_STATS["requests"] += 1
    VLM_STATS["seconds"] += time.time() - t0
    return response


def load_skill_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [s["name"] for s in data["skills"]]


def load_skill_descriptions(yaml_path):
    """Đọc description của từng skill trong ontology, dùng để đưa vào prompt
    giúp VLM phân biệt các skill dễ nhầm khi nhìn ảnh tĩnh (vd Lift vs Place:
    tên khác hẳn nhau nhưng hình ảnh tay đang giữ vật thể trông rất giống
    nhau nếu không thấy rõ hướng chuyển động)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {s["name"]: s.get("description", "") for s in data["skills"]}


def load_vi_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def build_window_prompt(skill_names, skill_descriptions=None):
    skill_descriptions = skill_descriptions or {}
    if skill_descriptions:
        skill_list = "\n".join(
            f"   - {name}: {skill_descriptions.get(name, '')}" for name in skill_names
        )
    else:
        skill_list = ", ".join(skill_names)

    return f"""Bạn xem 3 khung hình LIÊN TIẾP theo đúng thứ tự thời gian
(khung 1 = trước, khung 2 = hiện tại, khung 3 = sau), trích từ video một
người đang thao tác với vật thể.

Dựa vào SỰ THAY ĐỔI giữa 3 khung hình (không chỉ nhìn 1 khung riêng lẻ),
xác định 2 thông tin về những gì đang diễn ra ở khung 2 (khung giữa):

1. "skill": CHỈ chọn MỘT trong danh sách sau (tiếng Anh, đúng chính tả).
   Đọc kỹ mô tả để không nhầm các skill trông giống nhau ở ảnh tĩnh —
   đặc biệt CHÚ Ý HƯỚNG CHUYỂN ĐỘNG của vật thể/bàn tay giữa 3 khung:
   nếu vật thể đang đi XUỐNG và tiến gần bề mặt/dừng lại → Place, không
   phải Lift; nếu đang đi LÊN rời khỏi bề mặt → Lift, không phải Place.
{skill_list}
   Nếu không rõ hành động (tay chưa xuất hiện, mọi thứ đứng yên), trả về "Unknown".

2. "object": tên vật thể mà bàn tay đang chạm vào hoặc thao tác, viết
   ngắn gọn bằng TIẾNG VIỆT (vd "cốc", "chai nước", "tủ bếp"). Nếu không
   có vật thể nào đang được thao tác, trả về "".

3. "touch_location": vị trí CỤ THỂ trên vật thể mà bàn tay/ngón tay đang
   chạm/nắm vào (không phải tên vật thể chung chung ở mục 2), viết ngắn
   gọn bằng TIẾNG VIỆT — vd "quai cốc", "nắp chai", "tay nắm tủ", "mép
   trên của bát", "thân cốc". Nếu không thấy rõ vị trí chạm cụ thể, trả
   về "".

Trả lời DUY NHẤT bằng JSON, đúng định dạng, không giải thích thêm:
{{"skill": "<tên skill>", "object": "<tên vật thể tiếng Việt>", "touch_location": "<vị trí chạm tiếng Việt>"}}"""


def classify_window(frames_window, skill_names, skill_descriptions=None):
    """frames_window: list gồm 1-3 frame (ít hơn 3 ở đầu/cuối video).
    Trả về (skill, object_vi, touch_location_vi)."""
    prompt = build_window_prompt(skill_names, skill_descriptions)
    images_b64 = [encode_frame(f) for f in frames_window]

    response = _post_vlm({
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }, timeout=150)
    if response.status_code != 200:
        print(f"LỖI HTTP {response.status_code}: {response.text}")
        return "Unknown", "", ""

    raw = response.json()["response"].strip()
    return parse_skill_object(raw, skill_names)


def parse_skill_object(raw_text, skill_names):
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start == -1 or end == -1:
        # Fallback: model có thể chỉ trả 1 từ skill như v2 (không kèm object)
        cleaned = raw_text.strip(".*` \n").split()[0] if raw_text.strip() else "Unknown"
        for name in skill_names:
            if name.lower() == cleaned.lower():
                return name, "", ""
        return "Unknown", "", ""

    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return "Unknown", "", ""

    skill_raw = str(parsed.get("skill", "Unknown")).strip()
    obj = re.sub(r"\s+", " ", str(parsed.get("object", "") or "").strip())
    touch_location = re.sub(r"\s+", " ", str(parsed.get("touch_location", "") or "").strip())

    for name in skill_names:
        if name.lower() == skill_raw.lower():
            return name, obj, touch_location
    return "Unknown", obj, touch_location


# HỆ TOẠ ĐỘ BBOX KHÁC NHAU GIỮA CÁC THẾ HỆ MODEL — đây là cái bẫy lớn nhất
# khi đổi model, vì cả 2 đều trả JSON "bbox_2d" trông y hệt nhau, không có
# lỗi nào báo ra, chỉ là mọi bbox đều sai chỗ:
#   - Qwen2.5-VL: toạ độ PIXEL tuyệt đối của ảnh đầu vào.
#   - Qwen3-VL:   toạ độ CHUẨN HOÁ trong khung 0-1000 (x/1000 của chiều rộng,
#                 y/1000 của chiều cao) -> phải nhân lại với W/1000, H/1000.
# Đã kiểm chứng thực tế trên frame 1280x720: qwen3-vl trả y=905 và y=825 (lớn
# hơn cả chiều cao ảnh 720 -> không thể là pixel), và sau khi quy đổi thì bbox
# của cốc/bình/vợt/bàn tay đều khớp chính xác với ảnh thật.
NORMALIZED_BBOX_MODEL_PREFIXES = ("qwen3-vl", "qwen3vl")
BBOX_COORD_SCALE = 1000.0


def bbox_uses_normalized_coords(model_name=None):
    name = (model_name or MODEL_NAME).lower()
    return any(name.startswith(pre) for pre in NORMALIZED_BBOX_MODEL_PREFIXES)


def locate_object_bbox_px(frame, object_name):
    """Tìm bounding box (PIXEL của frame) cho vật thể được nêu tên, dùng
    format grounding bbox_2d. Tự quy đổi hệ toạ độ theo model đang dùng
    (xem NORMALIZED_BBOX_MODEL_PREFIXES). Trả về (x1,y1,x2,y2) hoặc None."""
    prompt = f"""Xác định vị trí của "{object_name}" trong ảnh này.

Trả lời DUY NHẤT bằng JSON đúng định dạng sau (toạ độ PIXEL, không chuẩn
hoá), KHÔNG giải thích, KHÔNG bọc markdown:
[{{"bbox_2d": [x1, y1, x2, y2], "label": "{object_name}"}}]"""
    response = _post_vlm({
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [encode_frame(frame)],
        "stream": False,
        "options": {"temperature": 0.1},
    }, timeout=60)
    if response.status_code != 200:
        return None
    raw = response.json()["response"].strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not parsed or not isinstance(parsed, list) or not isinstance(parsed[0], dict):
        return None
    box = parsed[0].get("bbox_2d")
    if not (isinstance(box, list) and len(box) == 4):
        return None
    try:
        coords = tuple(float(v) for v in box)
    except (TypeError, ValueError):
        return None

    if bbox_uses_normalized_coords():
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = coords
        coords = (x1 * w / BBOX_COORD_SCALE, y1 * h / BBOX_COORD_SCALE,
                  x2 * w / BBOX_COORD_SCALE, y2 * h / BBOX_COORD_SCALE)
    return coords


def closest_point_to_bbox(points, bbox):
    """Trong các điểm đầu ngón tay, trả về điểm gần bbox vật thể nhất và
    khoảng cách (pixel) từ điểm đó tới bbox (0 nếu điểm nằm trong bbox)."""
    if not points or bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    best_point, best_dist = None, None
    for (x, y) in points:
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_point, best_dist = (x, y), dist
    return best_point, best_dist


def hand_region_bbox(points, pad=15):
    """Bbox bao trọn toàn bộ landmark bàn tay (không phải 1 điểm gần nhất
    duy nhất). Lý do: khi ngón tay cuộn/che khuất quanh vật nhỏ (vd nắp
    chai), MediaPipe vẫn phát hiện được TAY nhưng có thể đặt SAI vị trí
    từng khớp riêng lẻ (điểm gần bbox nhất tình cờ rơi lệch chỗ) — dùng cả
    VÙNG bàn tay để so chồng lấn với bbox từng bộ phận đáng tin hơn nhiều
    so với chỉ 1 điểm gần nhất bị lệch."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def bbox_overlap_ratio(bbox_a, bbox_b):
    """Tỉ lệ diện tích giao nhau so với diện tích bbox_b (0-1)."""
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / b_area


# Vật thể không có quai (chai/lọ/ly thuỷ tinh...) hay bị VLM hallucination
# ra "quai" vì đó là bộ phận "quen thuộc" của cốc/ca — giới hạn theo từ
# khoá trong tên vật thể để chọn đúng bộ nhận diện bộ phận đặc trưng.
NO_HANDLE_KEYWORDS = ["chai", "lọ", "bottle", "hộp", "jar", "can"]
HANDLE_KEYWORDS = ["cốc", "ca", "ấm", "nồi", "cup", "mug", "kettle", "pot"]

# Mỗi LOẠI vật thể có các điểm chạm đặc trưng RIÊNG — cốc có quai + miệng
# (không có nắp/đáy đáng chạm), chai có nắp + thân + đáy (không có quai).
# Dùng để hỏi VLM định vị TRỰC TIẾP bbox của từng bộ phận này, thay vì suy
# luận tỉ lệ trên 1 bbox tổng của cả vật thể (đã kiểm chứng: bbox tổng
# thường bị cắt cụt ở đầu bị tay che hoặc ở cổ chai, không bao gồm nắp,
# khiến suy luận theo tỉ lệ chiều cao bị sai khi tay cầm đúng chỗ nắp).
# "thân" KHÔNG nằm trong các danh sách này — nó là bộ phận MẶC ĐỊNH/còn
# lại, không đưa vào vòng so khoảng cách. Lý do: bbox của "thân" luôn bao
# trùm gần hết bbox tổng của vật thể (thân chiếm phần lớn diện tích), nên
# nếu so cùng các bộ phận khác, "thân" sẽ luôn thắng do khoảng cách ≈0 bất
# kể điểm chạm thực tế ở đâu — làm mất tác dụng phân biệt nắp/quai/đáy.
# "đáy" CHỦ ĐỘNG bỏ khỏi danh sách: khi tay nắm giữa thân chai, bbox vật
# thể do VLM trả về luôn bị cắt cụt ngay tại chỗ tay bắt đầu che (không
# thấy được phần dưới thật) — nên điểm chạm luôn rơi gần "đáy nhìn thấy
# được" của bbox bị cắt cụt đó, kể cả khi tay đang cầm ở thân (đã kiểm
# chứng: false positive "đáy" liên tục trong nhiều mốc thực ra là "thân").
# "nắp" thì ngược lại luôn lộ rõ, hiếm khi bị che, nên đáng tin cậy hơn
# nhiều để giữ lại tự động nhận diện.
BOTTLE_TOUCH_PARTS = ["nắp"]
CUP_TOUCH_PARTS = ["quai", "miệng"]
GENERIC_TOUCH_PARTS = []


def touch_parts_for_object(object_name):
    name = (object_name or "").lower()
    if any(kw in name for kw in HANDLE_KEYWORDS) and not any(kw in name for kw in NO_HANDLE_KEYWORDS):
        return CUP_TOUCH_PARTS
    if any(kw in name for kw in NO_HANDLE_KEYWORDS):
        return BOTTLE_TOUCH_PARTS
    return GENERIC_TOUCH_PARTS


def classify_touch_region_grounded(frame, point, object_name, whole_bbox, hand_points=None,
                                    part_margin_ratio=0.35, min_overlap_ratio=0.15):
    """Xác định bộ phận đang chạm bằng cách hỏi VLM định vị TRỰC TIẾP bbox
    của từng bộ phận đặc trưng của LOẠI vật thể (touch_parts_for_object).

    Ưu tiên 2 tín hiệu, dùng cái nào có trước:
    1. CHỒNG LẤN vùng bàn tay (hand_region_bbox từ TOÀN BỘ landmark, không
       phải 1 điểm) với bbox từng bộ phận — đáng tin hơn nhiều so với chỉ
       1 điểm gần nhất khi ngón tay cuộn/che khuất quanh vật nhỏ (đã kiểm
       chứng: MediaPipe phát hiện được tay nhưng đặt SAI vị trí từng khớp
       riêng lẻ do occlusion, khiến "điểm gần nhất" bị lệch khỏi vùng tay
       thật, trong khi cả VÙNG tay vẫn bao đúng bộ phận đang cầm).
    2. Nếu không có overlap đáng kể, fallback về khoảng cách từ điểm chạm
       (closest_point_to_bbox) tới bbox bộ phận — dùng cho trường hợp tay
       chạm nhẹ/chưa cuộn kín.
    Nếu không bộ phận đặc trưng nào đủ gần/chồng lấn, mặc định trả "thân".
    """
    parts = touch_parts_for_object(object_name)
    if not parts:
        return "thân"

    x1, y1, x2, y2 = whole_bbox
    whole_area = max(1.0, (x2 - x1) * (y2 - y1))
    max_dim = max(x2 - x1, y2 - y1)
    margin = max(20, part_margin_ratio * max_dim)
    hand_bbox = hand_region_bbox(hand_points) if hand_points else None

    best_overlap_part, best_overlap = None, 0.0
    best_dist_part, best_dist = None, None
    for part in parts:
        part_bbox = locate_object_bbox_px(frame, f"{part} của {object_name}")
        if part_bbox is None:
            continue
        px1, py1, px2, py2 = part_bbox
        part_area = max(0.0, (px2 - px1) * (py2 - py1))
        # Nếu bbox "bộ phận" chiếm gần hết bbox TOÀN bộ vật thể, nghĩa là
        # VLM không định vị được riêng bộ phận đó mà chỉ trả lại nguyên
        # vật thể (đã kiểm chứng thực tế: hỏi "đáy của chai" có lúc trả về
        # bbox gần như trùng khít cả chai) — loại bỏ, không đáng tin.
        if part_area / whole_area > 0.6:
            continue

        if hand_bbox is not None:
            overlap = bbox_overlap_ratio(hand_bbox, part_bbox)
            if overlap > best_overlap:
                best_overlap_part, best_overlap = part, overlap

        _, dist = closest_point_to_bbox([point], part_bbox)
        if dist is not None and (best_dist is None or dist < best_dist):
            best_dist_part, best_dist = part, dist

    if best_overlap_part is not None and best_overlap >= min_overlap_ratio:
        return best_overlap_part
    if best_dist_part is not None and best_dist <= margin:
        return best_dist_part
    return "thân"


def describe_touch_location_grounded(frames, object_name, fallback_text="", min_margin_px=25, margin_ratio=0.12):
    """Quy trình hybrid VLM + hand-object interaction để giảm hallucination
    khi mô tả vị trí chạm:
    1. MediaPipe Hands tìm toạ độ pixel THẬT của TOÀN BỘ landmark bàn tay
       — model chuyên dụng cho bàn tay, chính xác hơn nhiều so với để VLM
       tự đoán 1 điểm chạm. `frames` là 1 frame chính + vài frame liền kề
       ngay sau đó (~vài chục ms, tư thế gần như không đổi) — thử lần lượt
       (detect_fingertips_px_multi) vì đã kiểm chứng MediaPipe không ổn
       định 100%, cùng 1 tư thế nắm có lúc phát hiện được có lúc không.
    2. Qwen2.5-VL định vị bounding box của vật thể (locate_object_bbox_px)
       trên frame đầu tiên (frames[0]).
    3. Đo khoảng cách THẬT (bằng code, không qua VLM) từ landmark gần nhất
       tới bbox đó. Ngưỡng "coi là chạm" CO GIÃN theo kích thước vật thể
       (margin_ratio * kích thước bbox, tối thiểu min_margin_px). Nếu vẫn
       quá xa, coi là CHƯA chạm — trả rỗng.
    4. Nếu MediaPipe HOÀN TOÀN không thấy tay ở BẤT KỲ frame nào trong
       burst (occlusion nặng) — dùng tạm mô tả tự do (fallback_text) từ
       bước classify_window thay vì im lặng giấu luôn thông tin.
    5. Khi xác nhận có tiếp xúc thật, phân loại bộ phận đang chạm bằng
       HÌNH HỌC (classify_touch_region) thay vì hỏi VLM mô tả bằng chữ.
    Trả về chuỗi mô tả tiếng Việt, hoặc "" nếu chưa thực sự chạm tới.
    Trả về (text, verified) — verified=True nghĩa là MediaPipe đã xác thực
    được (kể cả khi kết quả là rỗng do xác nhận chưa chạm), verified=False
    nghĩa là MediaPipe hoàn toàn không thấy tay, text lúc đó chỉ là
    fallback_text (chưa kiểm chứng, caller có thể chọn không tin).
    """
    fingertips = detect_fingertips_px_multi(frames)
    if not fingertips:
        # Không thấy tay ở frame nào trong burst -> không thể kiểm chứng
        # bằng MediaPipe, dùng tạm mô tả tự do thay vì giấu hẳn thông tin.
        return fallback_text, False

    bbox = locate_object_bbox_px(frames[0], object_name) if object_name else None
    if bbox is None:
        # VLM không định vị được bbox vật thể (vd ảnh mờ do chuyển động,
        # vật bị che một phần) dù MediaPipe vẫn thấy tay -> không thể kiểm
        # chứng bằng hình học, nhưng KHÔNG được giấu hẳn thông tin: dùng
        # tạm fallback_text như nhánh "không thấy tay" ở trên, để nhất
        # quán (đã từng là bug thật: trả "" làm mất luôn vị trí chạm ở
        # đúng lúc vừa Grasp một vật mới, ảnh dễ bị mờ/chuyển động nhất).
        return fallback_text, False

    x1, y1, x2, y2 = bbox
    contact_margin_px = max(min_margin_px, margin_ratio * max(x2 - x1, y2 - y1))

    point, dist = closest_point_to_bbox(fingertips, bbox)
    if point is None or dist > contact_margin_px:
        return "", True  # tay còn cách xa vật thể -> đã kiểm chứng CHƯA chạm

    return classify_touch_region_grounded(frames[0], point, object_name, bbox, hand_points=fingertips), True


def smooth_skill_labels(labels_at_sample):
    """Khử nhiễu: nếu 1 mốc bị phân loại khác hẳn 2 mốc liền kề trước/sau
    (vd Reach, Push, Grasp -> mốc "Push" ở giữa là nhiễu 1 khung hình lẻ),
    thay bằng skill của mốc trước đó — để tên skill hiển thị ổn định,
    không nhấp nháy sai giữa hành động cho đến khi thực sự đổi hành động
    (được xác nhận bởi ít nhất 2 mốc liên tiếp cùng skill mới)."""
    n = len(labels_at_sample)
    if n < 3:
        return labels_at_sample

    smoothed = list(labels_at_sample)
    for i in range(1, n - 1):
        prev_skill = smoothed[i - 1][0]
        cur_skill = smoothed[i][0]
        next_skill = labels_at_sample[i + 1][0]
        if cur_skill != prev_skill and cur_skill != next_skill and prev_skill == next_skill:
            skill, obj, touch = smoothed[i]
            smoothed[i] = (prev_skill, obj, touch)
    return smoothed


# Các skill thuộc CÙNG 1 chuỗi thao tác liên tục trên MỘT vật thể đang
# cầm trên tay — về mặt vật lý, vật thể không thể tự đổi tên giữa chừng
# (đang Pour "chai nước" thì không thể đột nhiên Place "nắp chai", vì
# nắp chai không có trong tay lúc đó). Có cả "Reach" vì tiến lại gần rồi
# Grasp thường là tiếp cận CHÍNH vật thể sắp cầm — nếu ngắt chuỗi ở Reach,
# tên vật thể dễ "trôi" theo từ đồng nghĩa VLM chọn khác nhau mỗi lần
# (đã kiểm chứng: cùng 1 cái bình thuỷ tinh bị gọi luân phiên "cốc" ->
# "cốc thuỷ tinh" -> "ly" qua các mốc Reach/Grasp xen kẽ).
HOLDING_SKILLS = {"Reach", "Grasp", "Lift", "MoveToTarget", "Pour", "Rotate", "Place", "Release", "Close", "Open"}

# Các nhóm từ đồng nghĩa tiếng Việt cho cùng 1 LOẠI vật thể — VLM không ổn
# định trong cách gọi tên giữa các lần suy luận độc lập (mỗi mốc là 1
# request riêng, không nhớ đã gọi tên gì ở mốc trước), nên cùng 1 vật thể
# vật lý có thể bị đổi qua lại giữa các từ đồng nghĩa suốt NHIỀU mốc liên
# tiếp — không phải nhiễu 1 khung hình lẻ nên khử nhiễu lân cận thông
# thường không bắt được. Mỗi nhóm có 1 TÊN CHUẨN (phần tử đầu) — chuẩn hoá
# ÁP DỤNG TOÀN CỤC cho mọi mốc trong cả video (không chỉ trong 1 chuỗi
# thao tác liên tục), vì cùng 1 vật thể có thể được cầm lại nhiều lần sau
# khi tay đã chuyển qua thao tác trên vật khác ở giữa (vd cầm cốc -> rót
# từ chai -> cầm lại cốc lúc sau vẫn phải gọi đúng tên như lúc đầu).
# QUAN TRỌNG: "cốc" (cốc đục, sứ/nhựa) và "ly"/"cốc thuỷ tinh" (ly/bình
# thuỷ tinh TRONG SUỐT) KHÔNG được gộp chung 1 nhóm — đã kiểm chứng thực
# tế trên video có CẢ 2 vật thể này CÙNG XUẤT HIỆN (1 cái cốc sứ + 1 cái
# bình thuỷ tinh có quai, khác nhau hoàn toàn), gộp chung sẽ xoá mất phân
# biệt thật giữa 2 vật thể (VLM dùng "cốc" NHẤT QUÁN cho cốc sứ, và
# "ly"/"cốc thuỷ tinh" NHẤT QUÁN cho bình thuỷ tinh — không hề lẫn lộn
# giữa 2 vật, chỉ lẫn lộn CÁCH GỌI TÊN trong nội bộ từng vật).
OBJECT_SYNONYM_GROUPS = [
    ["cốc", "cốc nước", "cốc uống nước", "ca", "ca nước"],
    ["ly", "cốc thủy tinh", "ly thủy tinh", "ly nước", "bình thủy tinh"],
    ["chai nước", "chai", "bình nước", "chai nhựa"],
    ["bát", "tô", "chén"],
    ["nồi", "ấm", "ấm nước"],
]

_OBJECT_SYNONYM_LOOKUP = {
    alias.lower(): group[0] for group in OBJECT_SYNONYM_GROUPS for alias in group
}


def normalize_object_synonyms(labels_at_sample):
    """Chuẩn hoá tên vật thể về TÊN CHUẨN của nhóm đồng nghĩa (nếu có),
    áp dụng ĐỘC LẬP cho từng mốc — không cần biết mốc trước/sau, nên vẫn
    đúng kể cả khi 1 vật thể bị "bỏ dở giữa chừng" rồi cầm lại sau khi đã
    thao tác trên vật khác ở giữa (chuỗi liên tục đã bị ngắt)."""
    normalized = []
    for skill, obj, touch in labels_at_sample:
        canonical = _OBJECT_SYNONYM_LOOKUP.get((obj or "").strip().lower())
        normalized.append((skill, canonical if canonical else obj, touch))
    return normalized


def find_synonym_group(name):
    name_l = (name or "").strip().lower()
    for group in OBJECT_SYNONYM_GROUPS:
        if name_l in (alias.lower() for alias in group):
            return group
    return None


def smooth_isolated_object_noise(labels_at_sample):
    """Khử nhiễu tên vật thể ở MỐC ĐẦU TIÊN: nếu mốc đầu bị đặt tên vật
    thể khác hẳn mốc kế tiếp — coi là nhiễu 1 khung hình lẻ (vd VLM bịa ra
    "bóng" ở khung đầu tiên dù ảnh chỉ có bàn tay và chai nước, không có
    quả bóng nào) và thay bằng tên vật thể của mốc kế tiếp.
    CHỈ áp dụng cho mốc đầu (không áp dụng mốc cuối): khác với mốc đầu
    (thường đang tiếp cận 1 vật đã xác định ngay sau đó), mốc CUỐI hợp lệ
    khi đổi sang vật thể khác (vd Reach sang vật kế tiếp) — ép mốc cuối
    khớp mốc trước có thể lan truyền ngược lỗi nếu chính mốc trước đó mới
    là mốc bị sai (đã kiểm chứng: gây lỗi thật khi áp dụng đối xứng)."""
    n = len(labels_at_sample)
    if n < 2:
        return labels_at_sample

    smoothed = list(labels_at_sample)
    skill0, obj0, touch0 = smoothed[0]
    next_obj = labels_at_sample[1][1]
    if next_obj and obj0 != next_obj:
        smoothed[0] = (skill0, next_obj, touch0)

    for i in range(1, n - 1):
        skill, obj, touch = smoothed[i]
        prev_obj = labels_at_sample[i - 1][1]
        next_obj = labels_at_sample[i + 1][1]
        if obj != prev_obj and obj != next_obj and prev_obj == next_obj and prev_obj:
            smoothed[i] = (skill, prev_obj, touch)
    return smoothed


def _bridge_chain_reversions(labels_at_sample):
    """Vá lỗi hallucination BỀN qua NHIỀU mốc liên tiếp (>1 mốc, nên vượt
    qua được ngưỡng xác nhận 1-mốc của smooth_object_labels): nếu trong
    CÙNG 1 chuỗi giữ-vật liên tục (không bị ngắt bởi skill không giữ vật),
    tên vật thể đổi khác đi rồi sau đó QUAY LẠI đúng tên ban đầu, thì toàn
    bộ đoạn ở giữa chắc chắn là nhiễu (vật lý không thể đổi tên rồi tự đổi
    lại thành tên cũ giữa 1 thao tác cầm liên tục) — gán lại theo tên ban
    đầu, bất kể đoạn nhiễu dài bao nhiêu mốc.
    Vd đã kiểm chứng: cầm 1 bình thủy tinh liên tục rót nước, bị gọi
    "chai nước" -> "ly" (4 mốc liền) -> "chai nước" — đoạn "ly" ở giữa
    được vá lại thành "chai nước"."""
    n = len(labels_at_sample)
    if n < 3:
        return labels_at_sample

    smoothed = list(labels_at_sample)
    i = 0
    while i < n:
        if smoothed[i][0] not in HOLDING_SKILLS:
            i += 1
            continue
        chain_start = i
        while i < n and smoothed[i][0] in HOLDING_SKILLS:
            i += 1
        chain_end = i  # exclusive

        held_obj = smoothed[chain_start][1]
        k = chain_start + 1
        while k < chain_end:
            if smoothed[k][1] != held_obj:
                run_end = k
                while run_end < chain_end and smoothed[run_end][1] != held_obj:
                    run_end += 1
                if run_end < chain_end:
                    # Tìm lại đúng tên ban đầu trong cùng chuỗi -> đoạn
                    # giữa [k, run_end) là nhiễu, vá lại theo held_obj.
                    for m in range(k, run_end):
                        skill_m, _obj_m, touch_m = smoothed[m]
                        smoothed[m] = (skill_m, held_obj, touch_m)
                    k = run_end
                else:
                    # Không quay lại tên cũ nữa trong chuỗi -> có thể là
                    # đổi vật thật, để smooth_object_labels tự xử lý.
                    held_obj = smoothed[k][1]
                    k += 1
            else:
                k += 1
    return smoothed


def smooth_object_labels(labels_at_sample):
    """Khử nhiễu tên vật thể: nếu đang trong 1 chuỗi thao tác liên tục
    (HOLDING_SKILLS) trên cùng vật thể, mà 1 mốc giữa chừng đột nhiên đổi
    tên vật thể khác hẳn (VLM hallucination — vd đang cầm/rót "chai nước"
    suốt nhưng 1 mốc lại gọi thành "nắp chai" dù ảnh vẫn thấy nguyên cả
    chai), thay bằng tên vật thể của mốc TRƯỚC ĐÓ trong cùng chuỗi. Chuỗi
    bị ngắt khi gặp skill không giữ vật (Reach/GoHome/...) hoặc khi vật
    thể đổi tên nhưng được XÁC NHẬN bởi mốc kế tiếp cũng đổi tên giống vậy
    (nghĩa là đổi vật thật, không phải nhiễu 1 mốc)."""
    n = len(labels_at_sample)
    if n < 2:
        return labels_at_sample

    labels_at_sample = _bridge_chain_reversions(labels_at_sample)
    smoothed = list(labels_at_sample)
    held_obj = None
    for i in range(n):
        skill, obj, touch = smoothed[i]
        if skill not in HOLDING_SKILLS:
            held_obj = None
            continue
        if held_obj is None:
            held_obj = obj
            continue
        if obj != held_obj:
            held_group = find_synonym_group(held_obj)
            if held_group is not None and obj.strip().lower() in held_group:
                # Chỉ là đổi từ đồng nghĩa cho CÙNG 1 loại vật thể (vd
                # "cốc" <-> "ly") -> coi là cùng 1 vật thể, giữ nguyên tên
                # đã dùng từ đầu chuỗi, KHÔNG cần mốc kế tiếp xác nhận
                # (vì đây không phải nghi ngờ đổi vật thật, mà là VLM tự
                # thiếu nhất quán cách gọi tên qua các lần suy luận độc lập).
                smoothed[i] = (skill, held_obj, touch)
                continue
            # ĐÃ THỬ ngưỡng xác nhận 2 mốc (thay vì 1) để lọc hallucination
            # bền — nhưng kiểm chứng thực tế cho kết quả TỆ HƠN (4/29 mốc
            # sai so với 2/29 của ngưỡng 1 mốc), vì hallucination có thể
            # bền ngẫu nhiên NHIỀU hơn 2 mốc tuỳ lần chạy — không có ngưỡng
            # cố định nào an toàn tuyệt đối, và ngưỡng cao hơn chỉ khiến
            # hallucination dài hơn bị khoá lâu hơn khi nó thực sự xảy ra.
            # Quay lại ngưỡng 1 mốc (đơn giản nhất, thực nghiệm tốt nhất).
            next_obj = labels_at_sample[i + 1][1] if i + 1 < n else None
            if next_obj == obj:
                # Mốc kế tiếp cũng xác nhận tên vật thể mới -> đổi vật thật.
                held_obj = obj
            else:
                smoothed[i] = (skill, held_obj, touch)
    return smoothed


def put_vietnamese_text(frame, text, org, font_scale=1.0, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, int(28 * font_scale))
    except OSError:
        font = ImageFont.load_default()
    draw.text(org, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_video(video_path, skill_names, vi_names, interval_sec, out_path,
                   skill_descriptions=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(fps * interval_sec))
    # Vài frame ngay sau mốc sample cuối cùng, giữ riêng làm "khung sau" cho
    # cửa sổ trượt của mốc cuối — nếu không có, VLM chỉ thấy 2 khung (trước +
    # hiện tại) và dễ nhầm hành động đang đi LÊN (Lift) hay ĐI XUỐNG (Place)
    # vì thiếu ngữ cảnh chuyển động tương lai.
    tail_peek_offset = max(1, int(fps * 0.5))

    sample_frames = []  # [(timestamp, frame), ...]
    # Vài frame NGAY SAU mỗi mốc sample (cách nhau vài chục mili-giây, tư
    # thế tay gần như không đổi) — dùng làm phương án dự phòng khi
    # MediaPipe không phát hiện được tay ở đúng frame sample (đã kiểm
    # chứng: MediaPipe không ổn định 100%, cùng 1 tư thế nắm có lúc phát
    # hiện được có lúc không, tuỳ chính xác frame nào được đưa vào).
    touch_retry_burst = max(3, int(fps * 0.15))
    sample_retry_frames = []  # list[list[frame]], song song với sample_frames
    tail_peek_frame = None
    idx = 0
    burst_left = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            sample_frames.append((idx / fps, frame))
            sample_retry_frames.append([])
            burst_left = touch_retry_burst
        elif burst_left > 0:
            sample_retry_frames[-1].append(frame)
            burst_left -= 1
        idx += 1
    cap.release()

    if sample_frames:
        last_sample_idx = round(sample_frames[-1][0] * fps)
        peek_idx = last_sample_idx + tail_peek_offset
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, peek_idx)
        ret, frame = cap.read()
        if ret:
            tail_peek_frame = frame
        cap.release()

    print(f"Đã lấy {len(sample_frames)} mốc thời gian để phân loại "
          f"(cửa sổ 3 frame liên tiếp, kèm nhận diện vật thể).\n")

    # GIAI ĐOẠN 1: phân loại skill thô (VLM) cho từng mốc — CHƯA tính vị
    # trí chạm vội, vì skill ở đây có thể còn bị nhiễu 1 mốc đơn lẻ.
    raw_labels = []  # [(skill_en, object_vi, touch_location_vi_raw), ...]
    for i, (ts, frame) in enumerate(sample_frames):
        window = []
        if i > 0:
            window.append(sample_frames[i - 1][1])
        window.append(frame)
        if i < len(sample_frames) - 1:
            window.append(sample_frames[i + 1][1])
        elif tail_peek_frame is not None:
            window.append(tail_peek_frame)
        raw_labels.append(classify_window(window, skill_names, skill_descriptions))

    # ĐÃ THỬ VÀ BỎ: định vị bbox MỖI mốc + so vị trí không gian
    # (resolve_object_identity_by_position) để phân biệt vật thể thay vì
    # chỉ tin tên gọi. Kiểm chứng thực tế trên video có nhiều vật thể
    # thuỷ tinh trong suốt dùng LUÂN PHIÊN ở gần cùng 1 vị trí bàn (rót từ
    # chai nước ngay tại chỗ vừa đặt bình thuỷ tinh xuống) cho kết quả TỆ
    # HƠN cách khử nhiễu theo tên (7/29 mốc sai so với 2/29) — vị trí gần
    # nhau không đảm bảo CÙNG vật thể khi 2 vật khác nhau được dùng nối
    # tiếp ở cùng chỗ. Đã bỏ, quay lại cách khử nhiễu theo tên (bên dưới).

    # GIAI ĐOẠN 2: khử nhiễu skill TRƯỚC — nếu làm ngược lại (tính vị trí
    # chạm dựa trên skill thô rồi mới khử nhiễu skill), mốc nào bị đổi
    # skill thành "Grasp" bởi bước khử nhiễu sẽ bị bỏ sót hoàn toàn bước
    # grounding vị trí chạm (đã từng là bug thật: t bị đổi thành Grasp sau
    # khử nhiễu nhưng vẫn giữ nguyên mô tả chữ thô chưa kiểm chứng).
    smoothed_skills = smooth_skill_labels(raw_labels)
    smoothed_skills = smooth_isolated_object_noise(smoothed_skills)
    smoothed_skills = normalize_object_synonyms(smoothed_skills)
    smoothed_skills = smooth_object_labels(smoothed_skills)

    # GIAI ĐOẠN 3: dựa trên skill ĐÃ khử nhiễu, mới tính vị trí chạm cho
    # đúng những mốc thực sự là Grasp.
    segments = []
    labels_at_sample = []  # [(skill_en, object_vi, touch_location_vi), ...]
    # Vị trí chạm đã được MediaPipe XÁC THỰC gần nhất trong cùng 1 chuỗi
    # Grasp liên tục (cùng object) — dùng làm phương án dự phòng tốt hơn
    # nhiều so với tin thẳng mô tả tự do (hay hallucination) của VLM, khi
    # 1 mốc giữa chuỗi Grasp bị MediaPipe hoàn toàn không thấy tay (nắm
    # quá chặt, che khuất hết ngón) — vì trong lúc giữ yên vật, vị trí nắm
    # tay hầu như không đổi giữa các mốc liền kề.
    last_verified_touch = None
    last_verified_obj = None
    for i, (ts, frame) in enumerate(sample_frames):
        skill, obj, touch_location = smoothed_skills[i]
        # Cũng kiểm tra cả "Reach" — VLM đôi khi phân loại nhầm cả 1 chuỗi
        # liên tục thành "Reach" dù ảnh cho thấy tay đang nắm chặt vật thể
        # (đã kiểm chứng thực tế: sai liên tục nhiều mốc nên smoothing
        # không bắt được). Dùng chính hệ thống kiểm chứng tiếp xúc
        # (MediaPipe + bbox) để TỰ NÂNG CẤP "Reach" thành "Grasp" khi phát
        # hiện tay thực sự đang chạm vật, thay vì chỉ tin nhãn VLM gốc.
        if skill in TOUCH_DISPLAY_SKILLS or skill == "Reach":
            retry_frames = [frame] + sample_retry_frames[i]
            grounded, verified = describe_touch_location_grounded(retry_frames, obj, fallback_text=touch_location)
            if skill == "Reach":
                if verified and grounded:
                    # Có tiếp xúc thật được xác nhận -> đây là Grasp, không
                    # còn là Reach (Reach nghĩa là tiến lại gần, CHƯA chạm).
                    skill = "Grasp"
                    touch_location = grounded
                    last_verified_touch, last_verified_obj = grounded, obj
                else:
                    touch_location = ""
                    last_verified_touch, last_verified_obj = None, None
            elif verified:
                if not grounded and last_verified_obj == obj and last_verified_touch is not None:
                    # "verified" nghĩa là MediaPipe THẤY tay và đo được
                    # khoảng cách, nhưng kết luận "chưa chạm" (dist > ngưỡng)
                    # -- có thể đúng, nhưng cũng có thể do bbox vật thể mà
                    # VLM định vị bị XÊ DỊCH nhẹ giữa các lần gọi (không hoàn
                    # toàn ổn định), làm khoảng cách đo được lệch quá ngưỡng
                    # dù tay thực ra vẫn đang giữ nguyên vật. Nếu mốc NGAY
                    # TRƯỚC đó (cùng vật, cùng chuỗi giữ liên tục) vừa xác
                    # thực có chạm thật, ưu tiên tin vị trí đó hơn là 1 phép
                    # đo đơn lẻ mâu thuẫn với toàn bộ chuỗi đang giữ vật.
                    touch_location = last_verified_touch
                else:
                    touch_location = grounded
                    last_verified_touch, last_verified_obj = grounded, obj
            elif last_verified_obj == obj and last_verified_touch is not None:
                # MediaPipe hoàn toàn thất bại mốc này (nắm quá chặt) NHƯNG
                # vẫn đang trong cùng chuỗi Grasp cùng vật thể -> tin vị trí
                # đã xác thực gần nhất hơn là mô tả tự do chưa kiểm chứng.
                touch_location = last_verified_touch
            else:
                touch_location = grounded  # fallback_text gốc, chưa kiểm chứng được lần nào
        else:
            last_verified_touch, last_verified_obj = None, None
        labels_at_sample.append((skill, obj, touch_location))

    for (ts, _), (skill, obj, touch_location) in zip(sample_frames, labels_at_sample):
        skill_vi = vi_names.get(skill, skill)
        segments.append({
            "time_sec": round(ts, 1),
            "skill_en": skill,
            "skill_vi": skill_vi,
            "object_vi": obj,
            "touch_location_vi": touch_location,
        })
        obj_display = f" - {obj}" if obj else ""
        touch_display = f"  (chạm: {touch_location})" if touch_location else ""
        print(f"  t={ts:5.1f}s  ->  {skill} ({skill_vi}){obj_display}{touch_display}")

    # Chỉ hiện chớp vị trí chạm ở mốc đầu tiên của MỖI lần chạm mới (giá
    # trị khác mốc liền trước) — không lặp lại chớp ở các mốc kế tiếp
    # cùng giá trị (vd giữ nguyên "thân" suốt 5 mốc liên tiếp thì chỉ chớp
    # 1 lần lúc bắt đầu, không nháy liên tục suốt cả lúc đang giữ yên).
    is_new_touch_event = []
    prev_touch = None
    for skill, obj, touch in labels_at_sample:
        is_new_touch_event.append(bool(touch) and touch != prev_touch)
        prev_touch = touch

    # Ghi video overlay: gán nhãn gần nhất theo thời gian cho mỗi frame gốc
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Chữ "Chạm/nắm" chỉ hiện trong khoảnh khắc ngắn quanh mỗi mốc sample,
    # không hiện liên tục suốt cả đoạn skill (vd suốt cả đoạn Pour kéo dài
    # nhiều giây) — tránh gây rối vì vị trí chạm chỉ có ý nghĩa ngay lúc đó.
    # Bật/tắt đột ngột trông như nhấp nháy, nên fade in/out mượt bằng alpha
    # thay vì show/hide nhị phân.
    touch_flash_sec = min(0.6, interval_sec / 2)
    fade_in_sec = min(0.15, touch_flash_sec / 3)
    fade_out_sec = min(0.3, touch_flash_sec / 2)

    def touch_alpha(elapsed):
        if elapsed < 0 or elapsed > touch_flash_sec:
            return 0.0
        if elapsed < fade_in_sec:
            return elapsed / fade_in_sec
        if elapsed > touch_flash_sec - fade_out_sec:
            return max(0.0, (touch_flash_sec - elapsed) / fade_out_sec)
        return 1.0

    idx = 0
    sample_idx = 0
    current_skill, current_obj, current_touch_location = (
        labels_at_sample[0] if labels_at_sample else ("Unknown", "", "")
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        while (sample_idx + 1 < len(sample_frames)
               and t >= sample_frames[sample_idx + 1][0]):
            sample_idx += 1
            current_skill, current_obj, current_touch_location = labels_at_sample[sample_idx]

        sample_ts = sample_frames[sample_idx][0]
        # Chỉ hiện vị trí chạm/nắm khi skill có thao tác chạm tay VÀ đây là
        # mốc ĐẦU TIÊN của lần chạm này (không nháy lặp lại ở các mốc kế
        # tiếp cùng giá trị trong lúc đang giữ yên).
        show_this_touch = (current_touch_location and current_skill in TOUCH_DISPLAY_SKILLS
                            and is_new_touch_event[sample_idx])
        alpha = touch_alpha(t - sample_ts) if show_this_touch else 0.0

        skill_vi = vi_names.get(current_skill, current_skill)
        line1 = f"Skill: {skill_vi}" + (f" - {current_obj}" if current_obj else "")

        overlay = frame.copy()
        box_width = min(width - 20, 60 + 14 * max(len(line1), len(current_touch_location or "")))
        cv2.rectangle(overlay, (0, 0), (box_width, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        frame = put_vietnamese_text(frame, line1, (10, 8), font_scale=0.9)

        if alpha > 0.01:
            faded = put_vietnamese_text(frame, current_touch_location, (10, 46),
                                         font_scale=0.75, color=(0, 220, 255))
            frame = cv2.addWeighted(faded, alpha, frame, 1 - alpha, 0)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"\nĐã ghi video overlay vào: {out_path}")
    return segments


def main():
    global MODEL_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--skills", default="../skill_ontology/skills.yaml")
    parser.add_argument("--names-vi", default="../skill_ontology/skill_names_vi.yaml")
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--model", default=MODEL_NAME,
                         help=f"Model VLM trên Ollama (mặc định {MODEL_NAME}). "
                              "Vd: --model qwen3-vl:8b-instruct")
    parser.add_argument("--out", default="annotated_output_vi_v3.mp4")
    parser.add_argument("--log", default="segments_log_vi_v3.json")
    parser.add_argument("--task-log", default="video_task_log.json",
                         help="File tích luỹ task_name + skill_sequence qua nhiều video "
                              "(append, không ghi đè). Dùng --skip-task-log để tắt.")
    parser.add_argument("--skip-task-log", action="store_true",
                         help="Không suy luận task_name / không ghi vào video_task_log.json")
    args = parser.parse_args()

    MODEL_NAME = args.model
    print(f"Model VLM: {MODEL_NAME}")

    skill_names = load_skill_names(args.skills)
    skill_descriptions = load_skill_descriptions(args.skills)
    vi_names = load_vi_names(args.names_vi)
    print(f"Skill ontology: {skill_names}\n")

    segments = process_video(args.video_path, skill_names, vi_names, args.interval, args.out,
                              skill_descriptions=skill_descriptions)

    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu timeline vào: {args.log}")

    if not args.skip_task_log:
        # Suy luận tên task tổng quát của cả video từ chuỗi skill nhận diện
        # được, rồi TÍCH LUỸ (append) vào video_task_log.json — dữ liệu
        # tham khảo, KHÔNG tự động đưa vào Task Planner hay skills.yaml.
        skill_sequence = [seg["skill_en"] for seg in segments]
        task_name, deduped_sequence = infer_task_name(skill_sequence, model_name=MODEL_NAME)
        total = append_task_log(args.log, deduped_sequence, task_name, args.task_log)
        print(f"Chuỗi skill (đã rút gọn): {' -> '.join(deduped_sequence)}")
        print(f"Tên task suy luận được: {task_name}")
        print(f"Đã ghi vào: {args.task_log} (tổng {total} bản ghi tích luỹ)")

    if VLM_STATS["requests"]:
        print(f"\nThống kê VLM ({MODEL_NAME}): {VLM_STATS['requests']} request, "
              f"tổng {VLM_STATS['seconds']:.1f}s, "
              f"trung bình {VLM_STATS['seconds'] / VLM_STATS['requests']:.1f}s/request")

    print("\nLƯU Ý: nếu video output không phát được, convert bằng ffmpeg:")
    print(f"  ffmpeg -i {args.out} -c:v libx264 -pix_fmt yuv420p annotated_output_vi_v3_h264.mp4")


if __name__ == "__main__":
    main()
