"""
hand_pose_masked.py

Chạy MediaPipe HandLandmarker để lấy TƯ THẾ NGÓN TAY (ngón cuộn vào hay duỗi
ra), dùng mask bàn tay của Grounded-SAM để CHỌN ĐÚNG bàn tay và KIỂM CHỨNG
kết quả — rồi cung cấp chỉ số `curl` cho hoi_skill_inference.py.

MỤC ĐÍCH: mask SAM2 cho biết tay có CHẠM vật hay không, nhưng không biết ngón
tay đang cuộn hay duỗi. Hai tư thế rất khác nhau về ý nghĩa lại cho cùng một
tín hiệu "đang chạm":
    - ngón cuộn vào  -> đang CẦM (Grasp)
    - ngón duỗi ra   -> đang CHẠM/ĐẨY (Contact/Push), hoặc vừa MỞ RA (Release)
Module này bổ sung đúng thông tin đó.

--------------------------------------------------------------------------------
MỘT GIẢ THUYẾT SAI ĐÃ ĐƯỢC ĐO VÀ BÁC BỎ — đọc trước khi sửa file này
--------------------------------------------------------------------------------
Giả thuyết ban đầu: cắt ảnh theo bbox mask rồi phóng to sẽ giúp MediaPipe
nhận được bàn tay đang nắm chặt tốt hơn, vì bàn tay chiếm tỉ lệ lớn hơn trong
khung hình đầu vào.

Đã đo thật trên 347 frame của video demo (6919101... xem out/gsam2_demo):

    cách làm                                    phát hiện được
    toàn ảnh (baseline)                          337/347  (97%)
    crop theo mask + tô đen nền + phóng to 2x    295/347  (85%)
    |-- trong lúc CÓ tương tác (0.97-9.9 s)      257/267  vs  215/267
    |-- số frame CHỈ toàn ảnh làm được           47
    |-- số frame CHỈ crop làm được                5

=> CẮT THEO MASK LÀM KẾT QUẢ TỆ HƠN HẲN, không phải tốt hơn. Lý do: mask
"hand" của SAM2 gồm cả CẲNG TAY, nên crop là một vùng dài; tô đen nền ngoài
mask xoá luôn đường viền cánh tay — thứ MediaPipe dùng để định vị bàn tay —
và việc phóng to còn đẩy bàn tay ra khỏi thang kích thước mà model quen nhận.

VÌ VẬY file này CHẠY TRÊN TOÀN ẢNH (mặc định), và dùng mask cho 2 việc khác:
  1. CHỌN bàn tay: khung có thể có nhiều tay; chọn tay khớp nhất với mask.
  2. KIỂM CHỨNG: landmark phải nằm trong mask (nới rộng), nếu không thì bỏ —
     vì đó là dấu hiệu model dựng bàn tay sai chỗ.

Chế độ `mode="crop"` vẫn giữ để tái lập phép đo trên, KHÔNG dùng làm mặc định.

Cách dùng (độc lập, để soi kết quả):
    python3 hand_pose_masked.py masks_gsam2.json video.mp4 --dump curl.json
"""

import os

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:                        # pragma: no cover
    MEDIAPIPE_AVAILABLE = False

HAND_LANDMARKER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "hand_landmarker.task")

# (mcp, pip, tip) cho từng ngón. Ngón cái dùng khớp IP thay PIP — giải phẫu
# ngón cái chỉ có 2 đốt nên không có PIP thật.
FINGERS = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
TIP_IDS = [4, 8, 12, 16, 20]
PALM_IDS = [0, 5, 9, 13, 17]        # cổ tay + 4 khớp bàn — phần bám chắc nhất

# Landmark phải nằm trong mask (đã nới) với tỉ lệ tối thiểu này, nếu không thì
# coi là model dựng bàn tay sai chỗ và bỏ. Nới mask vì đầu ngón khi chạm vật
# thường nằm ngay sát rìa mask, và mask SAM2 không khít tuyệt đối.
MASK_VERIFY_DILATE_PX = 15
MIN_INSIDE_FRAC = 0.60


def _angle_at(a, b, c):
    """Góc (độ) tại điểm b, tạo bởi 2 vector b->a và b->c."""
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def hand_shape_features(lm):
    """lm: 21 landmark (x, y) theo PIXEL của frame gốc.

    Trả dict:
      curl     0..1 — độ cuộn trung bình của các ngón (0 = duỗi thẳng, 1 = nắm chặt)
      n_curled      — số ngón có độ cuộn > 0.5
      pinch         — khoảng cách đầu ngón cái - đầu ngón trỏ, chuẩn hoá theo bàn tay
      openness      — độ "mở" trung bình của các đầu ngón, chuẩn hoá theo bàn tay
      scale         — kích thước bàn tay (px), dùng để chuẩn hoá

    CÁCH ĐO CURL: tại khớp PIP, góc tạo bởi vector PIP->MCP và PIP->TIP.
    Ngón duỗi thẳng thì 2 vector này gần như ngược hướng -> góc ~180°.
    Ngón cuộn vào thì đầu ngón quay về phía lòng bàn tay -> góc nhỏ dần.
    Lấy 1 - góc/180 cho về thang 0..1.

    Chuẩn hoá theo "kích thước bàn tay" = khoảng cách cổ tay -> khớp MCP ngón
    giữa, vì đại lượng này KHÔNG phụ thuộc việc ngón đang duỗi hay cuộn (khác
    bbox — bbox phình ra khi duỗi ngón), nên tỉ lệ thu được ổn định hơn khi
    tay ở xa hay gần camera.
    """
    wrist = lm[0]
    scale = float(np.linalg.norm(np.asarray(lm[9], float) - np.asarray(wrist, float)))
    if scale < 1e-3:
        return None

    curls = []
    for name, (mcp, pip, tip) in FINGERS.items():
        ang = _angle_at(lm[mcp], lm[pip], lm[tip])
        if ang is not None:
            curls.append(1.0 - ang / 180.0)
    if not curls:
        return None

    pinch = float(np.linalg.norm(np.asarray(lm[4], float) - np.asarray(lm[8], float))) / scale
    openness = float(np.mean([np.linalg.norm(np.asarray(lm[t], float) - np.asarray(wrist, float))
                              for t in TIP_IDS])) / scale
    return {
        "curl": float(np.mean(curls)),
        "n_curled": int(sum(1 for c in curls if c > 0.5)),
        "pinch": pinch,
        "openness": openness,
        "scale": scale,
    }


class MaskedHandPose:
    """Bọc MediaPipe HandLandmarker; mask bàn tay dùng để chọn tay + kiểm chứng.

    Khởi tạo model 1 lần rồi dùng lại cho cả video — tạo lại mỗi frame rất
    chậm vì model 7.8 MB phải nạp lại liên tục."""

    def __init__(self, model_path=HAND_LANDMARKER_MODEL, num_hands=2,
                 mode="full", crop_pad=20, upscale=2.0, min_confidence=0.1):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "Chưa cài mediapipe: pip install mediapipe\n"
                "và cần file hand_landmarker.task trong video_pipeline/")
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Thiếu model: {model_path}")
        if mode not in ("full", "crop"):
            raise ValueError("mode phải là 'full' hoặc 'crop'")
        self.mode, self.crop_pad, self.upscale = mode, crop_pad, upscale
        self.detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                num_hands=num_hands,
                # Hạ ngưỡng như bản v3: tay đang nắm chặt có hình dạng khác hẳn
                # tay xoè mà model quen nhận, nên conf mặc định 0.5 dễ trượt.
                min_hand_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            ))

    def _detect_on(self, bgr_in):
        rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
        return self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).hand_landmarks

    def detect(self, bgr, hand_mask):
        """bgr: frame gốc (BGR). hand_mask: mask bool cùng kích thước.

        Trả (landmarks_px, features) trong hệ toạ độ FRAME GỐC, hoặc
        (None, None) nếu không có bàn tay nào khớp với mask đủ tin cậy."""
        if hand_mask is None or not hand_mask.any():
            return None, None

        # MASK VÀ VIDEO CÓ THỂ KHÁC KÍCH THƯỚC: gsam2_video.py có cờ --max-side
        # thu nhỏ frame TRƯỚC khi detect, nên mask sinh ra theo hệ toạ độ ĐÃ
        # THU NHỎ. Với video_test2.mp4 (1280x720) và --max-side 720, mask là
        # 720x405 còn video gốc vẫn 1280x720 — tra mask bằng toạ độ frame gốc
        # sẽ văng IndexError. Ở đây detect trên ảnh đã hạ về đúng cỡ mask, rồi
        # mới nhân ngược lên hệ frame gốc khi trả về.
        mh, mw = hand_mask.shape[:2]
        fh, fw = bgr.shape[:2]
        if (mh, mw) != (fh, fw):
            det_img = cv2.resize(bgr, (mw, mh), interpolation=cv2.INTER_AREA)
        else:
            det_img = bgr
        sx, sy = fw / mw, fh / mh

        if self.mode == "crop":
            img, ox, oy, s = self._make_crop(det_img, hand_mask)
            if img is None:
                return None, None
        else:
            img, ox, oy, s = det_img, 0.0, 0.0, 1.0

        hands = self._detect_on(img)
        if not hands:
            return None, None

        # Mask nới để kiểm chứng: đầu ngón khi chạm vật nằm sát rìa mask, và
        # mask SAM2 không khít tuyệt đối quanh ngón.
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * MASK_VERIFY_DILATE_PX + 1,) * 2)
        mask_soft = cv2.dilate(hand_mask.astype(np.uint8), k) > 0

        # Nhiều tay trong khung: chọn tay khớp nhất với mask thay vì lấy tay
        # đầu tiên — người trong video có thể có 2 tay, chỉ 1 tay đang thao tác.
        best, best_frac = None, -1.0
        for hl in hands:
            # toạ độ trong hệ ẢNH ĐEM ĐI DETECT (= hệ mask)
            pts = np.array([[(lm.x * img.shape[1]) / s + ox,
                             (lm.y * img.shape[0]) / s + oy] for lm in hl], dtype=float)
            inside = 0
            for px, py in pts:
                ix, iy = int(round(px)), int(round(py))
                if 0 <= ix < mw and 0 <= iy < mh and mask_soft[iy, ix]:
                    inside += 1
            frac = inside / len(pts)
            if frac > best_frac:
                best, best_frac = pts, frac
        if best is None or best_frac < MIN_INSIDE_FRAC:
            return None, None

        # Trả landmark về hệ toạ độ FRAME GỐC cho người gọi.
        best = best * np.array([sx, sy], dtype=float)
        feats = hand_shape_features(best)
        if feats is None:
            return None, None
        feats["inside_frac"] = float(best_frac)
        return best, feats

    def _make_crop(self, bgr, hand_mask):
        """Chế độ crop — ĐÃ ĐO ĐƯỢC LÀ KÉM HƠN toàn ảnh (xem docstring đầu
        file). Giữ lại chỉ để tái lập phép đo, không dùng mặc định."""
        h, w = bgr.shape[:2]
        ys, xs = np.nonzero(hand_mask)
        pad = self.crop_pad
        x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        x1, y1 = min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, 0, 0, 1.0
        crop = bgr[y0:y1, x0:x1].copy()
        crop[~hand_mask[y0:y1, x0:x1]] = 0
        s = self.upscale
        if s != 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return crop, float(x0), float(y0), s


def main():
    """Chạy thử: đo tư thế tay từng frame rồi in curl theo thời gian, để đối
    chiếu với các pha skill nhìn thấy bằng mắt."""
    import argparse
    import json
    import hoi_skill_inference as H

    ap = argparse.ArgumentParser()
    ap.add_argument("masks_json")
    ap.add_argument("video")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--mode", default="full", choices=["full", "crop"])
    args = ap.parse_args()

    data, per_frame = H.load_mask_series(args.masks_json)
    hand_ids, _ = H.pick_roles(data)
    fps = float(data["fps"])
    pose = MaskedHandPose(mode=args.mode)

    cap = cv2.VideoCapture(args.video)
    rows, idx = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            masks = per_frame[idx] if idx < len(per_frame) else {}
            hm = None
            for hh in hand_ids:
                if hh in masks:
                    hm = masks[hh] if hm is None else (hm | masks[hh])
            _lm, feats = pose.detect(bgr, hm)
            rows.append({"frame": idx, "t": round(idx / fps, 3),
                         "found": feats is not None,
                         **({k: round(v, 3) for k, v in feats.items()} if feats else {})})
        idx += 1
    cap.release()

    n = len(rows)
    hit = sum(1 for r in rows if r["found"])
    print(f"Phát hiện tay: {hit}/{n} frame ({100 * hit / n:.0f}%)  [mode={args.mode}]")
    print(f"\n{'t':>6} {'curl':>6} {'n_curled':>9} {'pinch':>7} {'openness':>9}")
    for r in rows[::max(1, n // 40)]:
        if r["found"]:
            print(f"{r['t']:6.2f} {r['curl']:6.2f} {r['n_curled']:9d} "
                  f"{r['pinch']:7.2f} {r['openness']:9.2f}")
        else:
            print(f"{r['t']:6.2f}   --")
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(rows, f)
        print(f"\nĐã ghi: {args.dump}")


if __name__ == "__main__":
    main()
