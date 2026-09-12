"""Tiện ích dùng chung cho 2 nhánh SAM+DINO: đọc video, tracking SAM2 theo chunk,
vẽ overlay, mã hoá mask, đo thời gian."""
import json
import time

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------- video I/O
def read_video(path, max_seconds=None, max_side=None):
    """Đọc video thành list frame RGB. Trả (frames, fps, (H, W))."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(fps * max_seconds) if max_seconds else None

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok or (limit and len(frames) >= limit):
            break
        if max_side:
            h, w = bgr.shape[:2]
            s = max_side / max(h, w)
            if s < 1.0:
                bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"video rỗng: {path}")
    return frames, fps, frames[0].shape[:2]


def keyframe_indices(n_frames, fps, interval):
    """Chỉ số các keyframe, cách nhau `interval` giây."""
    step = max(1, round(fps * interval))
    return list(range(0, n_frames, step))


def write_video(path, frames_bgr, fps):
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()


# ------------------------------------------------------------------- masks
def mask_bbox(mask):
    """xyxy (int) của mask bool, hoặc None nếu mask rỗng."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def box_containment(small, big):
    """Tỉ lệ diện tích `small` nằm trong `big`. 1.0 = nằm gọn hoàn toàn.
    Bắt được trường hợp Grounding DINO ra nhiều box lồng nhau cho cùng một vật,
    mà IoU không đủ cao để khử."""
    ix0, iy0 = max(small[0], big[0]), max(small[1], big[1])
    ix1, iy1 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = (small[2] - small[0]) * (small[3] - small[1])
    return inter / area if area > 0 else 0.0


def encode_rle(mask):
    """RLE row-major đơn giản: counts[0] là số pixel 0 đầu tiên, rồi xen kẽ 1/0.
    Tự cài để khỏi phụ thuộc pycocotools; giải mã bằng decode_rle bên dưới."""
    flat = mask.reshape(-1).astype(np.uint8)
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0] == 1:              # luôn bắt đầu bằng run của giá trị 0
        counts = [0] + counts
    return {"size": list(mask.shape), "counts": counts}


def decode_rle(rle):
    flat = np.zeros(int(np.prod(rle["size"])), dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(rle["size"])


# ----------------------------------------------------------- vẽ kết quả
_PALETTE = [
    (255, 82, 82), (66, 165, 245), (102, 187, 106), (255, 167, 38),
    (171, 71, 188), (38, 198, 218), (255, 238, 88), (141, 110, 99),
    (236, 64, 122), (120, 144, 156),
]


def color_for(obj_id):
    return _PALETTE[obj_id % len(_PALETTE)]


def draw_frame(rgb, objects, alpha=0.45):
    """objects: list dict {id, label, mask, bbox, caption}. Trả frame BGR đã vẽ."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    for o in objects:
        c = color_for(o["id"])[::-1]        # RGB -> BGR
        overlay[o["mask"]] = c
    bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

    for o in objects:
        c = color_for(o["id"])[::-1]
        x0, y0, x1, y1 = o["bbox"]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), c, 2)
        text = o["caption"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr, (x0, max(0, y0 - th - 6)), (x0 + tw + 6, y0), c, -1)
        cv2.putText(bgr, text, (x0 + 3, max(th, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


# --------------------------------------------------------------- SAM2 track
class Sam2ChunkTracker:
    """Chia video thành các chunk giữa 2 keyframe. Mỗi chunk mở một SAM2 video
    session, nạp box phát hiện được ở frame đầu chunk rồi lan truyền tới hết chunk.
    ID toàn cục được nối giữa các chunk bằng IoU của bbox tại frame giao nhau,
    nên một vật giữ nguyên ID xuyên suốt video."""

    def __init__(self, model, processor, device, dtype=torch.bfloat16, link_iou=0.5,
                 link_by_label=True):
        """link_by_label=False: nối ID chỉ theo IoU, bỏ qua nhãn. Dùng cho nhánh B,
        nơi nhãn là tên cụm (`cluster_2`) dùng chung cho nhiều vật khác nhau nên
        không phải căn cứ để phân biệt."""
        self.model, self.processor, self.device = model, processor, device
        self.dtype, self.link_iou = dtype, link_iou
        self.link_by_label = link_by_label
        self._next_id = 0
        self.objects = {}                    # global_id -> dict metadata

    def _assign_ids(self, dets, prev_boxes):
        """Ghép detection của chunk mới với object cuối chunk trước.

        Hai vòng: (1) khớp theo nhãn + IoU bbox; (2) với detection còn thừa, nếu
        nhãn đó chỉ còn đúng MỘT object cũ chưa dùng thì nối luôn — chai bị tay
        che hoặc xoay mạnh giữa 2 keyframe làm IoU tụt dưới ngưỡng, nếu không có
        vòng 2 thì cùng một vật bị tách thành 2 ID."""
        ids = [None] * len(dets)
        used = set()

        for i, d in enumerate(dets):                       # vòng 1: IoU
            best, best_iou = None, self.link_iou
            for gid, (box, label) in prev_boxes.items():
                if gid in used or (self.link_by_label and label != d["label"]):
                    continue
                iou = box_iou(d["box"], box)
                if iou >= best_iou:
                    best, best_iou = gid, iou
            if best is not None:
                used.add(best)
                ids[i] = best

        for i, d in enumerate(dets):                       # vòng 2: nhãn duy nhất
            if ids[i] is not None or not self.link_by_label:
                continue
            cands = [gid for gid, (_, label) in prev_boxes.items()
                     if gid not in used and label == d["label"]]
            if len(cands) == 1 and sum(x["label"] == d["label"] for x in dets) == 1:
                used.add(cands[0])
                ids[i] = cands[0]

        for i, d in enumerate(dets):                       # còn lại: object mới
            if ids[i] is not None:
                continue
            gid = self._next_id
            self._next_id += 1
            self.objects[gid] = {"id": gid, "label": d["label"],
                                 "first_frame": d["frame"], "score": d["score"]}
            used.add(gid)
            ids[i] = gid
        return ids

    def run(self, frames, keyframes, detect_fn, progress=True):
        """detect_fn(frame_rgb, frame_idx) -> list {box, label, score}.
        Trả (per_frame, timing) với per_frame[i] = list object đã vẽ được."""
        n = len(frames)
        per_frame = [[] for _ in range(n)]
        chunks = list(zip(keyframes, keyframes[1:] + [n]))
        prev_boxes, t_detect, t_track = {}, [], []

        for ci, (start, end) in enumerate(chunks):
            t0 = time.perf_counter()
            dets = detect_fn(frames[start], start)
            t_detect.append(time.perf_counter() - t0)

            if not dets:
                if progress:
                    print(f"  chunk {ci + 1}/{len(chunks)} frame {start}: không có detection")
                continue

            for d in dets:
                d["frame"] = start
            gids = self._assign_ids(dets, prev_boxes)

            t0 = time.perf_counter()
            session = self.processor.init_video_session(
                video=frames[start:end], inference_device=self.device, dtype=self.dtype,
            )
            self.processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=list(gids),
                input_boxes=[[list(map(float, d["box"])) for d in dets]],
            )

            last_boxes = {}
            with torch.inference_mode():
                for out in self.model.propagate_in_video_iterator(session, start_frame_idx=0):
                    masks = self.processor.post_process_masks(
                        [out.pred_masks],
                        original_sizes=[[session.video_height, session.video_width]],
                        binarize=True,
                    )[0]
                    fi = start + out.frame_idx
                    for k, gid in enumerate(gids):
                        m = masks[k].squeeze().cpu().numpy().astype(bool)
                        bb = mask_bbox(m)
                        if bb is None:
                            continue
                        meta = self.objects[gid]
                        per_frame[fi].append({
                            "id": gid, "label": meta["label"], "mask": m, "bbox": bb,
                            "score": meta["score"],
                            "caption": f'{meta["label"]} #{gid} {meta["score"]:.2f}',
                        })
                        last_boxes[gid] = (bb, meta["label"])
            t_track.append((time.perf_counter() - t0) / max(1, end - start))
            # nhớ box cuối của MỌI object đã thấy, không chỉ chunk liền trước:
            # một keyframe sót detection thì chunk sau vẫn nối lại được ID cũ
            prev_boxes.update(last_boxes)
            if progress:
                print(f"  chunk {ci + 1}/{len(chunks)} frame {start}-{end - 1}: "
                      f"{len(dets)} obj -> id {gids}")

        timing = {
            "n_frames": n, "n_keyframes": len(keyframes),
            "detect_ms_per_keyframe": round(1000 * float(np.mean(t_detect)), 1) if t_detect else None,
            "track_ms_per_frame": round(1000 * float(np.mean(t_track)), 1) if t_track else None,
        }
        return per_frame, timing


def dump_masks_json(path, meta, per_frame, fps, objects):
    frames_out = []
    for i, objs in enumerate(per_frame):
        if not objs:
            continue
        frames_out.append({
            "frame": i, "t": round(i / fps, 3),
            "objects": [{"id": o["id"], "label": o["label"], "bbox": o["bbox"],
                         "area": int(o["mask"].sum()), "rle": encode_rle(o["mask"])}
                        for o in objs],
        })
    with open(path, "w") as f:
        json.dump({**meta, "objects": list(objects.values()), "frames": frames_out}, f)
