"""Sinh mask tự động bằng SAM 2 với lưới điểm prompt (automatic mask generation).

transformers không đóng gói sẵn AMG cho SAM2, nên phần này tự cài: rải lưới điểm
đều trên ảnh, mỗi điểm là một prompt, rồi lọc mask theo chất lượng/diện tích và
khử trùng bằng mask-IoU.
"""
import numpy as np
import torch


def point_grid(h, w, points_per_side):
    """Lưới điểm đều, tránh sát mép ảnh."""
    ys = (np.arange(points_per_side) + 0.5) / points_per_side * h
    xs = (np.arange(points_per_side) + 0.5) / points_per_side * w
    return [[float(x), float(y)] for y in ys for x in xs]


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


@torch.inference_mode()
def generate_masks(model, processor, image, device, points_per_side=16,
                   batch_points=64, pred_iou_thresh=0.80, min_area_frac=0.0015,
                   max_area_frac=0.25, nms_iou=0.75, border_frac=0.85):
    """image: RGB (H, W, 3). Trả list dict {mask, bbox, iou, area} đã lọc & khử trùng."""
    H, W = image.shape[:2]
    grid = point_grid(H, W, points_per_side)
    min_area, max_area = min_area_frac * H * W, max_area_frac * H * W

    cands = []
    for i in range(0, len(grid), batch_points):
        chunk = grid[i:i + batch_points]
        inputs = processor(
            images=image,
            input_points=[[[p] for p in chunk]],          # (1, n_obj, 1, 2)
            input_labels=[[[1] for _ in chunk]],
            return_tensors="pt",
        ).to(device)
        out = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(
            out.pred_masks, original_sizes=inputs["original_sizes"], binarize=True)[0]
        scores = out.iou_scores[0]                        # (n_obj, 3)

        best = scores.argmax(dim=-1)
        for k in range(masks.shape[0]):
            j = int(best[k])
            iou = float(scores[k, j])
            if iou < pred_iou_thresh:
                continue
            m = masks[k, j].cpu().numpy().astype(bool)
            area = int(m.sum())
            if area < min_area or area > max_area:
                continue
            cands.append({"mask": m, "iou": iou, "area": area})

    # bỏ mask phủ gần hết chiều ảnh (mặt bàn, tường, nền)
    kept_border = []
    for c in cands:
        ys, xs = np.nonzero(c["mask"])
        if (xs.max() - xs.min()) / W > border_frac and (ys.max() - ys.min()) / H > border_frac:
            continue
        kept_border.append(c)

    # NMS theo mask-IoU, ưu tiên mask có iou dự đoán cao hơn
    kept_border.sort(key=lambda c: -c["iou"])
    final = []
    for c in kept_border:
        if any(mask_iou(c["mask"], k["mask"]) > nms_iou for k in final):
            continue
        final.append(c)
    return final
