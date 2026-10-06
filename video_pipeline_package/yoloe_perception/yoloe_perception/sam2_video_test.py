from pathlib import Path
import tempfile
import shutil

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "test.mp4"

SAM2_ROOT = Path.home() / "ros2_ws" / "third_party" / "sam2"

CHECKPOINT = (
    SAM2_ROOT
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

OBJ_ID = 1

WINDOW_NAME = "SAM2 Video Prompt"


# ============================================================
# GLOBAL UI STATE
# ============================================================

positive_points = []
negative_points = []

first_frame = None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        positive_points.append((x, y))

        print(f"[+] Positive: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:

        negative_points.append((x, y))

        print(f"[-] Negative: ({x}, {y})")


# ============================================================
# EXTRACT VIDEO -> JPG SEQUENCE
# ============================================================

def extract_video_frames(video_path):

    temp_dir = tempfile.mkdtemp(
        prefix="sam2_video_frames_"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame.copy())

        output_path = (
            Path(temp_dir)
            / f"{index:05d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        index += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            "Video has no readable frames."
        )

    print(
        f"Video: {len(frames)} frames | "
        f"{fps:.2f} FPS"
    )

    return temp_dir, frames, fps


# ============================================================
# SELECT OBJECT
# ============================================================

def select_object(frame):

    global first_frame

    first_frame = frame.copy()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    print()
    print("Controls:")
    print("LEFT CLICK  = positive point")
    print("RIGHT CLICK = negative point")
    print("ENTER       = start tracking")
    print("R           = reset points")
    print("Q / ESC     = quit")
    print()

    while True:

        display = first_frame.copy()

        # positive
        for x, y in positive_points:

            cv2.circle(
                display,
                (x, y),
                7,
                (0, 255, 0),
                -1,
            )

        # negative
        for x, y in negative_points:

            cv2.circle(
                display,
                (x, y),
                7,
                (0, 0, 255),
                -1,
            )

        cv2.putText(
            display,
            "L:+ R:- | ENTER:start | R:reset",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKey(20) & 0xFF

        # ENTER
        if key in (10, 13):

            if len(positive_points) == 0:

                print(
                    "Need at least one positive point."
                )

                continue

            break

        # R
        elif key == ord("r"):

            positive_points.clear()
            negative_points.clear()

            print("Prompts reset.")

        # Q / ESC
        elif key in (
            ord("q"),
            27,
        ):

            cv2.destroyAllWindows()

            raise SystemExit


# ============================================================
# SAM2 TRACKING
# ============================================================

def run_sam2(
    frame_dir,
    frames,
    fps,
):

    print("Loading SAM2...")

    predictor = build_sam2_video_predictor(
        MODEL_CFG,
        str(CHECKPOINT),
        device="cuda",
    )

    print("SAM2 loaded.")

    # --------------------------------------------------------
    # prompts
    # --------------------------------------------------------

    points = (
        positive_points
        + negative_points
    )

    labels = (
        [1] * len(positive_points)
        + [0] * len(negative_points)
    )

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    video_masks = {}

    # --------------------------------------------------------
    # SAM2 inference
    # --------------------------------------------------------

    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ),
    ):

        state = predictor.init_state(
            video_path=frame_dir,

            # giảm VRAM
            offload_video_to_cpu=True,

            offload_state_to_cpu=False,
        )

        predictor.reset_state(
            state
        )

        # ----------------------------------------------------
        # ADD PROMPT TO FRAME 0
        # ----------------------------------------------------

        (
            frame_idx,
            obj_ids,
            mask_logits,
        ) = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=OBJ_ID,
            points=points,
            labels=labels,
        )

        first_mask = (
            mask_logits[0] > 0.0
        )

        first_mask = (
            first_mask
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        video_masks[0] = first_mask

        print(
            "Prompt accepted. "
            "Propagating through video..."
        )

        # ----------------------------------------------------
        # VIDEO PROPAGATION
        # ----------------------------------------------------

        for (
            frame_idx,
            obj_ids,
            mask_logits,
        ) in predictor.propagate_in_video(
            state
        ):

            mask = (
                mask_logits[0] > 0.0
            )

            mask = (
                mask
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )

            video_masks[
                int(frame_idx)
            ] = mask

            print(
                f"\rTracking frame "
                f"{frame_idx + 1}/{len(frames)}",
                end="",
            )

    print()
    print("Tracking complete.")

    # --------------------------------------------------------
    # PLAY RESULT
    # --------------------------------------------------------

    delay = max(
        1,
        int(1000 / max(fps, 1)),
    )

    for index, frame in enumerate(frames):

        display = frame.copy()

        mask = video_masks.get(
            index
        )

        if mask is not None:

            overlay = np.zeros_like(
                display
            )

            overlay[mask] = (
                0,
                255,
                0,
            )

            display = cv2.addWeighted(
                display,
                1.0,
                overlay,
                0.45,
                0,
            )

        cv2.putText(
            display,
            f"Frame {index + 1}/{len(frames)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKey(
            delay
        ) & 0xFF

        if key in (
            ord("q"),
            27,
        ):
            break

    predictor.reset_state(
        state
    )

    del state

    torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    if not Path(VIDEO_PATH).is_file():

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    frame_dir = None

    try:

        # ----------------------------------------------------
        # MP4 -> JPG sequence
        # ----------------------------------------------------

        (
            frame_dir,
            frames,
            fps,
        ) = extract_video_frames(
            VIDEO_PATH
        )

        # ----------------------------------------------------
        # USER PROMPT ON FRAME 0
        # ----------------------------------------------------

        select_object(
            frames[0]
        )

        # ----------------------------------------------------
        # SAM2
        # ----------------------------------------------------

        run_sam2(
            frame_dir,
            frames,
            fps,
        )

    finally:

        if frame_dir is not None:

            shutil.rmtree(
                frame_dir,
                ignore_errors=True,
            )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()