"""
mcap_to_mp4.py

CẦU NỐI còn thiếu của pipeline: segment_and_overlay_vi_v3.py chỉ nhận file
.mp4 (đọc bằng cv2.VideoCapture), nhưng dữ liệu thật của robot UR3 được ghi
ra dưới dạng ROS2 bag (rosbag2, định dạng .mcap) chỉ chứa ảnh camera thô —
KHÔNG có sẵn nhãn skill nào. Script này rút luồng ảnh màu từ bag ra .mp4 để
đưa tiếp vào pipeline VLM.

Bag demo thực tế chỉ có 4 topic (không có topic nhãn skill):
    /camera/camera/color/image_raw                    <- dùng cái này
    /camera/camera/aligned_depth_to_color/image_raw
    /camera/camera/color/camera_info
    /tf_static

VỀ FPS: bag ghi theo timestamp THẬT, các frame không cách đều nhau (đo được
20-67 ms, trung bình 33.6 ms). cv2.VideoWriter chỉ ghi được CFR (hằng số),
còn segment_and_overlay_vi_v3.py tính mốc thời gian bằng idx/fps — nên fps
ghi ra được chọn sao cho idx/fps khớp nhất với timeline thật của bag
(mặc định: (n_frame-1) / khoảng thời gian thật). Sai lệch còn lại chỉ vài
chục ms, không ảnh hưởng vì pipeline lấy mẫu thưa (mặc định 1.5 s/mốc).

Cách dùng:
    python3 mcap_to_mp4.py demo.mcap --out demo.mp4
    python3 mcap_to_mp4.py demo.mcap --out depth.mp4 \\
        --topic /camera/camera/aligned_depth_to_color/image_raw
"""

import argparse
import os
import sys

import cv2
import numpy as np

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False

DEFAULT_TOPIC = "/camera/camera/color/image_raw"


def _to_bgr(msg):
    """Chuyển ảnh ROS (msg.height/width/encoding/data) sang ndarray BGR của
    OpenCV. Hỗ trợ các encoding thực tế hay gặp: rgb8 (RealSense color),
    bgr8, mono8/mono16, và 16UC1 (depth — chuẩn hoá thành ảnh xám xem được,
    vì depth thô 16-bit gần như đen thui nếu ghi thẳng)."""
    h, w, enc = msg.height, msg.width, msg.encoding
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()

    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(h, msg.step)[:, :w].copy(), cv2.COLOR_GRAY2BGR)

    if enc in ("mono16", "16UC1"):
        img16 = buf.view(np.uint16).reshape(h, msg.step // 2)[:, :w]
        # Chuẩn hoá theo phân vị 5-95% thay vì min/max: vài pixel nhiễu xa
        # (cạnh vật, phản chiếu) có giá trị cực đại/cực tiểu sẽ ép toàn bộ
        # ảnh depth về gần như 1 màu nếu dùng min/max.
        valid = img16[img16 > 0]
        if valid.size == 0:
            norm = np.zeros(img16.shape, np.uint8)
        else:
            lo, hi = np.percentile(valid, 5), np.percentile(valid, 95)
            if hi <= lo:
                hi = lo + 1
            norm = np.clip((img16.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
            norm = np.where(img16 > 0, norm, 0).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    raise RuntimeError(f"Encoding chưa hỗ trợ: {enc}")


def mcap_to_mp4(mcap_path, out_path, topic=DEFAULT_TOPIC):
    if not MCAP_AVAILABLE:
        raise RuntimeError(
            "Chưa cài thư viện đọc mcap. Cài bằng:\n"
            "    pip install mcap mcap-ros2-support"
        )
    if not os.path.isfile(mcap_path):
        raise RuntimeError(f"Không tìm thấy file bag: {mcap_path}")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        # Thu thập frame + timestamp trước, vì fps ghi ra phụ thuộc vào
        # khoảng thời gian thật của cả bag (xem docstring phần FPS).
        frames, stamps = [], []
        for _schema, _chan, rec, msg in reader.iter_decoded_messages(topics=[topic]):
            frames.append(_to_bgr(msg))
            stamps.append(rec.log_time)

    if not frames:
        raise RuntimeError(
            f"Topic '{topic}' không có message nào trong bag.\n"
            f"Chạy lại với --list để xem các topic có sẵn."
        )

    h, w = frames[0].shape[:2]
    span_sec = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
    # fps sao cho frame cuối (index n-1) rơi đúng vào cuối timeline thật.
    fps = (len(frames) - 1) / span_sec if span_sec > 0 else 30.0
    if not (1.0 < fps < 240.0):
        print(f"  CẢNH BÁO: fps tính được bất thường ({fps:.2f}), dùng 30.0")
        fps = 30.0

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter cho: {out_path}")
    for fr in frames:
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h))
        writer.write(fr)
    writer.release()

    print(f"Topic     : {topic}")
    print(f"Độ phân giải: {w}x{h}")
    print(f"Số frame  : {len(frames)}")
    print(f"Thời lượng: {span_sec:.2f}s  ->  fps ghi ra: {fps:.2f}")
    print(f"Đã ghi    : {out_path}")
    return out_path


def list_topics(mcap_path):
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    for cid, ch in summary.channels.items():
        n = summary.statistics.channel_message_counts.get(cid, 0)
        print(f"  {ch.topic}  ({summary.schemas[ch.schema_id].name}, {n} msg)")


def main():
    parser = argparse.ArgumentParser(
        description="Rút luồng ảnh từ ROS2 bag (.mcap) ra .mp4 để đưa vào segment_and_overlay_vi_v3.py")
    parser.add_argument("mcap_path")
    parser.add_argument("--out", default=None, help="File .mp4 đầu ra")
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help=f"Topic ảnh cần rút (mặc định {DEFAULT_TOPIC})")
    parser.add_argument("--list", action="store_true",
                        help="Chỉ liệt kê các topic có trong bag rồi thoát")
    args = parser.parse_args()

    if args.list:
        if not MCAP_AVAILABLE:
            sys.exit("Chưa cài mcap: pip install mcap mcap-ros2-support")
        list_topics(args.mcap_path)
        return

    out = args.out or os.path.splitext(args.mcap_path)[0] + ".mp4"
    mcap_to_mp4(args.mcap_path, out, args.topic)
    print("\nBước tiếp theo — chạy pipeline sinh video có tên skill:")
    print(f"  python3 segment_and_overlay_vi_v3.py {out} --model qwen3-vl:8b-instruct \\")
    print("      --out out.mp4 --log log.json")


if __name__ == "__main__":
    main()
