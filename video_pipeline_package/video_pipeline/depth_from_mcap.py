"""
depth_from_mcap.py

Rút luồng DEPTH từ ROS2 bag (.mcap) và CĂN nó theo đúng chỉ số frame của luồng
ảnh màu, để dùng chung hệ toạ độ với mask của Grounded-SAM.

VÌ SAO CẦN CĂN LẠI: trong bag demo, luồng màu có 347 message còn luồng depth có
346 — KHÔNG bằng nhau, nên không thể ghép theo chỉ số 0,1,2,... Dù topic tên là
`aligned_depth_to_color` (đã căn theo pixel với ảnh màu) thì SỐ LƯỢNG frame vẫn
lệch, do 2 luồng phát ở tần số khác nhau và có frame rơi. Cách đúng là ghép theo
TIMESTAMP: mỗi frame màu lấy frame depth có mốc thời gian gần nhất.

Cùng lý do đó, depth phải được lưu theo CHỈ SỐ FRAME MÀU (không phải theo thứ
tự message depth), vì mask SAM2 được sinh trên video màu nên đánh số theo frame
màu. Nhờ vậy `depth[i]` và `masks[i]` nói về cùng một thời điểm.

Đầu ra: file .npz gồm
    depth : uint16 (N, H, W) — milimét, 0 = pixel không đo được
    t     : float64 (N,)     — giây, mốc thời gian của từng frame màu
    fps, size, size_color
    fx, fy, cx, cy — nội tham số camera (từ topic camera_info, lấy 1 lần vì
        camera không zoom giữa video) — CẦN để đổi (pixel u,v + depth mm)
        thành toạ độ 3D thật (X, Y, Z mm) trong hệ CAMERA bằng mô hình
        pinhole chuẩn: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy. Không có thì chỉ dùng
        được Z (độ cao tương đối), không quy đổi được X,Y sang milimét.
Nén lại vì depth rất trơn (vùng liền màu) — nén tốt hơn nhiều so với ảnh màu.

Cách dùng:
    python3 depth_from_mcap.py demo.mcap --out demo_depth.npz
    python3 depth_from_mcap.py demo.mcap --out demo_depth.npz --list
"""

import argparse
import os

import numpy as np

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    MCAP_AVAILABLE = True
except ImportError:                        # pragma: no cover
    MCAP_AVAILABLE = False

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"


def _depth_array(msg):
    """sensor_msgs/Image -> mảng uint16 (H, W) milimét."""
    if msg.encoding not in ("16UC1", "mono16"):
        raise RuntimeError(f"Depth encoding lạ: {msg.encoding} (chỉ hỗ trợ 16UC1)")
    a = np.frombuffer(bytes(msg.data), dtype=np.uint16)
    a = a.reshape(msg.height, msg.step // 2)[:, :msg.width]
    return a.copy()


def extract(mcap_path, out_path, color_topic=COLOR_TOPIC, depth_topic=DEPTH_TOPIC,
           camera_info_topic=CAMERA_INFO_TOPIC):
    if not MCAP_AVAILABLE:
        raise RuntimeError("Chưa cài thư viện mcap: pip install mcap mcap-ros2-support")
    if not os.path.isfile(mcap_path):
        raise RuntimeError(f"Không tìm thấy bag: {mcap_path}")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        # đọc 1 lượt: gom timestamp của 2 luồng trước, rồi mới decode depth
        color_t, depth_t, depth_msgs = [], [], []
        intrinsics = None    # (fx, fy, cx, cy) — LẤY MỘT LẦN, camera không zoom
        for _s, _c, rec, msg in reader.iter_decoded_messages(
                topics=[color_topic, depth_topic, camera_info_topic]):
            if _c.topic == color_topic:
                color_t.append(rec.log_time)
            elif _c.topic == depth_topic:
                depth_t.append(rec.log_time)
                depth_msgs.append(msg)
            elif intrinsics is None:               # camera_info: chỉ cần msg đầu
                k = msg.k                           # row-major 3x3: [fx,0,cx, 0,fy,cy, 0,0,1]
                intrinsics = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    if not color_t:
        raise RuntimeError(f"Không có message nào ở topic màu {color_topic}")
    if not depth_msgs:
        raise RuntimeError(f"Không có message nào ở topic depth {depth_topic}")

    color_t = np.asarray(color_t, dtype=np.int64)
    depth_t = np.asarray(depth_t, dtype=np.int64)

    # Ghép mỗi frame màu với frame depth gần nhất về thời gian. Dùng
    # searchsorted trên mảng đã sắp xếp (bag ghi theo thời gian) nên O(N log M),
    # không phải quét cặp O(N*M).
    idx = np.searchsorted(depth_t, color_t)
    idx = np.clip(idx, 1, len(depth_t) - 1)
    left, right = depth_t[idx - 1], depth_t[idx]
    nearest = np.where(np.abs(color_t - left) <= np.abs(right - color_t), idx - 1, idx)
    deltas_ms = np.abs(color_t - depth_t[nearest]) / 1e6

    depth_frames = [_depth_array(m) for m in depth_msgs]
    h, w = depth_frames[0].shape[:2]
    stack = np.zeros((len(color_t), h, w), dtype=np.uint16)
    for i, j in enumerate(nearest):
        d = depth_frames[j]
        if d.shape[:2] == (h, w):
            stack[i] = d

    t = (color_t - color_t[0]) / 1e9
    fps = float((len(color_t) - 1) / t[-1]) if t[-1] > 0 else 30.0

    extra = {}
    if intrinsics is not None:
        extra = {"fx": intrinsics[0], "fy": intrinsics[1],
                 "cx": intrinsics[2], "cy": intrinsics[3]}
    np.savez_compressed(out_path, depth=stack, t=t, fps=fps,
                        size=np.array([h, w]),
                        size_color=np.array(stack.shape[1:]), **extra)

    valid = stack[stack > 0]
    print(f"Frame màu   : {len(color_t)}")
    print(f"Frame depth : {len(depth_msgs)}  -> ghép theo timestamp gần nhất")
    print(f"Lệch mốc    : trung vị {np.median(deltas_ms):.1f} ms, "
          f"lớn nhất {deltas_ms.max():.1f} ms")
    print(f"Kích thước  : {w}x{h}")
    if intrinsics is not None:
        print(f"Nội tham số : fx={intrinsics[0]:.1f} fy={intrinsics[1]:.1f} "
              f"cx={intrinsics[2]:.1f} cy={intrinsics[3]:.1f}  "
              f"(từ {camera_info_topic}, dùng để đổi pixel+depth -> mm 3D)")
    else:
        print(f"Nội tham số : KHÔNG có topic {camera_info_topic} trong bag — "
              f"sẽ không đổi được toạ độ pixel sang mm (X,Y), chỉ có Z (depth)")
    if valid.size:
        print(f"Depth hợp lệ: {100 * valid.size / stack.size:.1f}% pixel, "
              f"khoảng {valid.min()}–{valid.max()} mm, trung vị {np.median(valid):.0f} mm")
    print(f"Đã ghi      : {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def list_topics(mcap_path):
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    for cid, ch in summary.channels.items():
        n = summary.statistics.channel_message_counts.get(cid, 0)
        print(f"  {ch.topic}  ({summary.schemas[ch.schema_id].name}, {n} msg)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mcap_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        if not MCAP_AVAILABLE:
            raise SystemExit("Chưa cài mcap: pip install mcap mcap-ros2-support")
        list_topics(args.mcap_path)
        return

    out = args.out or os.path.splitext(args.mcap_path)[0] + "_depth.npz"
    extract(args.mcap_path, out)


if __name__ == "__main__":
    main()
