#!/usr/bin/env python3
"""So sánh 2 lần chạy segment_and_overlay_vi_v3.py trên CÙNG 1 video.

Dùng cho thí nghiệm A/B khi đổi model VLM: giữ nguyên mọi tham số, chỉ đổi
--model, rồi so từng mốc thời gian giữa 2 file log JSON.

Nhánh A đóng vai trò "đáp án" (kết quả đã được xem video thật và xác nhận
đúng), nhánh B là bản cần chấm điểm.

    python3 compare_runs.py log_test5_fix4.json log_test5_qwen3.json \
        --label-a qwen2.5vl:32b --label-b qwen3-vl:8b-instruct
"""

import argparse
import json


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_by_time(segments):
    return {round(float(s["time_sec"]), 1): s for s in segments}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a", help="Log nhánh A (đối chứng / đáp án)")
    ap.add_argument("log_b", help="Log nhánh B (cần chấm điểm)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, b = index_by_time(load(args.log_a)), index_by_time(load(args.log_b))
    times = sorted(set(a) | set(b))

    diff_skill = diff_obj = diff_touch = 0
    lost_touch = gained_touch = 0
    rows = []

    for t in times:
        sa, sb = a.get(t), b.get(t)
        if sa is None or sb is None:
            rows.append((t, "THIẾU MỐC", f"chỉ có ở {args.label_b if sa is None else args.label_a}"))
            continue

        marks = []
        if sa["skill_en"] != sb["skill_en"]:
            diff_skill += 1
            marks.append(f"skill: {sa['skill_en']} -> {sb['skill_en']}")
        if sa["object_vi"] != sb["object_vi"]:
            diff_obj += 1
            marks.append(f"vật: '{sa['object_vi']}' -> '{sb['object_vi']}'")

        ta, tb = sa["touch_location_vi"], sb["touch_location_vi"]
        if ta != tb:
            diff_touch += 1
            if ta and not tb:
                lost_touch += 1
                marks.append(f"MẤT vị trí chạm ('{ta}' -> rỗng)")
            elif tb and not ta:
                gained_touch += 1
                marks.append(f"THÊM vị trí chạm (rỗng -> '{tb}')")
            else:
                marks.append(f"chạm: '{ta}' -> '{tb}'")

        if marks:
            rows.append((t, "KHÁC", "; ".join(marks)))

    total = len(times)
    print(f"\n{'=' * 70}")
    print(f"A (đối chứng) = {args.label_a}   [{args.log_a}]")
    print(f"B (thử nghiệm) = {args.label_b}   [{args.log_b}]")
    print(f"{'=' * 70}\n")

    if rows:
        print(f"CÁC MỐC KHÁC NHAU ({len(rows)}/{total}):\n")
        for t, kind, detail in rows:
            print(f"  t={t:5.1f}s  [{kind}]  {detail}")
    else:
        print("Hai lần chạy TRÙNG KHỚP HOÀN TOÀN ở mọi mốc.")

    print(f"\n{'-' * 70}")
    print(f"Tổng số mốc:                    {total}")
    print(f"Khác skill:                     {diff_skill}")
    print(f"Khác tên vật thể:               {diff_obj}")
    print(f"Khác vị trí chạm:               {diff_touch}"
          f"  (mất: {lost_touch}, thêm: {gained_touch})")

    empty_a = sum(1 for s in a.values() if not s["touch_location_vi"])
    empty_b = sum(1 for s in b.values() if not s["touch_location_vi"])
    print(f"Mốc TRỐNG vị trí chạm:          A={empty_a}  B={empty_b}")

    objs_a = sorted({s["object_vi"] for s in a.values() if s["object_vi"]})
    objs_b = sorted({s["object_vi"] for s in b.values() if s["object_vi"]})
    print(f"\nTập tên vật thể A: {objs_a}")
    print(f"Tập tên vật thể B: {objs_b}")
    print("-" * 70)
    print("\nLƯU Ý: 'khác' KHÔNG đồng nghĩa với 'sai' — phải mở video đối chiếu\n"
          "ảnh thật để biết bên nào đúng. Bảng này chỉ khoanh vùng chỗ cần xem.")


if __name__ == "__main__":
    main()
