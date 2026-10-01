# vlm_recognize.py — cải tiến

## Vấn đề của bản cũ
- Không kiểm tra Ollama có chạy / model đã pull chưa → lỗi mù mờ, tốn cả `timeout=300s` mới biết fail.
- Không retry khi request tới Ollama lỗi tạm thời (network hiccup, model đang load) → phải chạy lại thủ công.
- Không validate skill VLM trả về so với `skills.yaml` → skill "ảo giác" (hallucinated) lẫn thẳng vào kết quả mà không ai biết.
- Dùng `print()` cho cả log lẫn kết quả, không phân biệt được mức độ (info/warning/error), không tắt/bật verbose được.
- Lỗi (file không tồn tại, ontology rỗng, JSON parse fail...) không có xử lý rõ ràng — dễ crash với traceback khó đọc hoặc âm thầm trả `[]`.
- Ollama URL / model / timeout / số lần retry đều hard-code, không cấu hình qua CLI được.

## Đã cải thiện
1. **Fail-fast health check** (`check_ollama_ready`): gọi `/api/tags` trước khi encode ảnh và gửi request nặng, cảnh báo sớm nếu Ollama chưa chạy hoặc model chưa pull. Có thể tắt bằng `--skip-health-check`.
2. **Retry có backoff** (`_post_with_retry`): tự thử lại tối đa `--max-retries` lần (mặc định 3) khi request lỗi mạng/HTTP, backoff tăng dần thay vì fail ngay lần đầu.
3. **Validate skill theo ontology** (`validate_skills`): sau khi parse JSON, lọc bỏ các skill không nằm trong `skills.yaml`, dedupe skill lặp liên tiếp (thay vì để VLM tự lo, giờ có double-check ở code). Danh sách skill bị loại được log cảnh báo và lưu vào `unknown_skills_filtered` trong file kết quả để dễ audit.
4. **Logging chuẩn** thay cho `print` rải rác: dùng module `logging`, có `-v/--verbose`, phân biệt INFO/WARNING/ERROR.
5. **Xử lý lỗi rõ ràng** qua exception riêng `VLMRecognizeError`: thiếu file ontology, ontology rỗng, thiếu keyframe, không kết nối được Ollama... đều thoát với message rõ ràng và exit code 1, thay vì traceback thô hoặc silent `[]`.
6. **Cấu hình qua CLI**: thêm `--ollama-url`, `--model`, `--timeout`, `--max-retries` để không phải sửa code khi đổi model/host.
7. **Tách `sample_frames` và `parse_skill_list` thành hàm riêng** — dễ unit test độc lập, không còn logic lồng trong `main()`.
8. **Output file phong phú hơn**: thêm `unknown_skills_filtered` và `model` vào `recognition_result.json` để truy vết được VLM đã trả gì thừa và chạy bằng model nào.

## Chưa làm (ngoài phạm vi lần này)
- Chưa thêm unit test tự động cho `parse_skill_list` / `validate_skills`.
- Chưa xử lý batch nhiều video cùng lúc (script vẫn 1-video-1-lần-chạy).
