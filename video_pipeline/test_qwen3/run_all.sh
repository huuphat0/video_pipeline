#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=../../venv/bin/python
[ -x "$PY" ] || PY=python3
for i in 0 1 2 3 4; do
  if [ -f "video_test$i.mp4" ]; then V="video_test$i.mp4"; else V="../video_test$i.mp4"; fi
  echo "=== video_test$i ($V) ==="
  "$PY" segment_and_overlay_vi_v3.py "$V" \
    --model qwen3-vl:8b-instruct --skills ../../skill_ontology/skills.yaml --names-vi ../../skill_ontology/skill_names_vi.yaml \
    --out "test_qwen3/annot_test${i}_qwen3.mp4" \
    --log "test_qwen3/log_test${i}_qwen3.json" \
    --task-log "test_qwen3/task_log_test${i}_qwen3.json" \
    2>&1 | tee "test_qwen3/run_test${i}.log"
done
