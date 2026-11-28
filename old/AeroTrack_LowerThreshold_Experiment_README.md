# AeroTrack lower-threshold single-video experiment

## Purpose
This is a controlled, more permissive matching experiment for the existing `drone_video_imgsz960` tracking output. It does not rerun Part 1 or fine-tune YOLO.

## Run order
1. Part 7 — lower-threshold ReID
2. Part 8 — counting dashboard
3. Part 9 — lower-threshold tracklet consolidation
4. Part 10 — manual review using a separate SQLite database

Part 1 does not need to be rerun as long as `outputs/active_video_config.json` still points to `drone_video_imgsz960` and its tracking log exists.

## Experimental settings

### Part 7
- AUTO_LINK_THRESHOLD: 0.74 (was 0.84)
- REVIEW_THRESHOLD: 0.70 (was 0.75)
- MIN_SIMILARITY_MARGIN: 0.02 (unchanged)
- Mutual-best and non-overlapping same-video tracklet constraints remain enabled.

### Part 9
- DIRECT_MATCH_THRESHOLD: 0.74 (was 0.83)
- DIRECT_MIN_MARGIN: 0.015 (unchanged)
- CHAIN_EXTENSION_THRESHOLD: 0.82 (was 0.84)
- CHAIN_EXTENSION_MIN_MARGIN: 0.01 (was 0.015)
- CHAIN_EXTENSION_MAX_RANK: 2 (unchanged)
- CLUSTER_MIN_PAIR_SIMILARITY: 0.74 (was 0.78)
- REVIEW_THRESHOLD: 0.70 (was 0.75)

The lower values may produce false merges. The safeguards remain: temporal overlap blocks, hard-negative manual decisions, top-1/top-2 chain rule, and complete-link cluster consistency. Review each new multi-tracklet group.

## Part 10 database isolation
Part 10 writes to `*_identity_registry_lower_threshold_v2.sqlite`, not the earlier baseline database. This matters because the original Part 10 database preserves its original Part 9 assignments after first initialization.

## Before running
Back up `outputs/drone_video_imgsz960/part7_global_identity`, `part8_counting_dashboard`, `part9_tracklet_consolidation`, and `part10_identity_review` if you want to preserve the baseline outputs. Part 7–9 write their outputs into the same experiment folders. Part 10 keeps the baseline database and creates a separate lower-threshold database.

## Interpretation
The new Global ID count is an experimental result, not ground truth. A lower count is not automatically better: incorrect merges can undercount people. Compare the new groups with representative crops and record manual SAME/DIFFERENT decisions.
