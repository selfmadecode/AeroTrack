# AeroTrack — Single-Video Workflow

This package contains the five notebooks for the active single-video pipeline. It processes one recording at a time. The default input is `data/people.mp4`.

## Run order

1. `AeroTrack_SingleVideo_Part1_Detection_Tracking.ipynb`
2. `AeroTrack_SingleVideo_Part7_ReIdentification.ipynb`
3. `AeroTrack_SingleVideo_Part8_Counting_Dashboard.ipynb`
4. `AeroTrack_SingleVideo_Part9_Tracklet_Consolidation.ipynb`
5. `AeroTrack_SingleVideo_Part10_Manual_Identity_Review.ipynb`

Do not run Parts 2–6 as part of this execution workflow. Fine-tuning (Part 2) is optional and is excluded because the pretrained `yolo26n.pt` baseline was better on the current test video. Parts 5–6 are separate VisDrone ReID evaluation experiments, not runtime prerequisites. Part 11 is not used.

## Project layout

Keep the notebooks in the project root and the selected source video in `data/`:

```text
AeroTrack/
  data/
    people.mp4
  outputs/
  AeroTrack_SingleVideo_Part1_Detection_Tracking.ipynb
  AeroTrack_SingleVideo_Part7_ReIdentification.ipynb
  AeroTrack_SingleVideo_Part8_Counting_Dashboard.ipynb
  AeroTrack_SingleVideo_Part9_Tracklet_Consolidation.ipynb
  AeroTrack_SingleVideo_Part10_Manual_Identity_Review.ipynb
```

## Select a different video

Change only `VIDEO_FILENAME` in Part 1 Section 1. For example:

```python
VIDEO_FILENAME = "people.mp4"
```

Part 1 automatically derives `VIDEO_ID`, creates `outputs/<video_id>/`, writes the detailed tracking CSV and saves `outputs/active_video_config.json`. Parts 7–10 read this configuration automatically; do not edit their video paths separately.

Set `VIDEO_START_SECONDS` and `VIDEO_DURATION_SECONDS` in Part 1 if you want to process a segment. Leave `VIDEO_DURATION_SECONDS = None` to process through the end of the video. The segment start and frame numbering are included in the shared configuration so later parts read the same segment.

## What each stage produces

- **Part 1:** `outputs/<video_id>/<video_id>_track_log.csv`, annotated MP4, tracker summary, frame metrics and `outputs/active_video_config.json`.
- **Part 7:** `outputs/<video_id>/part7_global_identity/` — OSNet crops and embeddings, `global_identity_registry.csv`, `candidate_matches.csv`, manual review queue, similarity contact sheet and Global-ID annotated video.
- **Part 8:** `outputs/<video_id>/part8_counting_dashboard/` — frame count timeline, visibility events, identity history, count summaries, dashboard and count-annotated MP4.
- **Part 9:** `outputs/<video_id>/part9_tracklet_consolidation/` — consolidated identity registry, identity-group report, merge events, manual merge decisions, tracklet-quality report and recalibrated counts.
- **Part 10:** `outputs/<video_id>/part10_identity_review/` — manual identity review, video-specific persistent SQLite database, audit log, persistent identity groups, final count timeline and dashboard.

Each video has its own output subfolder. Processing a different video will not overwrite another video's Part 7–10 reports.

## Re-identification behaviour

Part 7 compares tracklets **within the same video** when their observed frame intervals do not overlap. For example, if a person has local tracker ID 4, disappears, and later returns as tracker ID 19, the two tracklets can be compared. Overlapping tracklets are excluded from matching.

The OSNet similarity and automatic merge thresholds are experimental. A Global ID is a provisional identity group, not verified ground truth. Review candidate crops before treating the final number as an accurate count of real people.

## Environment notes

- Part 1 uses the existing Ultralytics YOLO installation and `yolo26n.pt`.
- Part 7 uses `from reid import ReIDModel` (the ReID package used in the earlier OSNet notebooks). It does **not** use `torchreid` and does not require the Part 11 installation.
- Do not reinstall PyTorch, CUDA or Torchvision to run this workflow. Use the Python environment that already runs YOLO and your earlier OSNet/ReID experiment.
- Part 10 uses `ipywidgets` for side-by-side review when available; it also exports CSV review files for a manual fallback.

## Important persistence note

Part 10 writes a video-specific SQLite database at:

`outputs/<video_id>/part10_identity_review/<video_id>_identity_registry.sqlite`

Keep this file to preserve approved identity assignments and decision history for that video. If you intentionally want a completely fresh identity review for the same video, back up the SQLite database and Part 9's `manual_merge_decisions.csv` before removing or resetting them.

## Execution advice

Run each notebook from top to bottom in the order above. If you change the source video, segment, tracking settings or detector, rerun Part 1 and then rerun Parts 7–10 in order. If you only make manual identity decisions in Part 10, rerun the count/dashboard sections in Part 10; you do not need to rerun detection or OSNet.
