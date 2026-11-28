# AeroTrack — fixed re-identification pipeline

Copy everything in this folder into your AeroTrack project root (next to `data/` and `outputs/`).

## Run order
1. `AeroTrack_Fixed_Part1_Detection_Tracking.ipynb`  (must be rerun: new tracker settings + camera motion)
2. `AeroTrack_Fixed_Part7_ReIdentification.ipynb`
3. `AeroTrack_Fixed_Part8_Counting_Dashboard.ipynb`  (unchanged logic)
4. `AeroTrack_Fixed_Part9_Tracklet_Consolidation.ipynb`
5. `AeroTrack_Fixed_Part10_Manual_Identity_Review.ipynb`  (new SQLite file `..._fixed_v3.sqlite`, so old decisions do not leak in)

Part 1 writes to a new folder `outputs/<video>_v2_imgsz1280/`, so your baseline outputs are not overwritten.

## Why IDs were changing
| Cause | Fix |
|---|---|
| `conf=0.25` == tracker high threshold, so BoT-SORT's low-score association pass never ran; flickering tiny people became new IDs | `conf=0.08`, custom `aerotrack_botsort.yaml` (high 0.30 / low 0.08 / new 0.35) |
| Tracker memory 30 frames (1 s) | `track_buffer=90`, `match_thresh=0.90` |
| People ~25-60 px tall at imgsz 960 | `imgsz=1280` |
| ReID ignored tracklets with <5 observations or boxes <20x40 px; every ignored tracklet became its own person | limits now 3 observations, 8x18 px; crops upscaled to 128 px high; 8 crops per tracklet |
| Appearance-only matching on tiny crops (weak) and no position cue, although the camera moves >1000 px | homography camera motion -> `world_x/world_y`; impossible pairs rejected; nearby pairs get a bounded bonus (<=0.10) |
| Thresholds | Part 7 auto 0.72 / review 0.65; Part 9 direct 0.72, chain 0.80, complete-link floor 0.68, review 0.65 |

## Files
- `aerotrack_utils.py` — camera motion, world coordinates, spatial gating (`SpatialConfig` holds all knobs)
- Part 7 candidate CSV now has `appearance_similarity` (raw OSNet), `spatial_bonus`, `world_distance`, `spatial_reason`; `multi_mean_similarity` is the fused score used downstream.

## Tuning
- Still too many IDs: lower Part 9 `DIRECT_MATCH_THRESHOLD` / `CLUSTER_MIN_PAIR_SIMILARITY` in 0.02 steps, raise `SpatialConfig.max_bonus` to 0.15.
- Different people merged: raise those back, lower `max_bonus`, or mark pairs DIFFERENT in Part 10.
- Sanity check: the final unique count should be >= the maximum number of people visible at once (Part 8 dashboard). A count far above it means fragmentation remains.

## Verified vs not verified
Verified here: camera-motion estimation on your sample clip (all 2249 frame pairs solved, median ~590 inliers), spatial gating/fusion logic on synthetic tracklets, patched Part 7 candidate cells executed on mock data, all notebooks syntax-checked.
Not verified: the full pipeline end to end. The original `drone_video.mov`, YOLO weights, GPU and the `reid` package were not available here, so the unique-person counts must be confirmed by you.
Measured limitation: chained homographies drift about 3-15 px/s on the sample clip, so positions are only used for gaps up to 8 s.

---
# v2 update (after your first run: 41 tracklets -> 31 IDs)

## What was still wrong
1. **A bug in my first fix.** The spatial gate compared body heights in the frame-1 "world" plane. That plane is not metric when the drone tilts (a fixed-size person measured 0.18x to 1.7x across your clip), so it rejected **483 of the 642** possible tracklet pairs, 391 of them long-gap pairs, i.e. exactly the people who leave and re-enter the view (e.g. T2 vs T102, T3 vs T104).
2. **Appearance alone cannot re-link people after a long absence.** Late tracklets such as T102/T104/T110 scored only 0.45-0.56 against anything.
3. **Fragments were counted as people**: very short low-confidence tracks, wide-and-short boxes (benches), oversized boxes (groups).
4. **Complete-link cluster check (all pairs >= 0.68)** blocked most chains.

## What changed (rerun Parts 7, 8, 9, 10; Part 1 does not need rerunning)
- `aerotrack_utils.py`
  - Short gaps (<= 8 s): positions are compared in the later frame's own image, in body heights, via the camera motion between the two frames. No more world-plane heights.
  - **Long gaps: `RevisitLocalizer`.** When the camera comes back, the two frames are registered directly (SIFT + RANSAC). A person standing in the same spot of the scene gets a bonus up to 0.20, with no accumulated drift. Positive evidence only: people who moved are not penalised.
  - `flag_junk_tracklets`: short+low-confidence, wide+short, oversized+short tracklets.
- Part 7: junk tracklets go to `excluded_tracklets.csv`; a cleaned log `<video>_track_log_clean.csv` is written and `active_video_config.json` is pointed at it, so Parts 8-10 count the same set (the raw log is kept under `raw_track_log_path`). New candidate columns: `revisit_d_rel`, `spatial_reason`.
- Part 9: average-link cluster check (floor 0.50, mean 0.62) and a third merge pass, `average_link` (mean >= 0.62). Every merge is logged in `merge_events.csv` with its type.

## Verified here
- On your clip, frames 3 s and 74 s register with 126-128 inliers; the two people standing together at 74 s land within 0.24 body heights (about 10-15 px) of the two people standing together at 3 s, while a person elsewhere scores 10. So those are the same two people returning, and the revisit evidence recovers them.
- Patched Part 7 candidate cells run end to end on mock tracklets with your real video and camera motion.
- Replaying your uploaded candidate graph through the new Part 9 logic (without the revisit bonus and without the 483 wrongly rejected pairs, so a pessimistic estimate): 31 -> 26 identities after the junk filter -> 23 after the average-link pass.

## Not verified
The final count with real OSNet embeddings. The thresholds (average-link 0.62, floor 0.50, revisit bonus 0.20, radius 1.5 body heights) are starting points, not calibrated. Check `merge_events.csv` and the contact sheet; if people are wrongly merged raise `AVERAGE_LINK_THRESHOLD`, if returning people are still separate lower it by 0.02 steps. If your source video has an overlay, set `REVISIT_HUD_MASK` in Part 7.
