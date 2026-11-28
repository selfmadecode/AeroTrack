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
