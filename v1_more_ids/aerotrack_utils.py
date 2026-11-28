"""AeroTrack helpers: camera-motion compensation + spatio-temporal tracklet gating.

Why this exists
---------------
The drone camera moves (in the sample video the view shifts by >1000 px), so image
coordinates say nothing about where a person really is. We estimate a frame-to-frame
homography from the static background, chain them into a *world frame* (= frame 1),
and express every detection's foot point in that frame. Two tracklets of the same
person must then be spatially plausible (the person cannot teleport), which lets us
(1) reject impossible ReID matches and (2) rescue matches whose OSNet score is weak
because the crops are tiny.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2


# ----------------------------------------------------------------------------
# 1. Camera motion
# ----------------------------------------------------------------------------
def _valid_homography(H, width, height):
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H[:2, :2])
    if not (0.75 < det < 1.33):
        return False
    if abs(H[2, 0]) > 5e-4 or abs(H[2, 1]) > 5e-4:
        return False
    if abs(H[0, 2]) > 0.35 * width or abs(H[1, 2]) > 0.35 * height:
        return False
    return True


def estimate_camera_motion(video_path, start_seconds=0.0, max_frames=None,
                           scale=0.5, max_corners=800, progress_every=500):
    """Return (Hs, quality_df).

    Hs[k-1] is a 3x3 matrix mapping pixel coordinates of frame k (1-based, same
    numbering as the tracking log) into the coordinates of frame 1.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_seconds) * 1000)

    S = np.diag([scale, scale, 1.0])
    S_inv = np.diag([1 / scale, 1 / scale, 1.0])

    def prep(frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first frame.")
    height, width = frame.shape[:2]
    prev = prep(frame)
    cum = np.eye(3)
    Hs = [cum.copy()]
    rows = [{"frame": 1, "inliers": 0, "method": "reference"}]
    k = 1
    while max_frames is None or k < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        k += 1
        cur = prep(frame)
        H_step, inliers, method = np.eye(3), 0, "identity"
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners,
                                     qualityLevel=0.01, minDistance=8)
        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None,
                                                 winSize=(21, 21), maxLevel=3)
            m = st.ravel() == 1
            if m.sum() >= 12:
                src, dst = p1[m], p0[m]          # current -> previous
                Hc, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
                Hc_full = None if Hc is None else S_inv @ Hc @ S
                if _valid_homography(Hc_full, width, height):
                    H_step, inliers, method = Hc_full, int(mask.sum()), "homography"
                else:
                    A, mask = cv2.estimateAffinePartial2D(
                        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
                    if A is not None:
                        Hc_full = S_inv @ np.vstack([A, [0, 0, 1]]) @ S
                        if _valid_homography(Hc_full, width, height):
                            H_step, inliers, method = Hc_full, int(mask.sum()), "similarity"
        cum = cum @ H_step
        Hs.append(cum.copy())
        rows.append({"frame": k, "inliers": inliers, "method": method})
        prev = cur
        if progress_every and k % progress_every == 0:
            print(f"camera motion: {k} frames")
    cap.release()
    return np.asarray(Hs), pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. World coordinates
# ----------------------------------------------------------------------------
def add_world_coordinates(track_log, Hs):
    """Add world_x/world_y (foot point in frame-1 coordinates) and world_h
    (box height in frame-1 pixels) to a tracking log with 1-based `frame`."""
    log = track_log.copy()
    idx = np.clip(log["frame"].astype(int).to_numpy() - 1, 0, len(Hs) - 1)
    H = Hs[idx]
    foot = np.stack([(log["x1"] + log["x2"]).to_numpy() / 2.0,
                     log["y2"].to_numpy(), np.ones(len(log))], axis=1)
    w = np.einsum("nij,nj->ni", H, foot)
    log["world_x"] = w[:, 0] / w[:, 2]
    log["world_y"] = w[:, 1] / w[:, 2]
    scale = np.sqrt(np.abs(np.linalg.det(H[:, :2, :2])))
    log["world_h"] = (log["y2"] - log["y1"]).to_numpy() * scale
    return log


# ----------------------------------------------------------------------------
# 3. Tracklet motion table + spatial compatibility
# ----------------------------------------------------------------------------
@dataclass
class SpatialConfig:
    edge_obs: int = 8                   # observations used to describe a tracklet's start/end
    base_radius_heights: float = 0.75   # localisation noise, in body heights
    max_speed_heights_per_sec: float = 2.5   # running ~2.3 body heights/s
    drift_px_per_sec: float = 8.0       # measured chained-homography drift on the sample video was ~3-15 px/s
    hard_factor: float = 1.5            # reject if distance > hard_factor * allowed
    long_gap_seconds: float = 8.0       # beyond this, position says nothing
    height_ratio_limit: float = 2.0     # reject if body sizes differ by more than this
    velocity_horizon_frames: int = 15
    max_bonus: float = 0.10             # max amount added to the appearance score
    bonus_halflife_seconds: float = 3.0
    min_appearance_for_bonus: float = 0.50


def build_tracklet_motion_table(track_log, key_column="tracklet_key", cfg=SpatialConfig()):
    """Per tracklet: start/end world position, end/start velocity, body height."""
    table = {}
    for key, g in track_log.groupby(key_column):
        g = g.sort_values("frame")
        head, tail = g.head(cfg.edge_obs), g.tail(cfg.edge_obs)

        def vel(part):
            if len(part) < 3:
                return np.zeros(2)
            f = part["frame"].to_numpy(float)
            if np.ptp(f) < 1:
                return np.zeros(2)
            vx = np.polyfit(f, part["world_x"].to_numpy(), 1)[0]
            vy = np.polyfit(f, part["world_y"].to_numpy(), 1)[0]
            return np.array([vx, vy])

        table[key] = {
            "first_frame": int(g["frame"].min()), "last_frame": int(g["frame"].max()),
            "start_xy": np.array([head["world_x"].median(), head["world_y"].median()]),
            "end_xy": np.array([tail["world_x"].median(), tail["world_y"].median()]),
            "start_vel": vel(head), "end_vel": vel(tail),
            "start_h": float(head["world_h"].median()), "end_h": float(tail["world_h"].median()),
        }
    return table


def spatial_compatibility(meta_a, meta_b, fps, cfg=SpatialConfig()):
    """Compare two NON-overlapping tracklets. Returns dict with
    rejected(bool), distance, allowed, proximity(0..1), reason."""
    if meta_a["last_frame"] < meta_b["first_frame"]:
        earlier, later = meta_a, meta_b
    else:
        earlier, later = meta_b, meta_a
    gap_frames = max(1, later["first_frame"] - earlier["last_frame"])
    gap_s = gap_frames / fps
    h_e, h_l = max(earlier["end_h"], 1.0), max(later["start_h"], 1.0)
    ratio = max(h_e, h_l) / min(h_e, h_l)
    if ratio > cfg.height_ratio_limit:
        return dict(rejected=True, distance=np.nan, allowed=np.nan, proximity=0.0,
                    gap_seconds=gap_s, reason="height_mismatch")
    if gap_s > cfg.long_gap_seconds:
        return dict(rejected=False, distance=np.nan, allowed=np.nan, proximity=0.0,
                    gap_seconds=gap_s, reason="long_gap_no_spatial_evidence")
    h = 0.5 * (h_e + h_l)
    pred = earlier["end_xy"] + earlier["end_vel"] * min(gap_frames, cfg.velocity_horizon_frames)
    d = min(np.linalg.norm(later["start_xy"] - pred),
            np.linalg.norm(later["start_xy"] - earlier["end_xy"]))
    allowed = (cfg.base_radius_heights * h
               + cfg.max_speed_heights_per_sec * h * gap_s
               + cfg.drift_px_per_sec * gap_s)
    if d > cfg.hard_factor * allowed:
        return dict(rejected=True, distance=float(d), allowed=float(allowed), proximity=0.0,
                    gap_seconds=gap_s, reason="too_far")
    proximity = max(0.0, 1.0 - d / allowed) * (1.0 / (1.0 + gap_s / cfg.bonus_halflife_seconds))
    return dict(rejected=False, distance=float(d), allowed=float(allowed),
                proximity=float(proximity), gap_seconds=gap_s, reason="ok")


def fuse_similarity(appearance, spatial, cfg=SpatialConfig()):
    """appearance cosine + bounded spatial bonus (never below appearance)."""
    if appearance < cfg.min_appearance_for_bonus:
        return float(appearance), 0.0
    bonus = cfg.max_bonus * spatial["proximity"]
    return float(appearance + bonus), float(bonus)
