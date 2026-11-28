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
    """Add world_x/world_y (foot point in frame-1 coordinates) to a tracking log
    with 1-based `frame`. NOTE: the frame-1 image plane is not metric when the camera
    tilts, so world coordinates are only used to *transfer* points between frames
    (see spatial_compatibility); never compare raw world distances or heights."""
    log = track_log.copy()
    idx = np.clip(log["frame"].astype(int).to_numpy() - 1, 0, len(Hs) - 1)
    H = Hs[idx]
    foot = np.stack([(log["x1"] + log["x2"]).to_numpy() / 2.0,
                     log["y2"].to_numpy(), np.ones(len(log))], axis=1)
    w = np.einsum("nij,nj->ni", H, foot)
    log["world_x"] = w[:, 0] / w[:, 2]
    log["world_y"] = w[:, 1] / w[:, 2]
    return log


def _apply(H, xy):
    p = H @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


# ----------------------------------------------------------------------------
# 3. Junk (non-person / fragment) tracklets
# ----------------------------------------------------------------------------
@dataclass
class JunkConfig:
    min_obs_lowconf: int = 20       # short AND low-confidence -> junk
    lowconf: float = 0.35
    max_aspect: float = 1.2         # width/height; people are tall
    aspect_max_obs: int = 40
    oversize_factor: float = 1.8    # box much taller than a typical person -> group/object
    oversize_max_obs: int = 40


def flag_junk_tracklets(track_log, key_column="tracklet_key", cfg=JunkConfig()):
    """Per-tracklet table with a `junk_reason` ('' = keep)."""
    g = track_log.assign(_w=track_log["x2"] - track_log["x1"],
                         _h=track_log["y2"] - track_log["y1"])
    t = g.groupby(key_column).agg(observations=("frame", "size"),
                                  mean_confidence=("confidence", "mean"),
                                  median_w=("_w", "median"),
                                  median_h=("_h", "median")).reset_index()
    t["aspect"] = t["median_w"] / t["median_h"].clip(lower=1)
    typical_h = float(t["median_h"].median())
    reasons = []
    for _, r in t.iterrows():
        out = []
        if r.observations < cfg.min_obs_lowconf and r.mean_confidence < cfg.lowconf:
            out.append("short_lowconf")
        if r.aspect > cfg.max_aspect and r.observations < cfg.aspect_max_obs:
            out.append("wide_short")
        if r.median_h > cfg.oversize_factor * typical_h and r.observations < cfg.oversize_max_obs:
            out.append("oversized_short")
        reasons.append(",".join(out))
    t["junk_reason"] = reasons
    return t


# ----------------------------------------------------------------------------
# 4. Short-gap spatial gating (camera-compensated, measured in the later frame)
# ----------------------------------------------------------------------------
@dataclass
class SpatialConfig:
    edge_obs: int = 8
    base_radius_heights: float = 0.75   # localisation noise, in body heights
    max_speed_heights_per_sec: float = 2.5
    drift_px_per_sec: float = 8.0       # at 1920 px width (measured ~3-15 px/s)
    px_scale: float = 1.0               # source width / 1920
    hard_factor: float = 1.5
    long_gap_seconds: float = 8.0       # beyond this chained motion is not trusted
    height_ratio_limit: float = 2.5     # IMAGE-space body height ratio, short gaps only
    max_bonus: float = 0.10
    bonus_halflife_seconds: float = 3.0
    min_appearance_for_bonus: float = 0.50
    # revisit (scene re-localisation) evidence for long gaps
    revisit_max_bonus: float = 0.20
    revisit_radius_heights: float = 1.5
    revisit_min_appearance: float = 0.35


def build_tracklet_motion_table(track_log, Hs, key_column="tracklet_key", cfg=SpatialConfig()):
    """Per tracklet: first/last frame, foot position (image coords of that frame) and
    image-space body height at both ends, plus reference observations for revisit
    localisation. Needs world_x/world_y from add_world_coordinates."""
    table = {}
    for key, g in track_log.groupby(key_column):
        g = g.sort_values("frame").reset_index(drop=True)
        n = len(g)
        head, tail = g.head(3), g.tail(3)
        f_s, f_e = int(g["frame"].iloc[0]), int(g["frame"].iloc[-1])
        w_s = np.array([head["world_x"].median(), head["world_y"].median()])
        w_e = np.array([tail["world_x"].median(), tail["world_y"].median()])
        start_img = _apply(np.linalg.inv(Hs[f_s - 1]), w_s)
        end_img = _apply(np.linalg.inv(Hs[f_e - 1]), w_e)
        hh = (g["y2"] - g["y1"]).to_numpy()

        def ref(i):
            r = g.iloc[int(np.clip(i, 0, n - 1))]
            return (int(r["frame"]), np.array([(r["x1"] + r["x2"]) / 2.0, r["y2"]]),
                    float(r["y2"] - r["y1"]))

        table[key] = {
            "first_frame": f_s, "last_frame": f_e,
            "start_img": start_img, "end_img": end_img,
            "start_h": float(np.median(hh[:cfg.edge_obs])), "end_h": float(np.median(hh[-cfg.edge_obs:])),
            "ref_start": ref(min(2, n - 1)), "ref_mid": ref(n // 2), "ref_end": ref(n - 1 - min(2, n - 1)),
        }
    return table


def spatial_compatibility(meta_a, meta_b, fps, Hs, cfg=SpatialConfig()):
    """Compare two NON-overlapping tracklets using camera-compensated position.
    Returns dict(rejected, distance, allowed, proximity, gap_seconds, reason).
    Gaps longer than cfg.long_gap_seconds are neither rejected nor rewarded here
    (see RevisitLocalizer for those)."""
    if meta_a["last_frame"] < meta_b["first_frame"]:
        earlier, later = meta_a, meta_b
    else:
        earlier, later = meta_b, meta_a
    gap_frames = max(1, later["first_frame"] - earlier["last_frame"])
    gap_s = gap_frames / fps
    if gap_s > cfg.long_gap_seconds:
        return dict(rejected=False, distance=np.nan, allowed=np.nan, proximity=0.0,
                    gap_seconds=gap_s, reason="long_gap")
    h_e, h_l = max(earlier["end_h"], 1.0), max(later["start_h"], 1.0)
    if max(h_e, h_l) / min(h_e, h_l) > cfg.height_ratio_limit:
        return dict(rejected=True, distance=np.nan, allowed=np.nan, proximity=0.0,
                    gap_seconds=gap_s, reason="height_mismatch")
    # earlier end point -> world (frame 1) -> later start frame image
    H_rel = np.linalg.inv(Hs[later["first_frame"] - 1]) @ Hs[earlier["last_frame"] - 1]
    p = _apply(H_rel, earlier["end_img"])
    d = float(np.linalg.norm(later["start_img"] - p))
    h = 0.5 * (h_e + h_l)
    allowed = (cfg.base_radius_heights * h + cfg.max_speed_heights_per_sec * h * gap_s
               + cfg.drift_px_per_sec * cfg.px_scale * gap_s)
    if d > cfg.hard_factor * allowed:
        return dict(rejected=True, distance=d, allowed=float(allowed), proximity=0.0,
                    gap_seconds=gap_s, reason="too_far")
    proximity = max(0.0, 1.0 - d / allowed) / (1.0 + gap_s / cfg.bonus_halflife_seconds)
    return dict(rejected=False, distance=d, allowed=float(allowed), proximity=float(proximity),
                gap_seconds=gap_s, reason="ok")


def fuse_similarity(appearance, spatial, cfg=SpatialConfig()):
    """appearance cosine + bounded spatial bonus (never below appearance)."""
    if appearance < cfg.min_appearance_for_bonus:
        return float(appearance), 0.0
    bonus = cfg.max_bonus * spatial["proximity"]
    return float(appearance + bonus), float(bonus)


# ----------------------------------------------------------------------------
# 5. Revisit localisation: drift-free position evidence for LONG gaps
# ----------------------------------------------------------------------------
class RevisitLocalizer:
    """When the camera comes back to an area, register the two frames directly
    (SIFT + RANSAC homography on the static scene). A person who stands in the same
    spot of the scene then lands on the same spot, even 60 s later, which chained
    motion cannot tell us because it drifts. Evidence is positive-only: people who
    moved are simply not rewarded."""

    def __init__(self, video_path, start_seconds=0.0, scale=0.5, n_features=2500,
                 ratio=0.75, min_inliers=30, min_inlier_ratio=0.25, mask_top_left=None):
        self.video_path, self.start_seconds = video_path, float(start_seconds)
        self.scale, self.n_features, self.ratio = scale, n_features, ratio
        self.min_inliers, self.min_inlier_ratio = min_inliers, min_inlier_ratio
        self.mask_top_left = mask_top_left   # (h, w) in scaled pixels to ignore (e.g. a HUD)
        self._feat, self._cache = {}, {}
        self._sift = cv2.SIFT_create(n_features)
        self._flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))

    def prepare(self, frames):
        """Compute features for the given 1-based segment frame numbers (one sequential pass)."""
        need = sorted(set(int(f) for f in frames) - set(self._feat))
        if not need:
            return
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, self.start_seconds * 1000)
        k, wanted = 0, set(need)
        while k < need[-1]:
            ok, frame = cap.read()
            if not ok:
                break
            k += 1
            if k not in wanted:
                continue
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            if self.mask_top_left:
                g[:self.mask_top_left[0], :self.mask_top_left[1]] = 0
            kp, des = self._sift.detectAndCompute(g, None)
            self._feat[k] = (np.float32([p.pt for p in kp]) if kp else np.zeros((0, 2), np.float32),
                             des)
        cap.release()

    def register(self, frame_a, frame_b):
        """Homography mapping pixels of frame_b -> pixels of frame_a (full resolution),
        or None if the views do not overlap enough."""
        key = (frame_a, frame_b)
        if key in self._cache:
            return self._cache[key]
        result = None
        fa, fb = self._feat.get(frame_a), self._feat.get(frame_b)
        if fa is not None and fb is not None and fa[1] is not None and fb[1] is not None \
                and len(fa[1]) >= 20 and len(fb[1]) >= 20:
            matches = self._flann.knnMatch(fb[1], fa[1], k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
            if len(good) >= 15:
                src = fb[0][[m.queryIdx for m in good]]
                dst = fa[0][[m.trainIdx for m in good]]
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if H is not None and mask is not None:
                    inl = int(mask.sum())
                    det = np.linalg.det(H[:2, :2])
                    if (inl >= self.min_inliers and inl / len(good) >= self.min_inlier_ratio
                            and 0.4 < det < 2.5 and abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3):
                        S = np.diag([self.scale, self.scale, 1.0])
                        result = (np.linalg.inv(S) @ H @ S, inl)
        self._cache[key] = result
        return result

    def pair_evidence(self, meta_a, meta_b, cfg=SpatialConfig()):
        """Best (smallest) body-height-normalised distance between the two tracklets'
        positions after scene registration. Returns dict or None."""
        a, b = (meta_a, meta_b) if meta_a["last_frame"] < meta_b["first_frame"] else (meta_b, meta_a)
        best = None
        for ra, rb in (("ref_end", "ref_start"), ("ref_mid", "ref_mid")):
            fa, foot_a, h_a = a[ra]
            fb, foot_b, h_b = b[rb]
            reg = self.register(fa, fb)
            if reg is None:
                continue
            H, inl = reg
            d = float(np.linalg.norm(_apply(H, foot_b) - foot_a))
            d_rel = d / max(0.5 * (h_a + h_b), 1.0)
            if best is None or d_rel < best["d_rel"]:
                best = dict(d_rel=d_rel, distance=d, inliers=inl, refs=f"{ra}/{rb}")
        return best

    @staticmethod
    def bonus(d_rel, cfg=SpatialConfig()):
        return float(cfg.revisit_max_bonus * max(0.0, 1.0 - d_rel / cfg.revisit_radius_heights))
