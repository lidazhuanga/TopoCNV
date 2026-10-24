"""Anatomical prior channels (protocol v0.4 §4, V4_PLAN §4.2). Built per outer fold k from T_k labels only.

(a) Stage-1 class-4 probability.
(b) distance to the predicted foramina (cleaned Stage-1 class 4): 1 - min(d, 30 mm) / 30 mm.
(c) population atlas of Y. Template = the T_k case whose GT class-4 side centroids are the median
    distance apart. Each T_k case is aligned to the template per side by similarity ICP on its GT
    class-4 points and its Y is resampled into template space; the atlas is the mean. A target case is
    aligned by ICP on its *predicted* class-4 points and the atlas is resampled back. Cases in T_k get
    the leave-one-out atlas (their own Y removed).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .common import LR_AXIS, SPACING, box_shape, box_slices, expand_box, mask_bbox
from .roi import head_midline, margin_vox, side_slices

DIST_TRUNC_MM = 30.0
TEMPLATE_MARGIN_MM = 45.0
MIN_ICP_POINTS = 20
MAX_ICP_POINTS = 4000
SCALE_BOUNDS = (0.8, 1.25)
ATLAS_ROI_THR = 0.02  # atlas support threshold for the ROI (dev check 2026-10-23, V4_DEV_LOG)
# Target cases are aligned rigidly (scale fixed at 1): their predicted class-4 cloud is often partial,
# and a free scale then shrinks the atlas (compared on training cases only, V4_DEV_LOG 2026-10-23).
TARGET_SCALE_BOUNDS = (1.0, 1.0)


# ---------------- similarity ICP ----------------

def umeyama(src, dst):
    """Least-squares similarity dst ~ s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    cov = Y.T @ X / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_s = (X ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def apply_sim(T, pts):
    s, R, t = T
    return s * pts @ R.T + t


def icp_similarity(src, dst, n_iter=60, trim=0.9, scale_bounds=SCALE_BOUNDS, tol=1e-5, seed=0):
    """Trimmed point-to-point ICP with a similarity transform; returns ((s, R, t), rmse_mm)."""
    rng = np.random.RandomState(seed)
    if len(src) > MAX_ICP_POINTS:
        src = src[rng.choice(len(src), MAX_ICP_POINTS, replace=False)]
    tree = cKDTree(dst)
    T = (1.0, np.eye(3), dst.mean(0) - src.mean(0))
    prev = np.inf
    rmse = np.inf
    for _ in range(n_iter):
        d, idx = tree.query(apply_sim(T, src))
        keep = d <= np.quantile(d, trim)
        s, R, t = umeyama(src[keep], dst[idx[keep]])
        if not scale_bounds[0] <= s <= scale_bounds[1]:
            s = float(np.clip(s, *scale_bounds))
            t = dst[idx[keep]].mean(0) - s * R @ src[keep].mean(0)
        T = (s, R, t)
        rmse = float(np.sqrt((d[keep] ** 2).mean()))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return T, rmse


def mask_points_mm(mask, spacing=SPACING, offset=(0, 0, 0)):
    idx = np.argwhere(mask)
    return (idx + np.asarray(offset)) * np.asarray(spacing, dtype=np.float64)


def side_points(mask_c4, mid, spacing=SPACING):
    """Class-4 points (mm, full-volume frame) of side 0 and side 1."""
    out = []
    for a0, a1 in side_slices(mask_c4.shape, mid):
        sub = np.zeros_like(mask_c4)
        sub[:, a0:a1] = mask_c4[:, a0:a1]
        out.append(mask_points_mm(sub, spacing))
    return out


# ---------------- prior (a) and (b) ----------------

def dist_prior(c4_clean_box, spacing=SPACING, trunc_mm=DIST_TRUNC_MM):
    if not c4_clean_box.any():
        return np.zeros(c4_clean_box.shape, dtype=np.float32)
    d = ndi.distance_transform_edt(~c4_clean_box, sampling=spacing)
    return (1.0 - np.minimum(d, trunc_mm) / trunc_mm).astype(np.float32)


def to_uint8(x):
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


# ---------------- prior (c): atlas ----------------

def side_c4_centroids(lab, image):
    mid = head_midline(image, None)
    pts = side_points(np.asarray(lab) == 4, mid)
    return mid, [p.mean(0) if len(p) else None for p in pts]


def choose_template(stats):
    """stats: {case: (mid, [c0, c1])}. Template = case with the median inter-centroid distance."""
    d = {c: float(np.linalg.norm(v[1][0] - v[1][1])) for c, v in stats.items()
         if v[1][0] is not None and v[1][1] is not None}
    med = np.median(list(d.values()))
    return min(sorted(d), key=lambda c: abs(d[c] - med))


class Atlas:
    """Per-side population atlas of Y in a template case's voxel grid."""

    def __init__(self, template_lab, template_image, spacing=SPACING):
        self.spacing = np.asarray(spacing, dtype=np.float64)
        lab = np.asarray(template_lab)
        self.shape = lab.shape
        self.mid = head_midline(template_image, None)
        lm = (lab == 1) | (lab == 4)
        c4 = lab == 4
        self.c4_pts = side_points(c4, self.mid, spacing)
        self.boxes, self.halfmasks = [], []
        for s, (a0, a1) in enumerate(side_slices(self.shape, self.mid)):
            sub = np.zeros_like(lm)
            sub[:, a0:a1] = lm[:, a0:a1]
            bb = mask_bbox(sub)
            box = expand_box(bb, margin_vox(TEMPLATE_MARGIN_MM, spacing), self.shape)
            self.boxes.append(box)
            lr = np.arange(box[LR_AXIS][0], box[LR_AXIS][1])
            half = (lr >= a0) & (lr < a1)
            self.halfmasks.append(half[None, :, None])
        self.sum = [np.zeros(box_shape(b), np.float32) for b in self.boxes]
        self.n = 0
        self.warped = {}

    def _grid_mm(self, box):
        zz, yy, xx = [np.arange(b0, b1) for b0, b1 in box]
        g = np.stack(np.meshgrid(zz, yy, xx, indexing="ij"), axis=-1).reshape(-1, 3)
        return g * self.spacing

    def add_case(self, case_id, lab, image, keep_for_loo=True):
        self.add_warped(case_id, self.warp_case(lab, image), keep_for_loo)

    def add_warped(self, case_id, ws, keep_for_loo=True):
        for s in (0, 1):
            if ws[s] is not None:
                self.sum[s] += ws[s].astype(np.float32)
        self.n += 1
        if keep_for_loo:
            self.warped[case_id] = ws

    def warp_case(self, lab, image):
        """A training case's GT Y in template space (per side, ICP on GT class 4); float16 per side."""
        lab = np.asarray(lab)
        y = ((lab >= 1) & (lab <= 4)).astype(np.float32)
        mid = head_midline(image, None)
        pts = side_points(lab == 4, mid, self.spacing)
        out = []
        for s in (0, 1):
            if len(pts[s]) < MIN_ICP_POINTS or len(self.c4_pts[s]) < MIN_ICP_POINTS:
                out.append(None)
                continue
            T, _ = icp_similarity(self.c4_pts[s], pts[s])  # template mm -> case mm
            q = apply_sim(T, self._grid_mm(self.boxes[s])) / self.spacing
            a0, a1 = side_slices(lab.shape, mid)[s]
            ok = (q[:, LR_AXIS] >= a0 - 0.5) & (q[:, LR_AXIS] < a1 - 0.5)
            vals = ndi.map_coordinates(y, q.T, order=1, mode="constant", cval=0.0) * ok
            w = vals.reshape(box_shape(self.boxes[s])).astype(np.float32) * self.halfmasks[s]
            out.append(w.astype(np.float16))
        return out

    def side_atlas(self, s, exclude=None):
        n = self.n
        tot = self.sum[s]
        if exclude is not None and exclude in self.warped:
            w = self.warped[exclude][s]
            n -= 1
            if w is not None:
                tot = tot - w.astype(np.float32)
        return tot / max(n, 1)

    def target_transforms(self, c4_pred_mask, image):
        """Per side: (T target mm -> template mm or None, info). Rigid ICP on the predicted class-4 points of
        that side; a side with too few points uses one bilateral ICP; no points at all -> None."""
        mid = head_midline(image, None)
        pts = side_points(c4_pred_mask, mid, self.spacing)
        all_t = np.concatenate([p for p in pts if len(p)], 0) if any(len(p) for p in pts) else np.zeros((0, 3))
        all_tpl = np.concatenate(self.c4_pts, 0)
        out = []
        for s in (0, 1):
            if len(pts[s]) >= MIN_ICP_POINTS:
                T, rmse = icp_similarity(pts[s], self.c4_pts[s], scale_bounds=TARGET_SCALE_BOUNDS)
                how = "side"
            elif len(all_t) >= MIN_ICP_POINTS:
                T, rmse = icp_similarity(all_t, all_tpl, scale_bounds=TARGET_SCALE_BOUNDS)
                how = "bilateral"
            else:
                out.append((None, {"side": s, "how": "none"}))
                continue
            out.append((T, {"side": s, "how": how, "scale": float(T[0]), "rmse_mm": rmse}))
        return mid, out

    def warp_to_target(self, c4_pred_mask, image, region, exclude=None, transforms=None):
        """Atlas resampled onto `region` of a target case, aligned by its predicted class-4 points.

        Returns (prior float32 on region, per-side info list)."""
        shape = c4_pred_mask.shape
        mid, tr = transforms if transforms is not None else self.target_transforms(c4_pred_mask, image)
        out = np.zeros(box_shape(region), np.float32)
        g = self._grid_mm(region)
        lr = g[:, LR_AXIS] / self.spacing[LR_AXIS]
        for s, (a0, a1) in enumerate(side_slices(shape, mid)):
            T = tr[s][0]
            if T is None:
                continue
            sel = (lr >= a0) & (lr < a1)
            q = apply_sim(T, g[sel]) / self.spacing - np.asarray([b[0] for b in self.boxes[s]])
            vals = ndi.map_coordinates(self.side_atlas(s, exclude), q.T, order=1, mode="constant", cval=0.0)
            flat = out.reshape(-1)
            flat[np.nonzero(sel)[0]] = vals
        return out, [t[1] for t in tr]

    def support_bboxes(self, shape, transforms, exclude=None, thr=ATLAS_ROI_THR):
        """Per side: bounding box (target voxels) of the atlas support (atlas > thr) mapped into the
        target by the inverse ICP transform, or None. Used as the v4 ROI (protocol §4, 2026-10-23)."""
        _, tr = transforms
        out = []
        for s in (0, 1):
            T = tr[s][0]
            if T is None:
                out.append(None)
                continue
            sc, R, t = T
            sup = np.argwhere(self.side_atlas(s, exclude) > thr)
            if len(sup) == 0:
                out.append(None)
                continue
            q = (sup + np.asarray([b[0] for b in self.boxes[s]])) * self.spacing
            p = ((q - t) @ R) / sc / self.spacing  # template mm -> target voxels (inverse similarity)
            lo, hi = np.floor(p.min(0)).astype(int), np.ceil(p.max(0)).astype(int) + 1
            out.append([[max(int(a), 0), min(int(b), int(n))] for a, b, n in zip(lo, hi, shape)])
        return out


def prior_box_from_rec(rec, margin):
    from .common import union_box
    from .roi import roi_boxes
    boxes = roi_boxes(rec, margin)
    return union_box(boxes) if boxes else None


def build_case_priors(p4, c4_clean, image, prior_box, atlas=None, exclude=None, transforms=None):
    """(3, *box) uint8 priors a/b/c on prior_box, and atlas warp info."""
    sl = box_slices(prior_box)
    a = np.asarray(p4[sl], dtype=np.float32)  # probabilities in [0, 1]
    b = dist_prior(c4_clean[sl])
    if atlas is not None:
        c, info = atlas.warp_to_target(c4_clean, image, prior_box, exclude=exclude, transforms=transforms)
    else:
        c, info = np.zeros_like(a), []
    return np.stack([to_uint8(a), to_uint8(b), to_uint8(c)]), info


def atlas_roi_record(rec, atlas, c4_clean, image, exclude=None, thr=ATLAS_ROI_THR, margins=None):
    """Replace the landmark side boxes of an ROI record by atlas-support boxes (protocol §4 as revised
    2026-10-23: landmarks place the atlas by ICP, the atlas support + m mm is the ROI). The landmark
    boxes are kept under 'landmark_boxes'. A side without a transform gets the other side mirrored."""
    from .common import MARGINS_MM
    from .roi import margin_vox, mirror_box
    margins = margins or MARGINS_MM
    shape = tuple(rec["shape"])
    transforms = atlas.target_transforms(c4_clean, image)
    bbs = atlas.support_bboxes(shape, transforms, exclude, thr)
    srcs = ["atlas" if b is not None else "none" for b in bbs]
    for s in (0, 1):
        if bbs[s] is None and bbs[1 - s] is not None:
            bbs[s] = mirror_box(bbs[1 - s], transforms[0], shape)
            srcs[s] = "mirror"
    for s, sd in enumerate(rec["sides"]):
        sd["landmark_boxes"] = sd["boxes"]
        sd["landmark_source"] = sd["source"]
        sd["atlas_bbox"] = bbs[s]
        sd["source"] = srcs[s]
        sd["boxes"] = {str(m): (expand_box(bbs[s], margin_vox(m), shape) if bbs[s] is not None else None)
                       for m in margins}
    rec["roi_found"] = any(b is not None for b in bbs)
    rec["roi_method"] = {"name": "atlas_support", "thr": thr}
    return rec, transforms
