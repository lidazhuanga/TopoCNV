"""按训练折构建双侧 Y 人群图谱，支持留一先验与目标病例刚性配准。"""

import numpy as np
from scipy import ndimage as ndi

from topocnv.data.utils import LR_AXIS, SPACING
from topocnv.priors.icp import MIN_ICP_POINTS, TARGET_SCALE_BOUNDS, apply_sim, icp_similarity, side_points
from topocnv.priors.roi import head_midline, margin_vox, side_slices
from topocnv.utils.box_ops import box_shape, expand_box, mask_bbox

TEMPLATE_MARGIN_MM = 45.0
ATLAS_ROI_THR = 0.02


def side_c4_centroids(lab, image):
    mid = head_midline(image, None)
    pts = side_points(np.asarray(lab) == 4, mid)
    return mid, [p.mean(0) if len(p) else None for p in pts]


def choose_template(stats):
    """选取双侧质心距离最接近中位数的训练病例作为模板。"""
    d = {c: float(np.linalg.norm(v[1][0] - v[1][1])) for c, v in stats.items()
         if v[1][0] is not None and v[1][1] is not None}
    med = np.median(list(d.values()))
    return min(sorted(d), key=lambda c: abs(d[c] - med))


class Atlas:
    """模板病例体素网格中的双侧 Y 人群图谱。"""

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
        """将训练病例 GT Y 按双侧 GT 类别 4 ICP 映射到模板空间。"""
        lab = np.asarray(lab)
        y = ((lab >= 1) & (lab <= 4)).astype(np.float32)
        mid = head_midline(image, None)
        pts = side_points(lab == 4, mid, self.spacing)
        out = []
        for s in (0, 1):
            if len(pts[s]) < MIN_ICP_POINTS or len(self.c4_pts[s]) < MIN_ICP_POINTS:
                out.append(None)
                continue
            T, _ = icp_similarity(self.c4_pts[s], pts[s])
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
        """按预测类别 4 做双侧刚性 ICP，点数不足时使用双侧合并点云。"""
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
        """将图谱采样到目标 region，返回 float32 先验及双侧配准信息。"""
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
        """将图谱支撑区经逆 ICP 映射为目标体素包围盒，缺失侧返回 None。"""
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
            p = ((q - t) @ R) / sc / self.spacing
            lo, hi = np.floor(p.min(0)).astype(int), np.ceil(p.max(0)).astype(int) + 1
            out.append([[max(int(a), 0), min(int(b), int(n))] for a, b, n in zip(lo, hi, shape)])
        return out


__all__ = ["TEMPLATE_MARGIN_MM", "ATLAS_ROI_THR", "side_c4_centroids", "choose_template", "Atlas"]
