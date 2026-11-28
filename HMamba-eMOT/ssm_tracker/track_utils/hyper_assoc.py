"""
Hypergraph-inspired association cost for two-stage Hungarian matching.

C_total(T_i, D_j) = w_base * C_base + w_tri * C_tri + w_pred * C_pred

- C_base: binary hyperedge {T_i, D_j}  -> IoU + appearance
- C_tri:  ternary {T_i, D_prev, D_j}  -> motion direction consistency (H >= 2)
- C_pred: binary {T_pred, D_j}        -> Mamba predicted bbox IoU (H >= enable_time_thresh)
"""

import numpy as np

from .matching import ious


def _l2_normalize(feat):
    if feat is None:
        return None
    norm = np.linalg.norm(feat)
    if norm < 1e-6:
        return feat
    return feat / norm


def _cosine_distance(a, b):
    if a is None or b is None:
        return 0.0
    a = _l2_normalize(np.asarray(a, dtype=np.float64))
    b = _l2_normalize(np.asarray(b, dtype=np.float64))
    return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))


def _motion_cosine_cost(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return 1.0 - float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def _xywh_to_tlbr(xywh):
    ret = np.asarray(xywh, dtype=np.float64).copy()
    ret[:2] -= 0.5 * ret[2:]
    ret[2:] += ret[:2]
    return ret


def _build_base_cost(tracks, detections, alpha, beta):
    n_trk, n_det = len(tracks), len(detections)
    if n_trk == 0 or n_det == 0:
        return np.zeros((n_trk, n_det), dtype=np.float64)

    atlbrs = [track.tlbr for track in tracks]
    btlbrs = [det.tlbr for det in detections]
    c_iou = 1.0 - ious(atlbrs, btlbrs)

    if beta <= 0.0:
        return alpha * c_iou

    c_app = np.zeros((n_trk, n_det), dtype=np.float64)
    for i, track in enumerate(tracks):
        app_t = track.get_app_ema()
        if app_t is None:
            continue
        for j, det in enumerate(detections):
            if det.app_feat is None:
                continue
            c_app[i, j] = _cosine_distance(app_t, det.app_feat)

    return alpha * c_iou + beta * c_app


def _build_tri_cost(tracks, detections):
    n_trk, n_det = len(tracks), len(detections)
    c_tri = np.zeros((n_trk, n_det), dtype=np.float64)

    for i, track in enumerate(tracks):
        if len(track.memo_bank) < 2:
            continue
        v_prev = track.memo_bank[-1][:2] - track.memo_bank[-2][:2]
        last_center = track.memo_bank[-1][:2]
        for j, det in enumerate(detections):
            v_cand = det._xywh[:2] - last_center
            c_tri[i, j] = _motion_cosine_cost(v_prev, v_cand)

    return c_tri


def _build_pred_cost(tracks, detections, min_hist_len):
    n_trk, n_det = len(tracks), len(detections)
    c_pred = np.zeros((n_trk, n_det), dtype=np.float64)

    det_tlbrs = [det.tlbr for det in detections]
    for i, track in enumerate(tracks):
        if len(track.memo_bank) < min_hist_len or track.predicted_last_bbox is None:
            continue
        pred_tlbr = _xywh_to_tlbr(track.predicted_last_bbox)
        iou_row = ious(np.asarray([pred_tlbr]), np.asarray(det_tlbrs))[0]
        c_pred[i, :] = 1.0 - iou_row

    return c_pred


def build_cost_matrix(tracks, detections, cfgs, stage=1):
    """
    Build pairwise association cost matrix for Hungarian assignment.

    stage: 1 (tracked) or 2 (lost/unmatched), reserved for stage-specific overrides.
    """
    n_trk, n_det = len(tracks), len(detections)
    if n_trk == 0 or n_det == 0:
        return np.zeros((n_trk, n_det), dtype=np.float64)

    alpha = float(cfgs.get('assoc_alpha', 0.6))
    beta = float(cfgs.get('assoc_beta', 0.4))
    w_base = float(cfgs.get('assoc_w_base', 1.0))
    w_tri = float(cfgs.get('assoc_w_tri', 0.15))
    w_pred = float(cfgs.get('assoc_w_pred', 0.25))

    min_hist_tri = int(cfgs.get('assoc_min_hist_tri', 2))
    min_hist_pred = int(cfgs.get('enable_time_thresh', 6))

    if stage == 2:
        w_tri = float(cfgs.get('assoc_w_tri_stage2', w_tri))
        w_pred = float(cfgs.get('assoc_w_pred_stage2', w_pred))

    c_base = _build_base_cost(tracks, detections, alpha, beta)
    c_tri = _build_tri_cost(tracks, detections) if w_tri > 0.0 else 0.0
    c_pred = _build_pred_cost(tracks, detections, min_hist_pred) if w_pred > 0.0 else 0.0

    if isinstance(c_tri, float):
        c_total = w_base * c_base + w_pred * c_pred
    else:
        tri_mask = np.array([len(t.memo_bank) >= min_hist_tri for t in tracks], dtype=bool)
        c_tri_weighted = c_tri * tri_mask[:, None]
        c_total = w_base * c_base + w_tri * c_tri_weighted + w_pred * c_pred

    return np.clip(c_total, 0.0, 2.0)


def _modal_feat(obj, modal):
    """Prefer EMA on tracks; fall back to raw feat on fresh detections."""
    ema = getattr(obj, f'{modal}_ema', None)
    if ema is not None:
        return ema
    return getattr(obj, f'{modal}_feat', None)


def _pairwise_cosine_distance(trk_feats, det_feats):
    """Cosine distance matrix; rows/cols with None become all-zeros contribution flags.

    Returns (dist [T,D], valid [T,D] bool) where invalid pairs have dist=0.
    """
    n_trk, n_det = len(trk_feats), len(det_feats)
    dist = np.zeros((n_trk, n_det), dtype=np.float64)
    valid = np.zeros((n_trk, n_det), dtype=bool)
    if n_trk == 0 or n_det == 0:
        return dist, valid

    trk_idx, trk_mat = [], []
    for i, feat in enumerate(trk_feats):
        if feat is None:
            continue
        trk_idx.append(i)
        trk_mat.append(_l2_normalize(np.asarray(feat, dtype=np.float64)))
    det_idx, det_mat = [], []
    for j, feat in enumerate(det_feats):
        if feat is None:
            continue
        det_idx.append(j)
        det_mat.append(_l2_normalize(np.asarray(feat, dtype=np.float64)))
    if not trk_idx or not det_idx:
        return dist, valid

    trk_mat = np.stack(trk_mat, axis=0)
    det_mat = np.stack(det_mat, axis=0)
    sim = np.clip(trk_mat @ det_mat.T, -1.0, 1.0)
    sub = 1.0 - sim
    dist[np.ix_(trk_idx, det_idx)] = sub
    valid[np.ix_(trk_idx, det_idx)] = True
    return dist, valid


def build_rematch_cost(tracks, detections, cfgs):
    """Stage-2 rematch: IoU + RGB/Event cosine, with spatial gating.

    C = (1-λ)·(1-IoU) + λ·(w_rgb·(1-cos_rgb) + w_evt·(1-cos_evt))
    Pairs with IoU < iou_gate (and optionally far centers) are forbidden.
    Missing modal features contribute 0 to C_app (graceful fallback to IoU).
    """
    n_trk, n_det = len(tracks), len(detections)
    if n_trk == 0 or n_det == 0:
        return np.zeros((n_trk, n_det), dtype=np.float64)

    lam = float(cfgs.get('assoc_modal_lambda', 0.4))
    w_rgb = float(cfgs.get('assoc_w_rgb', 0.5))
    w_evt = float(cfgs.get('assoc_w_evt', 0.5))
    iou_gate = float(cfgs.get('assoc_iou_gate', 0.05))
    center_gate = float(cfgs.get('assoc_center_gate', 0.0))  # 0 = disabled
    w_sum = w_rgb + w_evt
    if w_sum <= 1e-8:
        w_rgb, w_evt = 0.5, 0.5
        w_sum = 1.0
    w_rgb, w_evt = w_rgb / w_sum, w_evt / w_sum

    atlbrs = [track.tlbr for track in tracks]
    btlbrs = [det.tlbr for det in detections]
    iou_mat = ious(atlbrs, btlbrs)
    c_iou = 1.0 - iou_mat

    c_app = np.zeros((n_trk, n_det), dtype=np.float64)
    if lam > 0.0:
        c_rgb, v_rgb = _pairwise_cosine_distance(
            [_modal_feat(t, 'rgb') for t in tracks],
            [_modal_feat(d, 'rgb') for d in detections],
        )
        c_evt, v_evt = _pairwise_cosine_distance(
            [_modal_feat(t, 'evt') for t in tracks],
            [_modal_feat(d, 'evt') for d in detections],
        )
        # Renormalize weights per-pair when one modality is missing.
        w_r = np.where(v_rgb, w_rgb, 0.0)
        w_e = np.where(v_evt, w_evt, 0.0)
        w_tot = w_r + w_e
        has = w_tot > 1e-8
        c_app = np.zeros((n_trk, n_det), dtype=np.float64)
        c_app[has] = (w_r[has] * c_rgb[has] + w_e[has] * c_evt[has]) / w_tot[has]

    c_total = (1.0 - lam) * c_iou + lam * c_app

    # Spatial gate: forbid far pairs even if appearance matches.
    forbid = iou_mat < iou_gate
    if center_gate > 0.0:
        trk_xy = np.asarray([t.memo_bank[-1][:2] for t in tracks], dtype=np.float64)
        det_xy = np.asarray([d._xywh[:2] for d in detections], dtype=np.float64)
        trk_wh = np.asarray([t.memo_bank[-1][2:] for t in tracks], dtype=np.float64)
        det_wh = np.asarray([d._xywh[2:] for d in detections], dtype=np.float64)
        trk_diag = np.linalg.norm(trk_wh, axis=1, keepdims=True)
        det_diag = np.linalg.norm(det_wh, axis=1, keepdims=True).T
        scale = 0.5 * (trk_diag + det_diag) + 1e-6
        dist = np.linalg.norm(trk_xy[:, None, :] - det_xy[None, :, :], axis=-1) / scale
        forbid = forbid | (dist > center_gate)

    c_total = np.where(forbid, 1e5, c_total)
    return c_total


def compute_match_cost(tracks, detections, cfgs, stage=1):
    match_mode = cfgs.get('match_mode', 'iou')
    if match_mode == 'hyper_cost':
        return build_cost_matrix(tracks, detections, cfgs, stage=stage)
    if match_mode == 'rematch':
        # Stage1 stays pure IoU; Stage2 uses IoU + RGB/Event rematch.
        if stage == 2:
            return build_rematch_cost(tracks, detections, cfgs)
        from .matching import iou_distance
        return iou_distance(tracks, detections)

    from .matching import iou_distance
    return iou_distance(tracks, detections)
