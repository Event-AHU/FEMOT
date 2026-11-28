"""
Naive Base Tracker, mostly borrow from bytetrack
"""

import numpy as np
import torch
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import MambaTracklet
from .matching import *
from .hyper_assoc import compute_match_cost
from dataset.hypergraph_rules import ablate_incidence, build_rule_incidence, relations_to_drop


def _ablate_incidence(incidence, cfgs):
    """Zero whole relation matrices. Same rule as training."""
    drop = relations_to_drop(cfgs)
    incidence = ablate_incidence(incidence, cfgs)
    if drop and not getattr(_ablate_incidence, 'logged', False):
        from loguru import logger
        logger.info('hyper ablation dropping {}', sorted(drop))
        _ablate_incidence.logged = True
    return incidence


class MambaTracker(object):
    '''
    MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model
    '''
    def __init__(self, cfgs, device):
        self.tracked_tracklets = []  
        self.lost_tracklets = []  
        self.removed_tracklets = []  

        self.frame_id = 0
        self.cfgs = cfgs

        self.det_thresh = self.cfgs['filter_thresh']
        self.new_track_thresh = self.cfgs['new_track_thresh']
        self.max_time_lost = self.cfgs['max_time_lost']

        # set motion predictor
        self.device = device
        if not self.cfgs.get('zero_motion', False):
            MambaTracklet.set_motion_predictor(motion=cfgs['model'],
                                               cfgs=self.cfgs,
                                               device=device)


    def update(
        self,
        output_results,
        meta_data=None,
        det_app_feats=None,
        det_rgb_feats=None,
        det_evt_feats=None,
    ):
        """
        output_results: processed detections (scale to original size) tlwh format
        meta_data: List[float], img h and img w
        det_app_feats: optional (N,) appearance features aligned with output_results rows
        """

        self.frame_id += 1
        activated_starcks = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        categories = output_results[:, 5]
        bboxes = output_results[:, :4]

        # convert tlwh to xywh
        bboxes[:, :2] += 0.5 * bboxes[:, 2:]

        remain_inds = scores > self.cfgs['filter_thresh']
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cates_keep = categories[remain_inds]
        if det_app_feats is not None and len(det_app_feats) == len(scores):
            feats_keep = det_app_feats[remain_inds]
            if feats_keep.ndim == 1:
                feat_list = list(feats_keep)
            else:
                feat_list = [feats_keep[i] for i in range(len(dets))]
        else:
            feat_list = [None] * len(dets)
        rgb_list = self._filter_features(det_rgb_feats, remain_inds, len(scores), len(dets))
        evt_list = self._filter_features(det_evt_feats, remain_inds, len(scores), len(dets))


        if len(dets) > 0:
            detections = [MambaTracklet(self.cfgs, xywh, s, categoty, img_h=meta_data[0], img_w=meta_data[1],
                                        device=self.device, app_feat=feat, rgb_feat=rgb_feat, evt_feat=evt_feat)
                          for (xywh, s, categoty, feat, rgb_feat, evt_feat) in
                          zip(dets, scores_keep, cates_keep, feat_list, rgb_list, evt_list)]
        else:
            detections = []

        # predict traj motions
        tracklet_pool = joint_tracklets(self.tracked_tracklets, self.lost_tracklets)
        if self.cfgs.get('use_spatial_hyper', False) or self.cfgs.get(
            'use_spatial_ssm_mod', False
        ):
            self._batch_predict(tracklet_pool)
        else:
            for tracklet in tracklet_pool:
                tracklet.predict()

        match_thresh_stage1 = float(self.cfgs.get('match_thresh_stage1', 0.7))
        match_thresh_stage2 = float(self.cfgs.get('match_thresh_stage2', 0.9))

        # round one: associate detections with activate tracklets
        dists = compute_match_cost(self.tracked_tracklets, detections, self.cfgs, stage=1)
        matches, u_track, u_detection = linear_assignment(dists, thresh=match_thresh_stage1)

        # for matched pairs, update the active tracklets
        for itracked, idet in matches:
            track = self.tracked_tracklets[itracked]
            det = detections[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        # round two: associate unmtached detections with lost tracklets
        
        unmatched_detections = [detections[i] for i in u_detection]
        unmatched_tracklets = [self.tracked_tracklets[i] for i in u_track]
        unmatched_tracklets = joint_tracklets(unmatched_tracklets, self.lost_tracklets)

        dists = compute_match_cost(unmatched_tracklets, unmatched_detections, self.cfgs, stage=2)
        matches, u_track, u_detection = linear_assignment(dists, thresh=match_thresh_stage2)

        for itracked, idet in matches:
            track = unmatched_tracklets[itracked]
            det = unmatched_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)  # for matched lost tracklets, refine it

        for it in u_track:
            # update with last predicted bbox
            track = unmatched_tracklets[it]
            track.update(None, self.frame_id)
            track.mark_lost()
            lost_tracklets.append(track)


        # initial new detections
        for inew in u_detection:
            track = unmatched_detections[inew]
            if track.score < self.new_track_thresh:
                continue    
            track.activate(self.frame_id)
            activated_starcks.append(track)    

        # remove long lost tracklets
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)


        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, activated_starcks)
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, refind_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        self.tracked_tracklets, self.lost_tracklets = remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)
        # get scores of lost tracks
        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets

    @staticmethod
    def _filter_features(features, remain_inds, score_count, det_count):
        if features is None or len(features) != score_count:
            return [None] * det_count
        kept = features[remain_inds]
        return [kept[index] for index in range(len(kept))]

    def _batch_predict(self, tracklets):
        if not tracklets:
            return
        q = int(self.cfgs.get('q', 10))
        n = len(tracklets)
        per_frame = bool(self.cfgs.get('spatial_hyper_per_frame', False))
        histories = np.zeros((n, q, 4), dtype=np.float32)
        centers = np.zeros((n, 2), dtype=np.float32)
        rgb_dim = next(
            (len(track.rgb_ema) for track in tracklets if track.rgb_ema is not None),
            1,
        )
        evt_dim = next(
            (len(track.evt_ema) for track in tracklets if track.evt_ema is not None),
            1,
        )
        rgb = np.zeros((n, rgb_dim), dtype=np.float32)
        evt = np.zeros((n, evt_dim), dtype=np.float32)
        centers_seq = np.zeros((n, q, 2), dtype=np.float32)
        rgb_seq = np.zeros((n, q, rgb_dim), dtype=np.float32)
        evt_seq = np.zeros((n, q, evt_dim), dtype=np.float32)
        eligible = np.zeros(n, dtype=bool)

        for index, track in enumerate(tracklets):
            h, w = track.img_size
            centers[index] = track.memo_bank[-1][:2] / np.asarray([w, h])
            diffs = np.asarray(track.diff_memo_bank[1:], dtype=np.float32).copy()
            if len(diffs):
                diffs[:, [0, 2]] /= w
                diffs[:, [1, 3]] /= h
                diffs *= self.cfgs['scale_factor']
                diffs = diffs[-q:]
                histories[index, -len(diffs):] = diffs
            eligible[index] = len(track.memo_bank) >= self.cfgs['enable_time_thresh']
            if track.rgb_ema is not None and len(track.rgb_ema) == rgb_dim:
                rgb[index] = track.rgb_ema
            if track.evt_ema is not None and len(track.evt_ema) == evt_dim:
                evt[index] = track.evt_ema

            if per_frame:
                memo = track.memo_bank
                m_len = len(memo)
                valid = min(q, max(0, m_len - 1))
                for local_k in range(valid):
                    t = q - valid + local_k
                    end_idx = m_len - valid + local_k
                    centers_seq[index, t] = memo[end_idx][:2] / np.asarray([w, h])
                    rgb_snap = None
                    evt_snap = None
                    if end_idx < len(track.rgb_memo_bank):
                        rgb_snap = track.rgb_memo_bank[end_idx]
                    if end_idx < len(track.evt_memo_bank):
                        evt_snap = track.evt_memo_bank[end_idx]
                    if rgb_snap is not None and len(rgb_snap) == rgb_dim:
                        rgb_seq[index, t] = rgb_snap
                    elif track.rgb_ema is not None and len(track.rgb_ema) == rgb_dim:
                        rgb_seq[index, t] = track.rgb_ema
                    if evt_snap is not None and len(evt_snap) == evt_dim:
                        evt_seq[index, t] = evt_snap
                    elif track.evt_ema is not None and len(track.evt_ema) == evt_dim:
                        evt_seq[index, t] = track.evt_ema
                # Left-pad missing early steps with the first valid snapshot.
                first_valid = q - valid
                if valid > 0 and first_valid > 0:
                    centers_seq[index, :first_valid] = centers_seq[index, first_valid]
                    rgb_seq[index, :first_valid] = rgb_seq[index, first_valid]
                    evt_seq[index, :first_valid] = evt_seq[index, first_valid]

        pairwise = bool(self.cfgs.get('hyper_pairwise', False))
        rule_kwargs = dict(
            max_vertices=n,
            spatial_radius=float(self.cfgs.get('hyper_spatial_radius', 0.10)),
            rgb_threshold=float(self.cfgs.get('hyper_rgb_threshold', 0.80)),
            evt_threshold=float(self.cfgs.get('hyper_evt_threshold', 0.80)),
            topk=int(self.cfgs.get('hyper_topk', 5)),
            pairwise=pairwise,
        )
        edge_slots = max(n * (n - 1) // 2, 1) if pairwise else n
        if per_frame:
            incidence = {
                'spatial': np.zeros((n, q, edge_slots), dtype=np.float32),
                'rgb': np.zeros((n, q, edge_slots), dtype=np.float32),
                'evt': np.zeros((n, q, edge_slots), dtype=np.float32),
                'motion': np.zeros((n, q, 3), dtype=np.float32),
            }
            for t in range(q):
                frame_inc = build_rule_incidence(
                    centers_seq[:, t],
                    rgb_seq[:, t],
                    evt_seq[:, t],
                    histories[:, t],
                    **rule_kwargs,
                )
                for relation, matrix in frame_inc.items():
                    incidence[relation][:, t] = matrix
        else:
            incidence = build_rule_incidence(
                centers,
                rgb,
                evt,
                histories[:, -1],
                **rule_kwargs,
            )
        incidence = _ablate_incidence(incidence, self.cfgs)
        model = MambaTracklet.motion_predictor
        tensor = torch.from_numpy(histories).unsqueeze(0).to(self.device)
        hyper_inputs = {
            relation: torch.from_numpy(matrix).unsqueeze(0).to(self.device)
            for relation, matrix in incidence.items()
        }
        vertex_mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            outputs = model(
                tensor,
                hyper_inputs=hyper_inputs,
                vertex_mask=vertex_mask,
            ).squeeze(0).squeeze(1).cpu().numpy()

        for index, track in enumerate(tracklets):
            if eligible[index]:
                out = outputs[index] / self.cfgs['scale_factor']
                out[[0, 2]] *= track.img_size[1]
                out[[1, 3]] *= track.img_size[0]
                track.predicted_last_bbox = track.memo_bank[-1].copy() + out
            else:
                track.predicted_last_bbox = track.memo_bank[-1].copy()
            track.time_since_update += 1


def joint_tracklets(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_tracklets(tlista, tlistb):
    tracklets = {}
    for t in tlista:
        tracklets[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tracklets.get(tid, 0):
            del tracklets[tid]
    return list(tracklets.values())


def remove_duplicate_tracklets(trackletsa, trackletsb):
    pdist = iou_distance(trackletsa, trackletsb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = trackletsa[p].frame_id - trackletsa[p].start_frame
        timeq = trackletsb[q].frame_id - trackletsb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(trackletsa) if not i in dupa]
    resb = [t for i, t in enumerate(trackletsb) if not i in dupb]
    return resa, resb
