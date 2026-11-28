import numpy as np 
from .basetrack import BaseTrack, TrackState 
import torch
from copy import deepcopy

from loguru import logger

from models.MambaTrack import MambaTrack
from models.TrackSSM import TrackSSM

MOTION_PREDICTOR = {
    'MambaTrack': MambaTrack, 
    'TrackSSM': TrackSSM

}

_ALPHA_ATTRS = {
    'dt': 'alpha_dt',
    'b': 'alpha_b',
    'c': 'alpha_c',
}


def _apply_modulation_ablation(model, cfgs):
    """Zero trained Δ/B/C scales after the checkpoint is loaded."""
    zero = cfgs.get('ablate_zero_alphas') or []
    if isinstance(zero, str):
        zero = [zero]
    zero = list(zero)
    unknown = [name for name in zero if name not in _ALPHA_ATTRS]
    if unknown:
        raise ValueError(f'unknown alphas to zero: {unknown}')
    modulators = [
        module for module in model.modules()
        if module.__class__.__name__ == 'SpatialPriorSSMModulator'
    ]
    if zero and not modulators:
        raise RuntimeError('ablate_zero_alphas set but the checkpoint has no SSM modulator')
    with torch.no_grad():
        for module in modulators:
            for name in zero:
                getattr(module, _ALPHA_ATTRS[name]).zero_()
    if zero or int(cfgs.get('ssm_mod_last_k', 0)) != 0:
        summary = ' | '.join(
            'dt={:.4f} b={:.4f} c={:.4f} last_k={}'.format(
                float(module.alpha_dt),
                float(module.alpha_b),
                float(module.alpha_c),
                int(module.last_k),
            )
            for module in modulators
        )
        logger.info('ssm ablation zero={} {}', zero, summary)

def _preprocess_diff(bboxes):
    '''
    calculate the offset difference between frames, bbox format: xywh
    '''
    num_histories = bboxes.shape[0]
    assert num_histories > 1

    ret = np.zeros((num_histories - 1, 4))
    for i in range(num_histories - 1):
        ret[i] = bboxes[i + 1] - bboxes[i]

    return ret



class MambaTracklet(BaseTrack):
    motion_predictor = None  # share the motion predictor for saving the resource 

    @staticmethod
    def set_motion_predictor(motion, cfgs, device):
        logger.info(f'Initalizing motion predictor {motion}')
        model = MOTION_PREDICTOR[motion](cfgs).to(device)
        model.eval()
        logger.info(f'Now tracker is on device {next(model.parameters()).device}')

        # load ckpt
        ckpt_path = cfgs['motion_model_path']
        ckpt = torch.load(ckpt_path, map_location=device)
        logger.info(f'Loaded ckpt {ckpt_path}')
        model.load_state_dict(ckpt['model'])
        _apply_modulation_ablation(model, cfgs)

        MambaTracklet.motion_predictor = model
        MambaTracklet.plugtrack = None
        if cfgs.get('use_plugtrack', False):
            from models.plugtrack.fusion import PlugTrack
            plugin = PlugTrack(
                process_var=float(cfgs.get('plugtrack_process_var', 1e-5)),
                meas_var=float(cfgs.get('plugtrack_meas_var', 5e-4)),
            ).to(device)
            plugin_path = cfgs['plugtrack_ckpt']
            plugin_ckpt = torch.load(plugin_path, map_location=device)
            plugin.load_state_dict(plugin_ckpt['model'])
            plugin.eval()
            MambaTracklet.plugtrack = plugin
            logger.info(f'Loaded PlugTrack {plugin_path}')

    def __init__(
        self,
        cfgs,
        xywh,
        score,
        category,
        img_h,
        img_w,
        device,
        app_feat=None,
        rgb_feat=None,
        evt_feat=None,
    ):
        self.cfgs = cfgs

        # set manner, for mamba track, manner is diff, and for track_ssm, manner is bbox_and_diff
        self.manner = cfgs['manner']

        # initial position
        self._xywh = np.asarray(xywh, dtype=np.float32)  # match with precision of pytorch module
        self.is_activated = False

        self.score = score
        self.category = category

        self.app_feat = None if app_feat is None else np.asarray(app_feat, dtype=np.float32)
        self.app_ema = None
        self.app_ema_momentum = float(cfgs.get('app_ema_momentum', 0.9))
        if self.app_feat is not None:
            self.update_app(self.app_feat)
        self.rgb_feat = None if rgb_feat is None else np.asarray(rgb_feat, dtype=np.float32)
        self.evt_feat = None if evt_feat is None else np.asarray(evt_feat, dtype=np.float32)
        self.rgb_ema = self._normalize_feature(self.rgb_feat)
        self.evt_ema = self._normalize_feature(self.evt_feat)
        # Per-observation modal snapshots aligned with memo_bank (for per-frame hypergraph).
        self.rgb_memo_bank = [
            None if self.rgb_ema is None else self.rgb_ema.copy()
        ]
        self.evt_memo_bank = [
            None if self.evt_ema is None else self.evt_ema.copy()
        ]

        # motion ssm predictor
        self.predicted_last_bbox = None  # the predicted bbox in last step

        # historical observation storage
        self.memo_bank = [self._xywh]
        self.diff_memo_bank = [np.zeros((4, ), dtype=np.float32)]  # store the diff between frames
        # self.diff_memo_bank[i + 1] = self.memo_bank[i + 1] - self.memo_bank[i]

        # store img size for norm
        self.img_size = [img_h, img_w]

        # set device
        self.device = device

    def get_app_ema(self):
        return self.app_ema

    @staticmethod
    def _normalize_feature(feat):
        if feat is None:
            return None
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-6 else feat

    def _append_modal_memo(self):
        self.rgb_memo_bank.append(
            None if self.rgb_ema is None else self.rgb_ema.copy()
        )
        self.evt_memo_bank.append(
            None if self.evt_ema is None else self.evt_ema.copy()
        )

    def _trim_history(self):
        if len(self.memo_bank) > self.cfgs['max_window']:
            self.memo_bank = self.memo_bank[1:]
            self.diff_memo_bank = self.diff_memo_bank[1:]
            self.rgb_memo_bank = self.rgb_memo_bank[1:]
            self.evt_memo_bank = self.evt_memo_bank[1:]

    def update_modal_features(self, rgb_feat, evt_feat):
        momentum = self.app_ema_momentum
        for name, feat in (('rgb', rgb_feat), ('evt', evt_feat)):
            feat = self._normalize_feature(feat)
            if feat is None:
                continue
            setattr(self, f'{name}_feat', feat)
            ema = getattr(self, f'{name}_ema')
            ema = feat if ema is None else momentum * ema + (1.0 - momentum) * feat
            setattr(self, f'{name}_ema', self._normalize_feature(ema))

    def update_app(self, feat):
        if feat is None:
            return
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        if self.app_ema is None:
            self.app_ema = feat.copy()
        else:
            m = self.app_ema_momentum
            self.app_ema = m * self.app_ema + (1.0 - m) * feat
            ema_norm = np.linalg.norm(self.app_ema)
            if ema_norm > 1e-6:
                self.app_ema = self.app_ema / ema_norm

    @property
    def tlwh(self, ):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.predicted_last_bbox is None:
            ret = self._xywh.copy()
        else:
            ret = self.predicted_last_bbox.copy()

        ret[:2] -= 0.5 * ret[2:]
        return ret

    def get_bbox(self, ):
        '''
        get tlwh for visualization
        '''
        if not len(self.memo_bank):
            ret = self._xywh.copy()            
        else:
            ret = self.memo_bank[-1].copy()
        
        ret[:2] -= 0.5 * ret[2:]
        return ret

    @torch.no_grad()
    def predict(self):
        if self.cfgs.get('zero_motion', False):
            # Constant-position baseline: the next bbox is the last observed
            # (or propagated) bbox, i.e. predicted displacement is exactly 0.
            self.predicted_last_bbox = self.memo_bank[-1].copy()
            self.time_since_update += 1
            return

        pred = None
        if len(self.memo_bank) < self.cfgs['enable_time_thresh']:
            pred = self.memo_bank[-1]  # if only has one frame, return the initial position
        else:

            assert len(self.memo_bank) == len(self.diff_memo_bank)  

            # use the motion predictor
            hist_diff = np.array(self.diff_memo_bank[1:])

            # norm
            hist_diff[:, 0] /= self.img_size[1]
            hist_diff[:, 2] /= self.img_size[1]
            hist_diff[:, 1] /= self.img_size[0]
            hist_diff[:, 3] /= self.img_size[0]

            if self.manner == 'diff':

                hist_diff = torch.tensor(hist_diff * self.cfgs['scale_factor'], dtype=torch.float32).unsqueeze(0)  # (1, H, 4)                

                hist_diff = hist_diff.to(self.device)

                out = MambaTracklet.motion_predictor(hist_diff).squeeze()
                out = out.detach().cpu().numpy()

                # recover
                out[0] *= self.img_size[1]
                out[2] *= self.img_size[1]
                out[1] *= self.img_size[0]
                out[3] *= self.img_size[0]

                out /= self.cfgs['scale_factor']

                # add to last observation
                pred = deepcopy(self.memo_bank[-1])
                pred = pred + out

            elif self.manner == 'bbox_and_diff':

                hist = np.array(self.memo_bank[1:])

                # norm
                hist[:, 0] /= self.img_size[1]
                hist[:, 2] /= self.img_size[1]
                hist[:, 1] /= self.img_size[0]
                hist[:, 3] /= self.img_size[0]

                hist_bbox_and_diff = np.concatenate([hist * self.cfgs['scale_factor'], hist_diff * self.cfgs['scale_factor_diff']], axis=1)

                hist_bbox_and_diff = torch.tensor(hist_bbox_and_diff, dtype=torch.float32).unsqueeze(0)  # (1, H, 8)

                hist_bbox_and_diff = hist_bbox_and_diff.to(self.device)

                out = MambaTracklet.motion_predictor(hist_bbox_and_diff).squeeze()

                out = out.detach().cpu().numpy()

                # recover
                out[0] *= self.img_size[1]
                out[2] *= self.img_size[1]
                out[1] *= self.img_size[0]
                out[3] *= self.img_size[0]

                out /= self.cfgs['scale_factor']

                pred = out


        self.time_since_update += 1

        self.predicted_last_bbox = pred
        if (
            pred is not None
            and self.cfgs.get('use_plugtrack', False)
            and not self.cfgs.get('zero_motion', False)
            and len(self.memo_bank) >= self.cfgs['enable_time_thresh']
        ):
            self.predicted_last_bbox = self._fuse_plugtrack(pred)

    def _fuse_plugtrack(self, bbox_mp_pixel):
        """Blend the frozen Mamba box with a Kalman box. Off unless configured."""
        plugin = MambaTracklet.plugtrack
        height, width = self.img_size
        memo = np.asarray(self.memo_bank, dtype=np.float32).copy()
        memo[:, [0, 2]] /= width
        memo[:, [1, 3]] /= height
        boxes = memo[1:]
        diffs = memo[1:] - memo[:-1]
        real = np.concatenate([boxes, diffs], axis=1)
        history_len = int(self.cfgs.get('q', 10))
        if real.shape[0] >= history_len:
            lstm_hist = real[-history_len:]
            kf_hist = lstm_hist
        else:
            pad_count = history_len - real.shape[0]
            pad = np.zeros((pad_count, 8), dtype=np.float32)
            pad[:, :4] = real[0, :4]
            lstm_hist = np.concatenate([pad, real], axis=0)
            kf_hist = real
        lstm_t = torch.tensor(lstm_hist, dtype=torch.float32, device=self.device).unsqueeze(0)
        kf_t = torch.tensor(kf_hist, dtype=torch.float32, device=self.device).unsqueeze(0)
        mp = np.asarray(bbox_mp_pixel, dtype=np.float32).copy()
        mp[[0, 2]] /= width
        mp[[1, 3]] /= height
        mp_t = torch.tensor(mp, dtype=torch.float32, device=self.device).unsqueeze(0)
        fused, _, _, _, _ = plugin(lstm_t, mp_t, kf_history=kf_t)
        fused = fused.squeeze(0).detach().cpu().numpy()
        fused[[0, 2]] *= width
        fused[[1, 3]] *= height
        return fused

    def activate(self, frame_id):
        self.track_id = self.next_id()

        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        self.diff_memo_bank.append(new_track._xywh - self.memo_bank[-1])
        self.memo_bank.append(new_track._xywh)
        self.update_modal_features(new_track.rgb_feat, new_track.evt_feat)
        self._append_modal_memo()
        self._trim_history()

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.app_feat is not None:
            self.update_app(new_track.app_feat)

    def update(self, new_track, frame_id):      

        if new_track is None:
            self.diff_memo_bank.append(self.predicted_last_bbox - self.memo_bank[-1])
            self.memo_bank.append(self.predicted_last_bbox)  # add last predicted bbox
            self._append_modal_memo()
        else:
            self.frame_id = frame_id
            self.diff_memo_bank.append(new_track._xywh - self.memo_bank[-1])
            self.memo_bank.append(new_track._xywh)
            self.score = new_track.score
            if new_track.app_feat is not None:
                self.update_app(new_track.app_feat)
            self.update_modal_features(new_track.rgb_feat, new_track.evt_feat)
            self._append_modal_memo()

        self._trim_history()

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0
