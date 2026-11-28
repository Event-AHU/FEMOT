"""
implements base elements of trajectory
"""

import numpy as np 
from basetrack import BaseTrack, TrackState 
from kalman_filters.bytetrack_kalman import ByteKalman
from kalman_filters.botsort_kalman import BotKalman
from kalman_filters.ocsort_kalman import OCSORTKalman
from kalman_filters.sort_kalman import SORTKalman
from kalman_filters.strongsort_kalman import NSAKalman
from kalman_filters.ucmctrack_kalman import UCMCKalman
from kalman_filters.hybridsort_kalman import HybridSORTKalman

MOTION_MODEL_DICT = {
    'sort': SORTKalman, 
    'byte': ByteKalman, 
    'bot': BotKalman, 
    'ocsort': OCSORTKalman, 
    'strongsort': NSAKalman,
    'ucmc': UCMCKalman,  
    'hybridsort': HybridSORTKalman
}

STATE_CONVERT_DICT = {
    'sort': 'xysa', 
    'byte': 'xyah', 
    'bot': 'xywh', 
    'ocsort': 'xysa', 
    'strongsort': 'xyah',
    'ucmc': 'ground', 
    'hybridsort': 'xysca'
}

class Tracklet(BaseTrack):
    def __init__(self, tlwh, score, category, motion='byte'):

        # initial position
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.is_activated = False

        self.score = score
        self.category = category

        # kalman
        self.motion = motion
        self.kalman_filter = MOTION_MODEL_DICT[motion]()
        
        self.convert_func = self.__getattribute__('tlwh_to_' + STATE_CONVERT_DICT[motion])

        # init kalman
        self.kalman_filter.initialize(self.convert_func(self._tlwh))

    def predict(self):
        self.kalman_filter.predict()
        self.time_since_update += 1

    def activate(self, frame_id):
        self.track_id = self.next_id()

        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        self.kalman_filter.update(self.convert_func(new_track.tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.score = new_track.score

        self.kalman_filter.update(self.convert_func(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()
    
    def xyah_to_tlwh(self, ):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def xywh_to_tlwh(self, ):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret
    
    def xysa_to_tlwh(self, ):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[2] = np.sqrt(x[2] * x[3])
        ret[3] = x[2] / ret[2]

        ret[:2] -= ret[2:] / 2
        return ret