# encoding: utf-8

from yolox_dsec_dual_hyla_shared_mlp import Exp as HyLAExp


class Exp(HyLAExp):
    """HyLA detector with the original per-image learning rate and a 512 scale cap.

    random_size is multiplied by 32, so (10, 16) is 320--512 instead of 320--640.
    """

    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = "yolox_dsec_dual_hyla_shared_mlp_s512"
        self.random_size = (10, 16)
        self.basic_lr_per_img = 0.001 / 128.0
