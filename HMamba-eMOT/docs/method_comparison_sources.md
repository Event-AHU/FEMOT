# Method Comparison Figure: Sources and Scope

## (a) YOLOX multimodal fusion

- The original YOLOX implementation takes one image tensor through one backbone and has no native RGB-Event fusion: [YOLOX paper](https://arxiv.org/abs/2107.08430), [official model](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolox.py), [official PAFPN](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_pafpn.py).
- Element-wise addition is shown as one dual-stream baseline, not as a universal property of multimodal YOLOX methods. This project implements that baseline at dark3/dark4/dark5 in [yolox_femot_dual.py](../yolox_exps/custom/yolox_femot_dual.py) and [dual_yolo_pafpn.py](../third_party/ByteTrack/yolox/models/dual_yolo_pafpn.py).
- The proposed path applies HyLA to paired RGB/Event features at dark3/dark4/dark5 before the shared neck and YOLOX head: [yolox_femot_dual_hyla_shared_mlp.py](../yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py), [dual_yolo_pafpn.py](../third_party/ByteTrack/yolox/models/dual_yolo_pafpn.py).

## (b) Hypergraph and SSM integration

- HyperSSM uses hyper-convolution on the motion input used by the state update, and also includes a hyper-convolution output residual: Song et al., [Hypergraph-State Collaborative Reasoning for Multi-Object Tracking](https://openaccess.thecvf.com/content/CVPR2026/html/Song_Hypergraph-State_Collaborative_Reasoning_for_Multi-Object_Tracking_CVPR_2026_paper.html), CVPR 2026, Figure 2 and Section 3.2.
- The proposed comparison summarizes this project as hypergraph-prior modulation inside Mamba's selective parameterization. The figure intentionally omits individual parameter names to keep the comparison conceptual: [spatial_ssm_mod.py](../ssm_tracker/models/spatial_ssm_mod.py), [mamba_hyper.py](../ssm_tracker/models/mamba_hyper.py).
