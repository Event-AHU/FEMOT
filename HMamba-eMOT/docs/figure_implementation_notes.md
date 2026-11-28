# 三张框架图的实现依据

本次重绘另提供一套稀疏 Visio 风格输出：白底、细蓝线、少量淡色语义高亮、特征条和运算圆点。PNG 只是预览；对应 VSDX 内部为原生可编辑的矩形、椭圆、多边形、连接线和文字，没有把整张 PNG 嵌入页面。

输出文件：`01_overall_two_stage_tracking.vsdx`、`02_ssm_mod_mean_detail.vsdx`、`03_hyla_mlp_detail.vsdx`，以及三页合订版 `hyla_ssm_revised_framework.vsdx`。

核验日期：2026-10-08。本文用于审阅当前仓库的 `ssm_mod_mean + hyla_shared_mlp` 图示；仅记录实现，本次没有修改模型代码。图中的“两个阶段”指检测与跟踪；跟踪内部另有两轮关联。

## 1. 总体图：检测、预测与关联

- 检测：RGB / Event → 两路 CSPDarknet → dark3 / dark4 / dark5 各自执行 Shared-MLP + HyLA 融合 → 共享 PA-FPN → YOLOXHead。依据：[检测器组装](/home/lyt/mymodel/Mamba_Trackers/yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py:35)、[多尺度融合](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/dual_yolo_pafpn.py:168)。
- 预测：上一时刻 Tracked + Lost 轨迹池提供历史框差分、最新中心及 RGB / Event EMA；先构造超图并批量预测，再与当前帧检测关联。当前检测的特征在匹配更新后进入轨迹记忆，因此不直接参与本帧预测前的历史构图。依据：[预测与关联顺序](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/tracker.py:100)、[构图输入](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/tracker.py:209)、[EMA 更新](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/tracklet.py:171)。
- 当前基础配置为 `match_mode: iou`：第一轮 Tracked 对当前检测；第二轮“未匹配 Tracked + Lost”对未匹配检测。两轮代价均为 `1 − IoU`，assignment 代价阈值分别为 0.7 / 0.9，不是 IoU 下限；求解器为 `lap.lapjv`。依据：[两轮关联](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/tracker.py:110)、[代价选择](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/hyper_assoc.py:250)、[求解器](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/matching.py:50)。
- 预测头输出下一步框偏移，反归一化后加到最新框。历史不足阈值的轨迹仍在批量前向中，但不采用模型输出，沿用最后框。依据：[输出恢复](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/tracker.py:301)。

## 2. SSM-mod-mean 细节图

基础配置：[MambaTrack_femot_spatial_ssm_mod_mean.yaml](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/cfgs/MambaTrack_femot_spatial_ssm_mod_mean.yaml:27)。`q=10`、`d_m=512`、`d_state=16`、3 个 BiMambaBlock、超图维度 64，`use_spatial_hyper=false`、`use_spatial_ssm_mod=true`、`edge_agg=mean`、`last_k=0`。

**超图输入与规则。** 基础配置没有启用 `spatial_hyper_per_frame`：仅根据最新历史时刻构造四种固定 incidence 矩阵，不为 q 个时间步分别构图。`frame_hypergraph=true` 表示使用多对象帧级数据集，不等同于逐时间步构图。训练使用该时刻的特征，推理使用轨迹 EMA。依据：[训练构图](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/dataset/dataset.py:286)、[推理构图](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/track_utils/tracker.py:254)。

- Spatial：归一化中心距离在半径内的邻域，每个中心生成一个候选超边；至少两个成员。
- RGB / Event：余弦相似度阈值与 mutual top-k 同时满足，再取连通分量；至少两个成员。
- Motion：对 `sqrt(dx²+dy²)` 的速度大小按 1/3、2/3 分位数分组，不使用运动方向或 `dw/dh`；仅有两个对象时组成一组。依据：[构图规则](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/dataset/hypergraph_rules.py:62)。

**当前 mean 路径的精确聚合式。** 对每种关系分别计算，下式省略 batch 和关系下标。`z_i` 是最新框差分经 Linear → LayerNorm → SiLU 得到的 64 维特征，`H_ie ∈ {0,1}`，`φ_p` / `φ_v` 是 peer / vertex MLP：

\[
d_e=\sum_j H_{je},\qquad
\mu_e=\frac{\sum_jH_{je}z_j}{\max(d_e,1)},\qquad
v_{ie}=\mathbf{1}[d_e-H_{ie}>0].
\]

\[
p_{ie}=\frac{d_e\mu_e-H_{ie}z_i}{\max(d_e-H_{ie},1)},\qquad
c_i=\phi_v\!\left(\frac{\sum_e v_{ie}\phi_p(p_{ie})}{\max(\sum_eH_{ie}v_{ie},1)}\right).
\]

对 incident 超边，`p_ie` 是排除目标自身的均值。当前代码在 `peer_message.sum(dim=2)` 前只乘 `valid_peer`，没有再乘 `H_ie`，因此非 incident 的非空超边也进入分子，而分母只统计 incident 超边。这是当前实现与“仅在 incident 超边内汇聚”的标准写法的差异；图中采用“uniform peer aggregation / vertex normalization”，不将其写成仅对 `e ∋ i` 求和。依据：[mean 分支和顶点汇聚](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/spatial_hypergraph.py:67)。

**关系融合。** `a_i = softmax(MLP([z_i; c_spatial; c_rgb; c_evt; c_motion]))`，`u_i = γ · out_proj(Σ_r a_ir c_ir)`。`γ` 是初值 0.1 的可学习标量；softmax 只用于四类关系融合，mean 超边聚合没有 edge attention。依据：[relation gate](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/spatial_hypergraph.py:121)。

**SSM 参数调制。** 历史框差分单独经 Linear(4→512) 进入 Mamba；`u_i` 不与输入 embedding 直接融合，而是在各层各方向内部，与卷积后的 `x_it` 拼接。静态 `u_i` 沿全部 q 个步骤广播。设 `f_it=[x_it;u_i]`：

\[
\widetilde\Delta'_{it}=\widetilde\Delta_{it}+\alpha_\Delta\mathrm{MLP}_\Delta(f_{it}),\quad
B'_{it}=B_{it}\odot(1+\alpha_B\tanh(\mathrm{MLP}_B(f_{it}))),\quad
C'_{it}=C_{it}\odot(1+\alpha_C\tanh(\mathrm{MLP}_C(f_{it}))).
\]

`Δ` 分支没有 tanh；这里 `Δ̃` 是 scan 前的原始参数，scan 内再执行 `softplus(Δ̃′ + bias)`。各 `α` 可学习且初值 0.1，三个 MLP 的末层零初始化。依据：[先验传递](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/MambaTrack.py:407)、[广播与调制](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/spatial_ssm_mod.py:74)、[scan 的 softplus](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/mamba_hyper.py:74)。

**双向块与输出。** 当前实现为 `ForwardMamba(x) + BackwardMamba(flip(x))`，未将 backward 输出再 flip 回来；随后是 `y + MLP(LayerNorm(y))`。图中使用“Forward scan / Reverse-input scan → sum + residual MLP”，不增加代码中不存在的输出反转。预测头为 AdaptiveAvgPool2d((1,128)) → Linear(128→64) → LeakyReLU → Linear(64→4)。依据：[BiMambaBlock](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/MambaTrack.py:61)、[预测头](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/models/MambaTrack.py:145)。

## 3. HyLA 细节图

- 每个尺度内，RGB / Event 共用同一个残差 `SharedModalMLP`；各尺度有自己的 fusion block。MLP 为 1×1 Conv → BN → SiLU → 1×1 Conv，加输入残差。依据：[SharedModalMLP](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/modal_fusion.py:194)。
- 两路对齐特征拼接成 joint query；RGB-from-Event 和 Event-from-RGB 分别使用对方模态作为 K/V，再通过独立可学习 scale 加回各自模态。依据：[双向 HyLA](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/modal_fusion.py:186)。
- `A_hqk = RMS_heads(QKᵀ/√d)_hqk`，实际两次收缩为 `U_qkd = ReLU(Σ_h A_hqk V_hkd)`、`Y_qhd = Σ_k A_hqk U_qkd`。这是跨 head 的 HyLA target network，不应直接写成普通矩阵乘法 `A ReLU(AV)`；当前配置采用跨 head 的 RMS normalization。依据：[HyLA 收缩](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/modal_fusion.py:116)。
- 增强模态 `R′,E′` 的最终融合为 `R′ + E′ + W_o([g⊙R′; (1−g)⊙E′])`，`g=σ(W_g[R′;E′])`，包含两路直接相加及门控校正。依据：[融合出口](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/modal_fusion.py:255)。
- dark3 / dark4 / dark5 的 attention dim 为 64 / 96 / 128，均为 4 heads；窗口 7 / 7 / 0，其中 0 表示以最大空间边长补齐的单个全局窗口。依据：[配置](/home/lyt/mymodel/Mamba_Trackers/yolox_exps/custom/yolox_femot_dual_hyla_shared_mlp.py:14)、[窗口划分](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/modal_fusion.py:39)。

## 特征来源的两种已有变体

基础 mean 配置使用 `../data/femot_modal_gt`。默认 GT 特征提取及 `gen_det_results.py` 的检测特征导出均从融合前的原始双路 dark3 做 RoI pooling；推理时再形成 RGB / Event EMA，不能把这一路画成 HyLA 融合后的单路特征。依据：[基础数据配置](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/cfgs/MambaTrack_femot_spatial_ssm_mod_mean.yaml:9)、[GT 提取开关](/home/lyt/mymodel/Mamba_Trackers/tools/gen_femot_gt_modal_features.py:145)、[检测特征导出](/home/lyt/mymodel/Mamba_Trackers/tools/gen_det_results.py:317)。

HyLA 重训变体使用 `../data/femot_modal_gt_hyla`，配套提取器取 Shared MLP + 双向 HyLA 之后、gate-sum 融合之前的两路增强 dark3 特征，推理可使用 `gen_det_hyla_enhanced_feats.py` 导出的对应特征。二者的检测器融合链路可以相同，区别在进入超图构图的 RoI 特征来源；实际采用哪一路取决于配置和特征文件。依据：[HyLA 重训配置](/home/lyt/mymodel/Mamba_Trackers/ssm_tracker/cfgs/ablate_retrain_hyla/retrain_hyla_full.yaml:7)、[增强特征接口](/home/lyt/mymodel/Mamba_Trackers/third_party/ByteTrack/yolox/models/dual_yolo_pafpn.py:189)、[检测增强特征导出](/home/lyt/mymodel/Mamba_Trackers/tools/gen_det_hyla_enhanced_feats.py:137)。
