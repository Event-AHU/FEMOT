# 框架图视觉参考

下面的论文仅用于借鉴框架图的视觉布局、配色和信息组织方式；图中的模型结构、模块名称和数据流仍以本仓库的实际实现为准。

## 1. MamTrack：RGBE 跟踪与 Mamba 历史信息

- 论文：*Exploring Historical Information for RGBE Visual Tracking with Mamba*
- 会议：CVPR 2025
- 官方页面：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Exploring_Historical_Information_for_RGBE_Visual_Tracking_with_Mamba_CVPR_2025_paper.html)
- 参考图：Figure 1、Figure 3(a–c)、Figure 4
- 本地截图：[rgbe-03.png](/tmp/papers/rgbe-03.png)
- 借鉴方式：总体图采用浅蓝色的主容器，RGB/Event 作为左右或上下平行分支；核心模块用圆角虚线子框包围；输入 token 用蓝色/绿色小矩形表示；运算用圆形 `+`、`×`、`C` 符号表示；细节图采用正向/反向 SSM 两条彩色路径，并在右侧单独放置历史信息模块。该布局适合本项目的“两阶段 MOT 总体图”和 SSM 细节图。

## 2. RGB-Event Transformer-Trackers：跨模态 RGB/Event 交互

- 论文：*Cross-Modal Orthogonal High-Rank Augmentation for RGB-Event Transformer-Trackers*
- 会议：ICCV 2023
- 官方页面：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Cross-Modal_Orthogonal_High-Rank_Augmentation_for_RGB-Event_Transformer-Trackers_ICCV_2023_paper.html)
- 参考图：Figure 1(a–c)、Figure 2、Figure 3
- 本地截图：[rgb-03.png](/tmp/papers/rgb-03.png)
- 借鉴方式：采用真实 RGB/Event 小图作为输入端视觉锚点，再连接到少量浅绿色、浅粉色圆角模块；用虚线边框区分 mask、one-stream 和 two-stream 子结构；主箭头保持简洁，避免把每个张量操作都画成大模块。该方式适合 HyLA 检测融合分支，能清楚表现双模态输入、特征交互和融合输出。

## 3. HgVT：超图视觉 Transformer

- 论文：*Hypergraph Vision Transformers: Images are More than Nodes, More than Edges*
- 会议：CVPR 2025
- 官方页面：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Fixelle_Hypergraph_Vision_Transformers_Images_are_More_than_Nodes_More_than_CVPR_2025_paper.html)
- 参考图：Figure 1、Figure 2、Figure 3、Figure 4(a)
- 本地 PDF 与截图：[/tmp/papers/hypervit.pdf](/tmp/papers/hypervit.pdf)、[/tmp/papers/hypervit-04.png](/tmp/papers/hypervit-04.png)
- 借鉴方式：超图细节图采用“输入图像/节点与超边 → 大圆角 HgVT Block 容器 → Pooling/Output Head”的水平三段式；顶点流与超边流使用不同颜色，关系传递用粗彩色箭头，灰色箭头表示辅助流；超边通信池用单独的几何示意图解释。该结构适合表达本项目的 spatial、RGB、Event、motion 四类超边，以及 leave-one-out mean 聚合和 SSM 参数调制。
