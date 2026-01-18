# TIMEMIXER++: A GENERAL TIME SERIES PATTERN MACHINE FOR UNIVERSAL PREDICTIVE ANALYSIS

Shiyu Wang*, Jiawei Li*1,2, Xiaoming Shi, Zhou Ye, Baichuan Mo3, Wenze Lin4, Shengtong Ju4, Zhixuan Chu†4,5,6, Ming Jin†1

<sup>1</sup>Griffith University <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>3</sup>Massachusetts Institute of Technology <sup>4</sup>Zhejiang University

<sup>5</sup>The State Key Laboratory of Blockchain and Data Security

$^{6}$ Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security

kwuking@gmail.com, jarvis-li@outlook.com, sxm728@hotmail.com  
yezhou199032@gmail.com, baichuan@mit.edu, zhixuanchu@zju.edu.cn  
{linwenze75, shengtongju, mingjinedu}@gmail.com

# ABSTRACT

Time series analysis plays a critical role in numerous applications, supporting tasks such as forecasting, classification, anomaly detection, and imputation. In this work, we present the time series pattern machine (TSPM), a model designed to excel in a broad range of time series tasks through powerful representation and pattern extraction capabilities. Traditional time series models often struggle to capture universal patterns, limiting their effectiveness across diverse tasks. To address this, we define multiple scales in the time domain and various resolutions in the frequency domain, employing various mixing strategies to extract intricate, task-adaptive time series patterns. Specifically, we introduce TIMEMIXER++, a general-purpose TSPM that processes multi-scale time series using (1) multi-resolution time imaging (MRTI), (2) time image decomposition (TID), (3) multi-scale mixing (MCM), and (4) multi-resolution mixing (MRM) to extract comprehensive temporal patterns. MRTI transforms multi-scale time series into multi-resolution time images, capturing patterns across both temporal and frequency domains. TID leverages dual-axis attention to extract seasonal and trend patterns, while MCM hierarchically aggregates these patterns across scales. MRM adaptively integrates all representations across resolutions. TIMEMIXER++ achieves state-of-the-art performance across 8 time series analytical tasks, consistently surpassing both general-purpose and task-specific models. Our work marks a promising step toward the next generation of TSPMs, paving the way for further advancements in time series analysis.

# 1 INTRODUCTION

Time series analysis is crucial for identifying and predicting temporal patterns across various domains, including weather forecasting (Bi et al., 2023), medical symptom classification (Kiyasseh et al., 2021), anomaly detection in spacecraft monitoring (Xu et al., 2022), and imputing missing data in wearable sensors (Wu et al., 2020). These diverse applications highlight the versatility and importance of time series analysis in addressing real-world challenges. A key advancement in this field is the development of time series pattern machines (TSPMs), which aim to create a unified model architecture capable of handling a broad range of time series tasks across domains (Zhou et al., 2023; Wu et al., 2023).

At the core of TSPMs is their ability to recognize and generalize time series patterns inherent in time series data, enabling the model to uncover meaningful temporal structures and adapt to varying time series task scenarios. A line of research (Lai et al., 2018b; Zhao et al., 2017) has utilized recurrent neural networks (RNNs) to capture sequential patterns. However, these methods often struggle to capture long-term dependencies due to limitations like Markovian assumptions and inefficiencies. Temporal convolutional networks (TCNs) (Wu et al., 2023; Wang et al., 2023a; Liu et al., 2022a; Wang et al., 2024a) efficiently capture local patterns but face challenges with long-range dependencies (e.g.,

![](images/2f40005d14c5fae0356c7bea570428d0ec5cfd61f441a9d1e75a4d5455f45b90.jpg)


![](images/f9f8b51fb4550e1f08e8ff7708fe90e755accceafc6b026e02eb0a06363bc0a2.jpg)



Figure 1: Benchmarking model performance across eight tasks (left) and representation analysis in four tasks (right). For each model on the right, the centered kernel alignment (CKA) similarity (Kornblith et al., 2019) is computed between the representations from the first and last layers.


seasonality and trends) because of their fixed receptive fields. While some approaches reshape time series into 2D tensors based on frequency domain information (Wu et al., 2023) or downsample the time domain (Liu et al., 2022a), they fall short in comprehensively capturing long-range patterns. In contrast, transformer-based architectures (Nie et al., 2023; Liu et al., 2024; Zhou et al., 2022b; Wang et al., 2022; Shi et al., 2024) leverage token-wise self-attention to model long-range dependencies by allowing each token to attend to all others, overcoming the limitations of fixed receptive fields. However, unlike language tasks where tokens usually belong to distinct contexts, time series data often involve overlapping contexts at a single time point, such as daily, weekly, and seasonal patterns occurring simultaneously. This overlap makes it difficult to represent time series patterns effectively as tokens, posing challenges for transformer-based models in fully capturing the relevant temporal structures.

The recognition of the above challenges naturally raises a pivotal question:

What capabilities must a model possess, and what challenges must it overcome, to function as a TSPM?

Before addressing the design of TSPMs, we first reconsider how time series are generated from continuous real-world processes sampled at various scales. For example, daily data capture hourly fluctuations, while yearly data reflect long-term trends and seasonal cycles. This multi-scale, multi-periodicity nature presents a significant challenge for model design, as each scale emphasizes different temporal dynamics that must be effectively captured. Figure 1 illustrates this challenge in constructing a general TSPM. Specifically, lower CKA similarity (Kornblith et al., 2019) indicates more diverse representations across layers, which is advantageous for tasks like imputation and anomaly detection that require capturing irregular patterns and handling missing data. In these cases, diverse representations across layers help manage variations across scales and periodicities. Conversely, forecasting and classification tasks benefit from higher CKA similarity, where consistent representations across layers better capture stable trends and periodic patterns. This contrast emphasizes the challenge of designing a universal model flexible enough to adapt to multi-scale and multi-periodicity patterns across various analytical tasks, which may favor either diverse or consistent representations.

To address the aforementioned question and challenges, we propose TIMEMIXER++, a general-purpose TSPM designed to capture general, task-adaptive time series patterns by tackling the complexities of multi-scale and multi-periodicity dynamics. The key idea is to simultaneously capture intricate time series patterns across multiple scales in the time domain and various resolutions in the frequency domain. Specifically, TIMEMIXER++ processes multi-scale time series using (1) multi-resolution time imaging (MRTI), (2) time image decomposition (TID), (3) multi-scale mixing (MCM), and (4) multi-resolution mixing (MRM) to uncover comprehensive patterns. MRTI transforms multi-scale time series into multi-resolution time images, enabling pattern extraction across both temporal and frequency domains. TID applies dual-axis attention to disentangle seasonal and trend patterns in the latent space, while MCM hierarchically aggregates these patterns across

different scales. Finally, MRM adaptively integrates all representations across resolutions. As shown in Figure 1, TIMEMIXER++ achieves state-of-the-art performance across 8 analytical tasks, outperforming both general-purpose and task-specific models. Its adaptability is reflected in its varying CKA similarity scores across different tasks, indicating its ability to capture diverse task-specific patterns more effectively than other models. Our contributions are summarized as follows:

1. We introduce TIMEMIXER++, a general-purpose time series analysis model that processes multi-scale, multi-periodicity data by transforming time series into multi-resolution time images, enabling efficient pattern extraction across both temporal and frequency domains.

2. To capture intricate patterns, we disentangle seasonality and trend from time images using time image decomposition, followed by adaptive aggregation through multi-scale mixing and multi-resolution mixing, enabling patterns integration across scales and periodicities.

3. TIMEMIXER++ sets a new state-of-the-art across 8 time series analytical tasks in different benchmarks, consistently outperforming both general-purpose and task-specific models. This marks a significant step forward in the development of next-generation TSPMs.

# 2 RELATED WORK

Time Series Analysis. A pivotal aspect of time series analysis is the ability to extract diverse patterns from various time series while building powerful representations. This challenge has been explored across various model architectures. Traditional models like ARIMA (Anderson & Kendall, 1976) and STL (Cleveland et al., 1990) are effective for periodic and trend patterns but struggle with non-linear dynamics. Deep learning models, such as those by (Lai et al., 2018b) and (Zhao et al., 2017), capture sequential dependencies but face limitations with long-term dependencies. TCNs (Franceschi et al., 2019) improve local pattern extraction but are limited in capturing long-range dependencies. TimesNet (Wu et al., 2023) enhances long-range pattern extraction by treating time series as 2D signals, while MLP-based methods (Zeng et al., 2023; Ekambaram et al., 2023; Liu et al., 2023; Wang et al., 2023c) offer simplicity and effectiveness. Transformer-based models like PatchTST (Nie et al., 2023) and iTransformer (Liu et al., 2024) leverage self-attention to model long-range dependencies, demonstrating good forecasting performance. Given the strengths and limitations discussed above, there is a growing need for a TSPM capable of effectively extracting diverse patterns, adapting to various time series analytical tasks, and possessing strong generalization capabilities. As illustrated in Figure 1, TIMEmIXER++ meets this requirement by constructing robust representational capabilities, thereby demonstrating its potential for universal time series analysis.

Hierarchical Time Series Modeling. Numerous methodologies have been advanced utilizing specialized deep learning architectures for time series analysis, with an emphasis on the decomposition and integration of temporal patterns. For example, several studies (Wu et al., 2021; Wang et al., 2024b; Zhou et al., 2022b; Luo et al., 2023; Wang et al., 2023b) utilize moving averages to discern seasonal and trend components, which are subsequently modeled using attention mechanisms (Wu et al., 2021; Zhou et al., 2022b; Shi et al., 2025), convolutional networks (Wang et al., 2023a), or hierarchical MLP layers (Wang et al., 2024b; Wang, 2024). These components are individually processed prior to aggregation to yield the final output. Nonetheless, such approaches frequently depend on predefined and rigid operations for the disentanglement of seasonality and trends, thereby constraining their adaptability to complex and dynamic patterns. In contrast, as depicted in Figure 2, we propose a more flexible methodology that disentangles seasonality and trend directly within the latent space via dual-axis attention, thereby enhancing adaptability to a diverse range of time series patterns and task scenarios. Furthermore, by adopting a multi-scale, multi-resolution analytical framework (Mozer, 1991; Harti, 1993), we facilitate hierarchical interaction and integration across different scales and resolutions, substantially enhancing the effectiveness of time series modeling.

# 3 TIMEMIXER++

Building on the multi-scale and multi-periodic characteristics of time series, we introduce TIMEMIXER++, a general-purpose time series pattern machine that processes multi-scale time series using an encoder-only architecture, as shown in Figure 2. The architecture generally comprises three components: (1) input projection, (2) a stack of Mixerblocks, and (3) output projection.

![](images/f59f8e891eff5aec7f976933e03eab0c01716d1260a94454f0f5463724ff417b.jpg)



Figure 2: The framework of TIMEMIXER++. The multi-scale time series is first embedded through an input projection layer, followed by  $L$  stacked MixerBlocks. Each block converts the multi-scale input into multi-resolution time images, disentangles seasonality and trend via dual-axis attention, and mixes these patterns using multi-scale and multi-resolution mixing.


Multi-scale Time Series. We approach time series analysis using a multi-scale framework. Given an input time series  $\mathbf{x}_0\in \mathbb{R}^{T\times C}$ , where  $T$  represents the sequence length and  $C$  the number of variables, we generate a multi-scale representation through downsampling. Specifically, the input time series  $\mathbf{x}_0$  is progressively downsampled across  $M$  scales using convolution operations with a stride of  $2^1$ , producing the multi-scale set  $\mathcal{X}_{init} = \{\mathbf{x}_0,\dots ,\mathbf{x}_M\}$ , where  $\mathbf{x}_m\in \mathbb{R}^{\lfloor \frac{T}{2^m}\rfloor \times C}$ . The downsampling process follows the recursive relationship:

$$
\mathbf {x} _ {m} = \operatorname {C o n v} \left(\mathbf {x} _ {m - 1}, \text {s t r i d e} = 2\right), \quad m \in \{1, \dots , M \}. \tag {1}
$$

# 3.1 STRUCTURE OVERVIEW

Input Projection. Previous studies (2023; 2023) employ a channel-independence strategy to avoid projecting multiple variables into indistinguishable channels (Liu et al., 2024). In contrast, we adopt channel mixing to capture cross-variable interactions, which are crucial for revealing comprehensive patterns in time series data. The input projection has two components: channel mixing and embedding. We first apply self-attention to the variate dimensions at the coarsest scale  $\mathbf{x}_M \in \mathbb{R}^{\lfloor \frac{T}{2M} \rfloor \times C}$ , as it retains the most global context, facilitating the more effective integration of information across variables. This is formulated as follows:

$$
\mathbf {x} _ {M} = \operatorname {C h a n n e l - A t t n} \left(\mathbf {Q} _ {M}, \mathbf {K} _ {M}, \mathbf {V} _ {M}\right), \tag {2}
$$

where Channel-Attn denotes the variate-wise self-attention for channel mixing. The queries, keys, and values  $\mathbf{Q}_M,\mathbf{K}_M,\mathbf{V}_M\in \mathbb{R}^{C\times \lfloor \frac{T}{2^M}\rfloor}$  are derived from linear projections of  $\mathbf{x}_M$ . Then, we embed all multi-scale time series into a deep pattern set  $\mathcal{X}^0$  using an embedding layer, which can be expressed as  $\mathcal{X}^0 = \{\mathbf{x}_0^0,\dots ,\mathbf{x}_M^0\} = \mathrm{Embed}(\mathcal{X}_{init})$ , where  $\mathbf{x}_m^0\in \mathbb{R}^{\lfloor \frac{T}{2^m}\rfloor\times d_{\mathrm{model}}}$  and  $d_{\mathrm{model}}$  represents the dimensionality of the deep patterns.

MixerBlocks. Next, we apply a stack of  $L$  Mixerblocks with the goal to capture intricate patterns across scales in the time domain and resolutions in the frequency domain. Within the MixerBlocks, we convert multi-scale time series into multi-resolution time images, disentangle seasonal and

trend patterns through time image decomposition, and aggregate these patterns across different scales and resolutions. The forward propagation is defined as  $\mathcal{X}^{l + 1} = \mathrm{MixerBlock}(\mathcal{X}^l)$ , where  $\mathcal{X}^l = \{\mathbf{x}_0^l,\dots ,\mathbf{x}_M^l\}$  and  $\mathbf{x}_m^l\in \mathbb{R}^{\lfloor \frac{T}{2^m}\rfloor \times d_{\mathrm{model}}}$ . We will elaborate on this block in the next section.

Output Projection. After  $L \times$  MixerBlocks, we obtain the multi-scale representation set  $\mathcal{X}^L$ . Since different scales capture distinct temporal patterns and tasks vary in demands, as discussed in Section 1, we propose using multiple prediction heads, each specialized for a specific scale, and assembling their outputs. This design is task-adaptive, allowing each head to focus on relevant features at its scale, while the ensemble aggregates complementary information to enhance prediction robustness.

$$
o u t p u t = \operatorname {E n s e m b l e} \left(\left\{\operatorname {H e a d} _ {m} \left(\mathbf {x} _ {m} ^ {L}\right) \right\} _ {m = 0} ^ {M}\right), \tag {3}
$$

where Ensemble  $(\cdot)$  denotes the ensemble method (e.g., averaging or weighted sum), and  $\mathrm{Head}_m(\cdot)$  is the prediction head for the  $m$ -th scale, typically a linear layer.

# 3.2 MIXERBLOCK

We organize a stack of MixerBlocks in a residual way. For the  $(l + 1)$ -th block, the input is the multi-scale representation set  $\mathcal{X}^l$ , and the forward propagation can be formalized as:

$$
\mathcal {X} ^ {l + 1} = \operatorname {L a y e r N o r m} \left(\mathcal {X} ^ {l} + \operatorname {M i x e r B l o c k} \left(\mathcal {X} ^ {l}\right)\right), \tag {4}
$$

where LayerNorm normalizes patterns across scales and can stabilize the training. Time series exhibits complex multi-scale and multi-periodic dynamics. Multi-resolution analysis (Harti, 1993) models time series as a composite of various periodic components in the frequency domain. We introduce multi-resolution time images, converting 1D multi-scale time series into 2D images based on frequency analysis while preserving the original data. This captures intricate patterns across time and frequency domains, enabling efficient use of convolution methods for extracting temporal patterns and enhancing versatility across tasks. Specifically, we processes multi-scale time series using (1) multi-resolution time imaging (MRTI), (2) time image decomposition (TID), (3) multi-scale mixing (MCM), and (4) multi-resolution mixing (MRM) to uncover comprehensive time series patterns.

Multi-Resolution Time Imaging. At the start of each MixerBlock, we convert the input  $\mathcal{X}^l$  into  $(M + 1)\times K$  multi-resolution time images via frequency analysis (Wu et al., 2023). To capture representative periodic patterns, we first identify periods from the coarsest scale  $\mathbf{x}_M^l$ , which enables global interaction. Specifically, we apply the fast fourier transform (FFT) on  $\mathbf{x}_M^l$  and select the top- $K$  frequencies with the highest amplitudes:

$$
\mathbf {A}, \left\{f _ {1}, \dots , f _ {K} \right\}, \left\{p _ {1}, \dots , p _ {K} \right\} = \operatorname {F F T} \left(\mathbf {x} _ {M} ^ {l}\right), \tag {5}
$$

where  $\mathbf{A} = \{A_{f_1},\dots ,A_{f_K}\}$  represents the unnormalized amplitudes,  $\{f_1,\dots ,f_K\}$  are the top- $K$  frequencies, and  $p_k = \left\lceil \frac{T}{f_k}\right\rceil, k\in \{1,\ldots ,K\}$  denotes the corresponding period lengths. Each time series representation  $\mathbf{x}_m^l$  is then reshaped along the temporal dimension as follows:

$$
\begin{array}{l} \operatorname {M R T I} (\mathcal {X} ^ {l}) = \{\mathcal {Z} _ {m} ^ {l} \} _ {m = 0} ^ {M} = \left\{\mathbf {z} _ {m} ^ {(l, k)} \mid m = 0, \dots , M; k = 1, \dots , K \right\} \\ = \left\{\underset {1 D \rightarrow 2 D} {\operatorname {R e s h a p e}} _ {m, k} \left(\operatorname {P a d d i n g} _ {m, k} \left(\mathbf {x} _ {m} ^ {l}\right)\right) \mid m = 0, \dots , M; k = 1, \dots , K \right\}, \tag {6} \\ \end{array}
$$

where  $\text{Padding}_{m,k}(\cdot)$  zero-pads the time series to a length of  $p_k \cdot \lceil \frac{\lfloor\frac{T}{2^m}\rfloor}{p_k} \rceil$ , and  $\text{Reshape}_{m,k}(\cdot)$ $1D \to 2D$  converts it into a  $p_k \times \lceil \frac{\lfloor\frac{T}{2^m}\rfloor}{p_k} \rceil$  image, denoted as  $\mathbf{z}_m^{(l,k)}$ . Here,  $p_k$  represents the number of rows (period length), and the number of columns, denoted by  $f_{\mathrm{m,k}} = \lceil \frac{\lfloor\frac{T}{2^m}\rfloor}{p_k} \rceil$ , represent the corresponding frequency for scale  $m$ .

Time Image Decomposition. Time series patterns are inherently nested, with overlapping scales and periods. For example, weekly sales data reflects both daily shopping habits and broader seasonal trends. Conventional methods (Wu et al., 2021; Wang et al., 2024b) use moving averages across the entire series, often blurring distinct patterns. To address this, we utilize multi-resolution time

images, where each image  $\mathbf{z}_m^{(l,k)}\in \mathbb{R}^{p_k\times f_{\mathrm{m,k}}\times d_{\mathrm{model}}}$  encodes a specific scale and period, enabling finer disentanglement of seasonality and trend. By applying 2D convolution to these images, we capture long-range patterns and enhance temporal dependency extraction. Columns in each image correspond to time series segments within a period, while rows represent consistent time points across periods, facilitating dual-axis attention: column-axis attention (Attentioncol) captures seasonality within periods, and row-axis attention (Attentionrow) extracts trend across periods. Each axis-specific attention focuses on one axis, preserving efficiency by transposing the non-target axis to the batch dimension. For column-axis attention, queries, keys, and values  $\mathbf{Q}_{\mathrm{col}},\mathbf{K}_{\mathrm{col}},\mathbf{V}_{\mathrm{col}}\in \mathbb{R}^{f_{\mathrm{m,k}}\times d_{\mathrm{model}}}$  are computed via 2D convolution, which are shared across all images, and similarly for row-axis attention  $\mathbf{Q}_{\mathrm{row}},\mathbf{K}_{\mathrm{row}},\mathbf{V}_{\mathrm{row}}$ . The seasonal and trend components are then computed as:

$$
\mathbf {s} _ {m} ^ {(l, k)} = \operatorname {A t t e n t i o n} _ {\mathrm {c o l}} \left(\mathbf {Q} _ {\mathrm {c o l}}, \mathbf {K} _ {\mathrm {c o l}}, \mathbf {V} _ {\mathrm {c o l}}\right), \quad \mathbf {t} _ {m} ^ {(l, k)} = \operatorname {A t t e n t i o n} _ {\mathrm {r o w}} \left(\mathbf {Q} _ {\mathrm {r o w}}, \mathbf {K} _ {\mathrm {r o w}}, \mathbf {V} _ {\mathrm {r o w}}\right), \tag {7}
$$

where  $\mathbf{s}_m^{(l,k)}$ ,  $\mathbf{t}_m^{(l,k)} \in \mathbb{R}^{p_k \times f_{\mathrm{m},k} \times d_{\mathrm{model}}}$  represent the seasonal and trend images, respectively. Here, the transposed axis is restored to recover the original image size after the attention.

Multi-scale Mixing. For each period  $p_k$ , we obtain  $M + 1$  seasonal time images and  $M + 1$  trend time images, denoted as  $\{\mathbf{s}_m^{(l,k)}\}_{m=0}^M$  and  $\{\mathbf{t}_m^{(l,k)}\}_{m=0}^M$ , respectively. The 2D structure allows us to model both seasonal and trend patterns using 2D convolutional layers, which are more efficient and effective at capturing long-term dependencies than traditional linear layers (Wang et al., 2024b). For multi-scale seasonal time images, longer patterns can be interpreted as compositions of shorter ones (e.g., a yearly rainfall pattern formed by monthly changes). Therefore, we mix the seasonal patterns from fine-scale to coarse-scale. To facilitate this bottom-up information flow, we apply the 2D convolutional layers at the  $m$ -th scale in a residual manner, formalized as:

$$
\mathrm {f o r} m: 1 \rightarrow M \mathrm {d o :} \quad \mathbf {s} _ {m} ^ {(l, k)} = \mathbf {s} _ {m} ^ {(l, k)} + 2 \mathrm {D - C o n v} (\mathbf {s} _ {m - 1} ^ {(l, k)}), \tag {8}
$$

where 2D-Conv is composed of two 2D convolutional layers with a temporal stride of 2. Unlike seasonal patterns, for multi-scale trend time images, coarser scales naturally highlight the overall trend. Therefore, we adopt a top-down mixing strategy and apply the 2D transposed convolutional layer at the  $m$ -th scale in a residual manner, formalized as:

$$
\text {f o r} M - 1 \rightarrow 0 \text {d o :} \quad \mathbf {t} _ {m} ^ {(l, k)} = \mathbf {t} _ {m} ^ {(l, k)} + 2 \mathrm {D} - \operatorname {T r a n s C o n v} \left(\mathbf {t} _ {m + 1} ^ {(l, k)}\right), \tag {9}
$$

where 2D-TransConv is composed of two 2D transposed convolutional layers with a temporal stride of 2. After mixing, the seasonal and trend patterns are aggregated via summation and reshaped back to a 1D structure, as follows:

$$
\mathbf {z} _ {m} ^ {(l, k)} = \underset {2 D \rightarrow 1 D} {\operatorname {R e s h a p e}} _ {m, k} \left(\mathbf {s} _ {m} ^ {(l, k)} + \mathbf {t} _ {m} ^ {(l, k)}\right), \quad m \in \{0, \dots , M \}, \tag {10}
$$

where  $\operatorname{Reshape}_{m,k}(\cdot)$  convert a  $p_k \times f_{m,k}$  image into a time series of length  $p_k \cdot f_{m,k}$ .

Multi-resolution Mixing. Finally, at each scale, we mix the  $K$  periods adaptively. The amplitudes  $\mathbf{A}$  capture the importance of each period, and we aggregate the patterns  $\{\mathbf{z}_m^{(l,k)}\}_{k=1}^K$  as follows:

$$
\left\{\hat {\mathbf {A}} _ {f _ {k}} \right\} _ {k = 1} ^ {K} = \operatorname {S o f t m a x} \left(\left\{\mathbf {A} _ {f _ {k}} \right\} _ {k = 1} ^ {K}\right), \quad \mathbf {x} _ {m} ^ {l} = \sum_ {k = 1} ^ {K} \hat {\mathbf {A}} _ {f _ {k}} \circ \mathbf {z} _ {m} ^ {(l, k)}, \quad m \in \{0, \dots , M \}, \tag {11}
$$

where Softmax normalizes the weights, and  $\circ$  denotes element-wise multiplication.

# 4 EXPERIMENTS

To verify the effectiveness of the proposed TIMEMIXER++ as a general time series pattern machine, we perform extensive experiments across 8 well-established analytical tasks, including (1) long-term forecasting, (2) univariate and (3) multivariate short-term forecasting, (4) imputation, (5) classification, (6) anomaly detection, as well as (7) few-shot and (8) zero-shot forecasting. Overall, as summarized in Figure 1, TIMEMIXER++ consistently surpasses contemporary state-of-the-art models in a range of critical time series analysis tasks, which is demonstrated by its superior performance across 30 well-known benchmarks and against 27 advanced baselines. The detailed experimental configurations and implementations are in Appendix A.

# 4.1 MAIN RESULTS

# 4.1.1 LONG-TERM FORECASTING

**Setup.** Long-term forecasting is pivotal for strategic planning in areas such as weather prediction, traffic management, and energy utilization. To comprehensively assess our model's effectiveness over extended periods, we perform experiments on 8 widely-used real-world datasets, including the four subsets of the ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2), as well as Weather, Solar-Energy, Electricity, and Traffic, consistent with prior benchmarks set by Zhou et al. (2021b); Liu et al. (2022a).


Table 1: Long-term forecasting results. We average the results across 4 prediction lengths: {96, 192, 336, 720}. The best performance is highlighted in red, and the second-best is underlined. Full results can be found in Appendix H.


<table><tr><td>Models</td><td colspan="2">TimeMixer++ (Ours)</td><td colspan="2">TimeMixer (2024b)</td><td colspan="2">iTransformer (2024)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">TiDE (2023a)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">SCINet (2022a)</td><td colspan="2">FEDformer (2022b)</td><td colspan="2">Stationary (2022c)</td><td colspan="2">Autoformer (2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Electricity</td><td>0.165</td><td>0.253</td><td>0.182</td><td>0.272</td><td>0.178</td><td>0.270</td><td>0.205</td><td>0.290</td><td>0.244</td><td>0.334</td><td>0.251</td><td>0.344</td><td>0.192</td><td>0.295</td><td>0.212</td><td>0.300</td><td>0.268</td><td>0.365</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td></tr><tr><td>ETT (Avg)</td><td>0.349</td><td>0.399</td><td>0.367</td><td>0.388</td><td>0.383</td><td>0.377</td><td>0.381</td><td>0.397</td><td>0.685</td><td>0.578</td><td>0.482</td><td>0.470</td><td>0.391</td><td>0.404</td><td>0.442</td><td>0.444</td><td>0.689</td><td>0.597</td><td>0.408</td><td>0.428</td><td>0.471</td><td>0.464</td><td>0.465</td><td>0.459</td></tr><tr><td>Exchange</td><td>0.357</td><td>0.391</td><td>0.391</td><td>0.453</td><td>0.378</td><td>0.360</td><td>0.403</td><td>0.404</td><td>0.940</td><td>0.707</td><td>0.370</td><td>0.413</td><td>0.416</td><td>0.443</td><td>0.354</td><td>0.414</td><td>0.750</td><td>0.626</td><td>0.519</td><td>0.429</td><td>0.461</td><td>0.454</td><td>0.613</td><td>0.539</td></tr><tr><td>Traffic</td><td>0.416</td><td>0.264</td><td>0.484</td><td>0.297</td><td>0.428</td><td>0.282</td><td>0.481</td><td>0.304</td><td>0.550</td><td>0.304</td><td>0.760</td><td>0.473</td><td>0.620</td><td>0.336</td><td>0.625</td><td>0.383</td><td>0.804</td><td>0.509</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td></tr><tr><td>Weather</td><td>0.226</td><td>0.262</td><td>0.240</td><td>0.271</td><td>0.258</td><td>0.278</td><td>0.259</td><td>0.281</td><td>0.259</td><td>0.315</td><td>0.271</td><td>0.320</td><td>0.259</td><td>0.287</td><td>0.265</td><td>0.317</td><td>0.292</td><td>0.363</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td></tr><tr><td>Solar-Energy</td><td>0.203</td><td>0.238</td><td>0.216</td><td>0.280</td><td>0.233</td><td>0.262</td><td>0.270</td><td>0.307</td><td>0.641</td><td>0.639</td><td>0.347</td><td>0.417</td><td>0.301</td><td>0.319</td><td>0.330</td><td>0.401</td><td>0.282</td><td>0.375</td><td>0.291</td><td>0.381</td><td>0.261</td><td>0.381</td><td>0.885</td><td>0.711</td></tr></table>

Results. Table 1 shows TIMEMIXER++ outperforms other models in long-term forecasting across various datasets. On Electricity, it surpasses iTransformer by  $7.3\%$  in MSE and  $6.3\%$  in MAE. For ETT (Avg), TIMEMIXER++ achieves  $4.9\%$  lower MSE than TimeMixer. On the challenging Solar-Energy dataset (Table 8), it exceeds the second-best model by  $6.0\%$  in MSE and  $9.2\%$  in MAE, demonstrating its robustness in handling complex high-dimensional time series.

# 4.1.2 UNIVARIATE SHORT-TERM FORECASTING

**Setup.** Univariate short-term forecasting is crucial for demand planning and marketing. We evaluate our model using the M4 Competition dataset Makridakis et al. (2018), comprising 100,000 marketing time series with six frequencies from hourly to yearly, enabling comprehensive assessment across varied temporal resolutions.


Table 2: Univariate short-term forecasting results, averaged across all M4 subsets. Full results are available in Appendix H


<table><tr><td>Models</td><td>TimeMixer++ (Ours)</td><td>TimeMixer iTransformer (2024b)</td><td>TiDE (2024)</td><td>TimesNet (2023a)</td><td>N-HiTS (2023)</td><td>N-BEATS (2019)</td><td>PatchTST (2023)</td><td>MICN (2023a)</td><td>FiLM (2022a)</td><td>LightTS (2022a)</td><td>DLinear (2023)</td><td>FED. (2022b)</td><td>Stationary (2022c)</td><td>Auto. (2021)</td><td></td></tr><tr><td>SMAPE</td><td>11.448</td><td>11.723</td><td>12.684</td><td>13.950</td><td>11.829</td><td>11.927</td><td>11.851</td><td>13.152</td><td>19.638</td><td>14.863</td><td>13.525</td><td>13.639</td><td>12.840</td><td>12.780</td><td>12.909</td></tr><tr><td>MASE</td><td>1.487</td><td>1.559</td><td>1.764</td><td>1.940</td><td>1.585</td><td>1.613</td><td>1.559</td><td>1.945</td><td>5.947</td><td>2.207</td><td>2.111</td><td>2.095</td><td>1.701</td><td>1.756</td><td>1.771</td></tr><tr><td>OWA</td><td>0.821</td><td>0.840</td><td>0.929</td><td>1.020</td><td>0.851</td><td>0.861</td><td>0.855</td><td>0.998</td><td>2.279</td><td>1.125</td><td>1.051</td><td>1.051</td><td>0.918</td><td>0.930</td><td>0.939</td></tr></table>

Results. Table 2 demonstrates that TimeMixer++ significantly outperforms state-of-the-art models across all metrics. Compared to iTransformer, it reduces SMAPE by  $9.7\%$  and MASE by  $15.7\%$ , with even larger improvements over TiDE, achieving up to a  $23.3\%$  reduction in MASE. Additionally, TimeMixer++ records the lowest OWA, outperforming TimesNet by  $3.5\%$  and iTransformer by  $11.6\%$ .

# 4.1.3 MULTIVARIATE SHORT-TERM FORECASTING

**Setup.** We further evaluate the short-term forecasting performance in multivariate settings on the PeMS benchmark (Chen et al., 2001), which includes four publicly available high-dimensional traffic network datasets: PEMS03, PEMS04, PEMS07, and PEMS08. These datasets feature a large number of variables, ranging from 170 to 883, offering a challenging testbed for assessing the scalability and effectiveness of our model in predicting complex time series patterns across multiple variables.

Results. The results in Table 3 highlight the superior performance of TIMEMIXER++ across all key metrics in multivariate short-term forecasting. TIMEMIXER++ achieves a  $19.9\%$  reduction in MAE


Table 3: Results of multivariate short-term forecasting, averaged across all PEMS datasets. Full results can be found in Table 18 of Appendix H.


<table><tr><td>Models</td><td>TimeMixer++ (Ours)</td><td>TimeMixer iTransformer (2024b)</td><td>TiDE (2024)</td><td>SCINet (2023a)</td><td>Crossformer (2022a)</td><td>PatchTST (2023)</td><td>TimesNet (2023)</td><td>MICN (2023a)</td><td>DLinear FEDformer (2023)</td><td>Stationary (2022b)</td><td>Autoformer (2021)</td></tr><tr><td>MAE</td><td>15.91</td><td>17.41</td><td>19.87</td><td>21.86</td><td>19.12</td><td>19.03</td><td>23.01</td><td>20.54</td><td>19.34</td><td>23.31</td><td>21.32</td></tr><tr><td>MAPE</td><td>10.08</td><td>10.59</td><td>12.55</td><td>13.80</td><td>12.24</td><td>12.22</td><td>14.95</td><td>12.69</td><td>12.38</td><td>14.68</td><td>14.09</td></tr><tr><td>RMSE</td><td>27.06</td><td>28.01</td><td>31.29</td><td>34.42</td><td>30.12</td><td>30.17</td><td>36.05</td><td>33.25</td><td>30.40</td><td>37.32</td><td>36.78</td></tr></table>

and a  $19.6\%$  reduction in MAPE compared to iTransformer, and an  $8.6\%$  and  $4.8\%$  reduction in MAE and MAPE, respectively, compared to TimeMixer. Notably, PatchTST, a strong baseline, is outperformed by  $\mathrm{TIMEMIXER + + }$  with a  $30.8\%$  improvement in MAE,  $32.5\%$  in MAPE, and  $24.9\%$  in RMSE, highlighting the effectiveness of  $\mathrm{TIMEMIXER + + }$  in handling high-dimensional datasets.

# 4.1.4 IMPUTATION

**Setup.** Accurate imputation of missing values is crucial in time series analysis, affecting predictive models in real-world applications. To evaluate our model's imputation capabilities, we use datasets from electricity and weather domains, selecting ETT (Zhou et al. (2021b)), Electricity (UCI), and Weather (Wetterstation) as benchmarks.


Table 4: Results of imputation task across six datasets. To evaluate our model performance, we randomly mask  $\{12.5\%, 25\%, 37.5\%, 50\% \}$  of the time points in time series of length 1024. The final results are averaged across these 4 different masking ratios.


<table><tr><td>Models</td><td colspan="2">TimeMixer++ (Ours)</td><td colspan="2">TimeMixer iTransformer (2024b)</td><td colspan="2">PatchTST (2024)</td><td colspan="2">Crossformer FEDformer (2023)</td><td colspan="2">TIDE (2023a)</td><td colspan="2">DLinear (2023)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">MICN (2023a)</td><td>Autoformer (2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td></td></tr><tr><td>ETT(Avg)</td><td>0.055</td><td>0.154</td><td>0.097</td><td>0.220</td><td>0.096</td><td>0.205</td><td>0.120</td><td>0.225</td><td>0.150</td><td>0.258</td><td>0.124</td><td>0.230</td><td>0.314</td><td>0.366</td><td>0.115</td><td>0.229</td><td>0.079</td></tr><tr><td>ECL</td><td>0.109</td><td>0.197</td><td>0.142</td><td>0.261</td><td>0.140</td><td>0.223</td><td>0.129</td><td>0.198</td><td>0.125</td><td>0.204</td><td>0.181</td><td>0.314</td><td>0.182</td><td>0.202</td><td>0.080</td><td>0.200</td><td>0.135</td></tr><tr><td>Weather</td><td>0.049</td><td>0.078</td><td>0.091</td><td>0.114</td><td>0.095</td><td>0.102</td><td>0.082</td><td>0.149</td><td>0.150</td><td>0.111</td><td>0.064</td><td>0.139</td><td>0.063</td><td>0.131</td><td>0.071</td><td>0.107</td><td>0.061</td></tr></table>

Results. Table 4 presents TIMEMIXER++'s performance in imputing missing values across six datasets. It consistently outperforms all baselines, achieving the lowest MSE and MAE in the majority of cases. Compared to the second-best model, TimesNet, TIMEMIXER++ reduces MSE by an average of  $25.7\%$  and MAE by  $17.4\%$ . Notably, TIMEMIXER++ excels even when handling imputation tasks with input lengths of up to 1024, demonstrating its robust capability as a TSPM.

# 4.1.5 FEW-SHOT FORECASTING

**Setup.** Transformer-based models excel in various forecasting scenarios, especially with limited data. To evaluate their transferability and pattern recognition, we test across 6 diverse datasets, training each model on only  $10\%$  of available timesteps. This approach assesses adaptability to sparse data and the ability to discern general patterns, which is crucial for real-world predictive analysis where data is often limited.


Table 5: Few-shot learning on  $10\%$  training data. All results are averaged from 4 prediction lengths:  $\{96, 192, 336, 720\}$ .


<table><tr><td>Models</td><td colspan="2">TimeMixer++ (Ours)</td><td colspan="2">TimeMixer iTransformer (2024b)</td><td colspan="2">TiDE (2023a)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">FEDformer Autoformer (2022b) (2021)</td><td colspan="2">Stationary (2022c) (2022)</td><td colspan="2">ETSformer (2022)</td><td colspan="2">LightTS (2022b)</td><td colspan="2">Informer (2021b)</td><td>Reformer (2020)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ETT(Avg)</td><td>0.396</td><td>0.421</td><td>0.453</td><td>0.445</td><td>0.458</td><td>0.497</td><td>0.432</td><td>0.444</td><td>0.470</td><td>0.471</td><td>0.506</td><td>0.484</td><td>0.461</td><td>0.446</td><td>0.586</td><td>0.496</td><td>0.573</td><td>0.532</td><td>0.834</td><td>0.663</td><td>0.627</td><td>0.510</td><td>0.875</td><td>0.687</td><td>1.497</td><td>0.875</td><td>2.408</td><td>1.146</td><td>2.535</td><td>1.191</td></tr><tr><td>Weather</td><td>0.241</td><td>0.271</td><td>0.242</td><td>0.281</td><td>0.291</td><td>0.331</td><td>0.249</td><td>0.291</td><td>0.267</td><td>0.306</td><td>0.241</td><td>0.283</td><td>0.242</td><td>0.279</td><td>0.279</td><td>0.301</td><td>0.284</td><td>0.324</td><td>0.300</td><td>0.342</td><td>0.318</td><td>0.323</td><td>0.318</td><td>0.360</td><td>0.289</td><td>0.322</td><td>0.597</td><td>0.495</td><td>0.546</td><td>0.469</td></tr><tr><td>ECL</td><td>0.168</td><td>0.271</td><td>0.187</td><td>0.277</td><td>0.241</td><td>0.337</td><td>0.196</td><td>0.289</td><td>0.214</td><td>0.308</td><td>0.180</td><td>0.280</td><td>0.180</td><td>0.273</td><td>0.323</td><td>0.392</td><td>0.346</td><td>0.427</td><td>0.431</td><td>0.478</td><td>0.444</td><td>0.480</td><td>0.660</td><td>0.617</td><td>0.441</td><td>0.489</td><td>1.195</td><td>0.891</td><td>0.965</td><td>0.768</td></tr></table>

In few-shot learning, TIMEMIXER++ achieves superior performance across all datasets, reducing MSE by  $13.2\%$  compared to PatchTST. DLinear performs well on some datasets but degrades in zero-shot experiments, suggesting overfitting. TIMEMIXER++ outperforms TimeMixer with  $9.4\%$  lower MSE and  $4.6\%$  lower MAE, attributed to its attention mechanisms enhancing general time series pattern recognition.

# 4.1.6 ZERO-SHOT FORECASTING

**Setup.** We explore zero-shot learning to evaluate models' ability to generalize across different contexts. As shown in Table 6, models trained on dataset  $D_{a}$  are evaluated on unseen dataset  $D_{b}$  without further training. This direct transfer  $(D_{a} \rightarrow D_{b})$  tests models' adaptability and predictive robustness across disparate datasets.


Table 6: Zero-shot learning results. The results are averaged from 4 different prediction lengths:  $\{96,192,336,720\}$ .


<table><tr><td>Methods</td><td colspan="2">TimeMixer++ (Ours)</td><td colspan="2">TimeMixer (2024b)</td><td colspan="2">LLMTime (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">iTransformer (2024)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">Fedformer (2022b)</td><td colspan="2">Autoformer (2021)</td><td>TiDE (2023a)</td><td></td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td></td><td></td></tr><tr><td>ETTh1 → ETTh2</td><td>0.367</td><td>0.391</td><td>0.427</td><td>0.424</td><td>0.992</td><td>0.708</td><td>0.493</td><td>0.488</td><td>0.380</td><td>0.405</td><td>0.421</td><td>0.431</td><td>0.481</td><td>0.474</td><td>0.555</td><td>0.574</td><td>0.712</td><td>0.693</td><td>0.634</td><td>0.651</td><td>0.593</td><td>0.582</td></tr><tr><td>ETTh1 → ETTm2</td><td>0.301</td><td>0.357</td><td>0.361</td><td>0.397</td><td>1.867</td><td>0.869</td><td>0.415</td><td>0.452</td><td>0.314</td><td>0.360</td><td>0.327</td><td>0.361</td><td>0.311</td><td>0.361</td><td>0.613</td><td>0.629</td><td>0.681</td><td>0.588</td><td>0.647</td><td>0.609</td><td>0.563</td><td>0.547</td></tr><tr><td>ETTh2 → ETTh1</td><td>0.511</td><td>0.498</td><td>0.679</td><td>0.577</td><td>1.961</td><td>0.981</td><td>0.703</td><td>0.574</td><td>0.565</td><td>0.513</td><td>0.865</td><td>0.621</td><td>0.552</td><td>0.511</td><td>0.587</td><td>0.518</td><td>0.612</td><td>0.624</td><td>0.599</td><td>0.571</td><td>0.588</td><td>0.556</td></tr><tr><td>ETTm1 → ETTh2</td><td>0.417</td><td>0.422</td><td>0.452</td><td>0.441</td><td>0.992</td><td>0.708</td><td>0.464</td><td>0.475</td><td>0.439</td><td>0.438</td><td>0.457</td><td>0.454</td><td>0.434</td><td>0.438</td><td>0.624</td><td>0.541</td><td>0.533</td><td>0.594</td><td>0.579</td><td>0.568</td><td>0.543</td><td>0.535</td></tr><tr><td>ETTm1 → ETTm2</td><td>0.291</td><td>0.331</td><td>0.329</td><td>0.357</td><td>1.867</td><td>0.869</td><td>0.335</td><td>0.389</td><td>0.296</td><td>0.334</td><td>0.322</td><td>0.354</td><td>0.324</td><td>0.331</td><td>0.595</td><td>0.572</td><td>0.612</td><td>0.611</td><td>0.603</td><td>0.592</td><td>0.534</td><td>0.527</td></tr><tr><td>ETTm2 → ETTm1</td><td>0.427</td><td>0.448</td><td>0.554</td><td>0.478</td><td>1.933</td><td>0.984</td><td>0.649</td><td>0.537</td><td>0.568</td><td>0.492</td><td>0.769</td><td>0.567</td><td>0.559</td><td>0.491</td><td>0.611</td><td>0.593</td><td>0.577</td><td>0.601</td><td>0.594</td><td>0.597</td><td>0.585</td><td>0.571</td></tr></table>

Results. As demonstrated in Table 6, TIMEMIXER++ consistently outperforms other models in our zero-shot learning evaluation across all datasets. Notably, TIMEMIXER++ achieves a significant reduction in MSE by  $13.1\%$  and in MAE by  $5.9\%$  compared to iTransformer. Moreover, it exhibits a reduction in MSE of  $9.6\%$  and in MAE of  $3.8\%$  when compared with PatchTST. These improvements demonstrate the superior generalization capability and robustness of TIMEMIXER++ in handling unseen data patterns, highlighting its potential for real-world applications where adaptability to new scenarios is crucial.

# 4.1.7 CLASSIFICATION AND ANOMALY DETECTION

**Setup.** Classification and anomaly detection test models' ability to capture coarse and fine-grained patterns in time series. We use 10 multivariate datasets from UEA Time Series Classification Archive (2018) for classification. For anomaly detection, we evaluate on SMD (2019), SWaT (2016), PSM (2021), MSL and SMAP (2018).

![](images/19ff1ef725677c5f9b018a758a38ed36428f2d4f41adf4ae473ffa6664ba9816.jpg)



(a) Classification Results


![](images/400361bbad3fee6a4c4fb24818946eade356ce4fa4ba441c9c99d0377400c4c8.jpg)



(b) Anomaly Detection Results



Figure 3: Results of classification and anomaly detection. The results are averaged from several datasets. Higher accuracy and F1 score indicate better performance. *. indicates the Transformer-based models. See Table 19 and 20 in the Appendix H for full results.


Results. Results for both tasks are shown in Figure 4.1.7. For classification, TIMEMIXER++ achieves  $75.9\%$  accuracy, surpassing TimesNet by  $2.3\%$  and outperforming other models. Forecasting models like iTransformer and PatchTST perform poorly, highlighting TIMEMIXER++'s versatility. In anomaly detection, TIMEMIXER++ achieves an F1-score of  $87.47\%$ , exceeding TimesNet by  $2.59\%$ , SCINet by  $3.09\%$ , and Anomaly Transformer by  $6.62\%$ . These results emphasize TIMEMIXER++'s strong pattern learning capability, attributed to its multi-scale and multi-resolution architecture.

# 4.2 MODEL ANALYSIS

Ablation Study. To verify the effectiveness of each component of TIMEMIXER++, we conducted an ablation study by removing individual components (w/o). The results are in Table 7. TIMEMIXER++ with all components—channel mixing, time image decompose, multi-scale mixing, and multi-resolution mixing—achieves the best performance. On datasets like ECL, Traffic, and Solar, channel-mixing improves performance by  $5.36\%$ . Time image decomposition yields an  $8.81\%$  improvement, especially on seasonal datasets like ECL and Traffic. Multi-scale mixing provides a  $6.25\%$  improvement, particularly for less predictable datasets like ETT. Multi-resolution mixing adds a  $5.10\%$  improvement, highlighting the importance of a multi-resolution hybrid ensemble. We provide more ablation studies in Appednix C.


Table 7: MSE for long-term forecasting across 8 benchmarks, evaluated with different model components. We provide more analysis on other tasks in Tables 11 and 12.


<table><tr><td></td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>ECL</td><td>Traffic</td><td>Weather</td><td>Solar</td><td>Average</td><td>Promotion</td></tr><tr><td>TIMEMIXER++</td><td>0.419</td><td>0.339</td><td>0.369</td><td>0.269</td><td>0.165</td><td>0.416</td><td>0.226</td><td>0.203</td><td>0.300</td><td>-</td></tr><tr><td>w/o channel mixing</td><td>0.424</td><td>0.346</td><td>0.374</td><td>0.271</td><td>0.197</td><td>0.442</td><td>0.233</td><td>0.245</td><td>0.317</td><td>5.36%</td></tr><tr><td>w/o time image decomposition</td><td>0.441</td><td>0.358</td><td>0.409</td><td>0.291</td><td>0.198</td><td>0.445</td><td>0.251</td><td>0.241</td><td>0.329</td><td>8.81%</td></tr><tr><td>w/o multi-scale mixing</td><td>0.447</td><td>0.361</td><td>0.391</td><td>0.284</td><td>0.172</td><td>0.427</td><td>0.239</td><td>0.234</td><td>0.320</td><td>6.25%</td></tr><tr><td>w/o multi-resolution mixing</td><td>0.431</td><td>0.350</td><td>0.374</td><td>0.280</td><td>0.181</td><td>0.432</td><td>0.241</td><td>0.233</td><td>0.316</td><td>5.10%</td></tr></table>

Representation Analysis. Our analysis, depicted in Figure 4, present the original, seasonality, and trend images across two scales and three resolutions (periods: 12, 8, 6; frequencies: 16, 24, 32). TIMEMIXER++ demonstrates efficacy in the separation of distinct seasonality and trends, precisely capturing multi-periodicities and time-varying trends. Notably, the periodic characteristics vary across different scales and resolutions. This hierarchical structure permits the simultaneous capture of these features, underscoring the temporal variability of the data.

ing the robust representational capabilities of TIMEMIXER++ as a pattern machine.

![](images/b1532b1f217186d69b2db91e3112b1666e4d748ad285bb7aa58e95308da5ec06.jpg)



Figure 4: Visualization of representation on Time Image under Traffic dataset. More showcases in Figure 10, 11, 13.


Furthermore, as on the right side of Figure 1, from the perspective of representation learning, TIMEMIXER++ shows superior performance in prediction and anomaly detection with higher CKA similarity 2019, compared to imputation and classification tasks. Lower CKA similarity indicates more distinctive layer-wise representations, suggesting a hierarchical structure. Figure 1 demonstrates that TIMEMIXER++ captures distinct low-level representations for forecasting and anomaly detection, and hierarchical ones for imputation and classification. This highlights TIMEMIXER++'s potential as a general time series pattern machine, capable of identifying diverse patterns across tasks and domains, essential for universal predictive analysis in time series. See Appendix D for more details.

# 5 CONCLUSION

In this paper, we present TIMEMIXER++, a novel framework designed as a universal time series pattern machine for predictive analysis. By leveraging multi-resolution imaging, we construct time images at various resolutions, enabling enhanced representation of temporal dynamics. The use of dual-axis attention allows for effective decomposition of these time images, disentangling seasonal and trend components within deep representations. With multi-scale and multi-resolution mixing techniques, TIMEMIXER++ seamlessly fuses and extracts information across different hierarchical levels, demonstrating strong representational capabilities. Through extensive experiments and comprehensive evaluations, TIMEMIXER++ consistently outperforms existing general-purpose and task-specific time series models, establishing itself as a state-of-the-art solution with significant potential for broad applications in time series analysis. Limitations and directions for future research are discussed in Appendix K.

# ACKNOWLEDGEMENT



M. Jin was supported in part by the NVIDIA Academic Grant Program and CSIRO - National Science Foundation (US) AI Research Collaboration Program.



# REFERENCES



Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. Practical approach to asynchronous multivariate time series anomaly detection and localization. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 2485-2494, 2021.





O. Anderson and M. Kendall. Time-series. 2nd edn. J. R. Stat. Soc. (Series D), 1976.





Anthony Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large, Aaron Bostrom, Paul Southam, and Eamonn Keogh. The uea multivariate time series classification archive, 2018. arXiv preprint arXiv:1811.00075, 2018.





Donald J. Berndt and James Clifford. Using dynamic time warping to find patterns in time series. In KDD Workshop, 1994.





Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3d neural networks. Nature, 619(7970):533-538, 2023.





Cristian Challu, Kin G Olivares, Boris N Oreshkin, Federico Garza, Max Mergenthaler, and Artur Dubrawski. N-hits: Neural hierarchical interpolation for time series forecasting. AAAI, 2023.





Chao Chen, Karl F. Petty, Alexander Skabardonis, Pravin Pratap Varaiya, and Zhanfeng Jia. Freeway performance measurement system: Mining loop detector data. Transportation Research Record, 2001.





Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. KDD, 2016.





Robert B Cleveland, William S Cleveland, Jean E McRae, and Irma Terpenning. STL: A seasonal-trend decomposition. Journal of Official Statistics, 1990.





Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K Mathur, Rajat Sen, and Rose Yu. Long-term forecasting with tIDE: Time-series dense encoder. Transactions on Machine Learning Research, 2023a. ISSN 2835-8856.





Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K Mathur, Rajat Sen, and Rose Yu. Long-term forecasting with tide: Time-series dense encoder. Transactions on Machine Learning Research, 2023b.





Angus Dempster, Francois Petitjean, and Geoffrey I. Webb. Rocket: exceptionally fast and accurate time series classification using random convolutional kernels. Data Min. Knowl. Discov., 2020.





Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD, pp. 459-469. Association for Computing Machinery, 2023. doi: 10.1145/3580305.3599533.





Jean-Yves Franceschi, Aymeric Dieuleveut, and Martin Jaggi. Unsupervised scalable representation learning for multivariate time series. Advances in neural information processing systems, 32, 2019.





Georg Goerg. Forecastable component analysis. ICML, 2013.





Nate Gruver, Marc Anton Finzi, Shikai Qiu, and Andrew Gordon Wilson. Large language models are zero-shot time series forecasters. Advances in Neural Information Processing Systems, 2023.





Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022a.





Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In ICLR, 2022b.





Ami Harti. Discrete multi-resolution analysis and generalized wavelets. Applied numerical mathematics, 12(1-3):153-192, 1993.





S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 1997.





Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer: A transformer architecture for optical flow. In European conference on computer vision, pp. 668-685. Springer, 2022.





Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 387-395, 2018.





Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *ICLR*, 2015.





Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. *ICLR*, 2020.





Dani Kiyasseh, Tingting Zhu, and David A Clifton. Clocs: Contrastive learning of cardiac signals across space, time, and patients. In International Conference on Machine Learning, pp. 5606-5615. PMLR, 2021.





Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pp. 3519-3529. PMLR, 2019.





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In SIGIR, 2018a.





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 95-104, 2018b.





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. SIGIR, 2018c.





Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In NeurIPS, 2019.





Zhe Li, Shiyi Qi, Yiduo Li, and Zenglin Xu. Revisiting long-term time series forecasting: An investigation on linear mapping. arXiv preprint arXiv:2305.10721, 2023a.





Zhe Li, Zhongwen Rao, Lujia Pan, and Zenglin Xu. Mts-mixers: Multivariate time series forecasting via factorized temporal and channel mixing. arXiv preprint arXiv:2302.04501, 2023b.





Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. SCINet: time series modeling and forecasting with sample convolution and interaction. NeurIPS, 2022a.





Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations, 2022b.





Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. In Advances in Neural Information Processing Systems, 2022c.





Yong Liu, Chenyu Li, Jianmin Wang, and Mingsheng Long. Koopa: Learning non-stationary time series dynamics with koopman predictors. In Advances in Neural Information Processing Systems, 2023.





Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. *Itransformer: Inverted transformers are effective for time series forecasting.* In *The Twelfth International Conference on Learning Representations*, 2024.





Yuxiao Luo, Ziyu Lyu, and Xingyu Huang. Tfdnet: Time-frequency enhanced decomposed network for long-term time series forecasting. arXiv preprint arXiv:2308.13386, 2023.





Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The m4 competition: Results, findings, conclusion and way forward. International Journal of forecasting, 34(4):802-808, 2018.





Aditya P Mathur and Nils Ole Tippenhauer. Swat: A water treatment testbed for research and training on ics security. In 2016 international workshop on cyber-physical systems for smart water networks (CySWater), pp. 31-36. IEEE, 2016.





Michael C. Mozer. Induction of multiscale temporal structure. NeurIPS, 1991.





Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. *ICLR*, 2023.





Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR*, 2019.





Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.





Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Scaling to billion parameters for time series foundation models with mixture of experts. In NeurIPS Workshop on Time Series in the Age of Large Models, 2024.





Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Time-moe: Billion-scale time series foundation models with mixture of experts. In The Thirteenth International Conference on Learning Representations, 2025.





Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2828-2837, 2019.





UCI. Electricity. URL https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.





Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. MICN: Multiscale local and global context modeling for long-term series forecasting. ICLR, 2023a.





Shiyu Wang. Neuralreconciler for hierarchical time series forecasting. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pp. 731-739, 2024.





Shiyu Wang, Fan Zhou, Yinbo Sun, Lintao Ma, James Zhang, and Yangfei Zheng. End-to-end modeling of hierarchical time series using autoregressive transformer and conditional normalizing flow-based reconciliation. In 2022 IEEE International Conference on Data Mining Workshops (ICDMW), pp. 1087-1094. IEEE, 2022.





Shiyu Wang, Yinbo Sun, Xiaoming Shi, Zhu Shiyi, Lin-Tao Ma, James Zhang, YangFei Zheng, and Liu Jian. Full scaling automation for sustainable development of green data centers. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pp. 6264-6271, 2023b.





Shiyu Wang, Yinbo Sun, Yan Wang, Fan Zhou, Lin-Tao Ma, James Zhang, and YangFei Zheng. Flow-based end-to-end model for hierarchical time series forecasting via trainable attentive-reconciliation. In International Conference on Database Systems for Advanced Applications, pp. 167-176. Springer, 2023c.





Shiyu Wang, Zhixuan Chu, Yinbo Sun, Yu Liu, Yuliang Guo, Yang Chen, Huiyang Jian, Lintao Ma, Xingyu Lu, and Jun Zhou. Multiscale representation enhanced temporal flow fusion model for long-term workload forecasting. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 4948-4956, 2024a.





Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y Zhang, and JUN ZHOU. Timemixer: Decomposable multiscale mixing for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024b.





Wetterstation. Weather. URL https://www.bgc-jena.mpg.de/wetter/.





Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshit Kumar, and Steven C. H. Hoi. Etsformer: Exponential smoothing transformers for time-series forecasting. arXiv preprint arXiv:1406.1078, 2022.





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with Auto-Correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, 2021.





Haixu Wu, Jialong Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Flowformer: Linearizing transformers with conservation flows. In ICML, 2022.





Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2d-variation modeling for general time series analysis. ICLR, 2023.





Xian Wu, Stephen Mattingly, Shayan Mirjafari, Chao Huang, and Nitesh V Chawla. Personalized imputation on wearable-sensory time series via knowledge transfer. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pp. 1625-1634, 2020.





Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy. In International Conference on Learning Representations, 2022.





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? AAAI, 2023.





Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022a.





Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022b.





Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. ICLR, 2023.





Zheng Zhao, Weihai Chen, Xingming Wu, Peter CY Chen, and Jingmeng Liu. Lstm network: a deep learning approach for short-term traffic forecast. IET intelligent transport systems, 11(2):68-75, 2017.





Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. AAAI, 2021a.





Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11106-11115, 2021b.





Tian Zhou, Ziqing Ma, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, Rong Jin, et al. Film: Frequency improved legendre memory model for long-term time series forecasting. NeurIPS, 2022a.





Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. ICML, 2022b.





Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al. One fits all: Power general time series analysis by pretrained Im. Advances in neural information processing systems, 36:43322-43355, 2023.



# A IMPLEMENTATION DETAILS

Datasets Details. We evaluate the performance of different models for long-term forecasting on 8 well-established datasets, including Weather, Traffic, Electricity, Exchange, Solar-Energy, and ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2). Furthermore, we adopt PeMS and M4 datasets for short-term forecasting. We detail the descriptions of the dataset in Table 8. To comprehensively evaluate the model's performance in time series analysis tasks, we further introduced datasets for classification and anomaly detection. The classification task is designed to test the model's ability to capture coarse-grained patterns in time series data, while anomaly detection focuses on the recognition of fine-grained patterns. Specifically, we used 10 multivariate datasets from the UEA Time Series Classification Archive (2018) for the evaluation of classification tasks. For anomaly detection, we selected datasets such as SMD (2019), SWaT (2016), PSM (2021), MSL, and SMAP (2018). We detail the descriptions of the datasets for classification and anomaly detection in Table 9 and Table 10.


Table 8: Dataset detailed descriptions. The dataset size is organized in (Train, Validation, Test).


<table><tr><td>Tasks</td><td>Dataset</td><td>Dim</td><td>Series Length</td><td>Dataset Size</td><td>Frequency</td><td>Forecastability*</td><td>Information</td></tr><tr><td rowspan="9">Long-term Forecasting</td><td>ETTm1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>0.46</td><td>Temperature</td></tr><tr><td>ETTm2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>0.55</td><td>Temperature</td></tr><tr><td>ETTh1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>15 min</td><td>0.38</td><td>Temperature</td></tr><tr><td>ETTh2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>15 min</td><td>0.45</td><td>Temperature</td></tr><tr><td>Electricity</td><td>321</td><td>{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Hourly</td><td>0.77</td><td>Electricity</td></tr><tr><td>Traffic</td><td>862</td><td>{96, 192, 336, 720}</td><td>(12185, 1757, 3509)</td><td>Hourly</td><td>0.68</td><td>Transportation</td></tr><tr><td>Exchange</td><td>8</td><td>{96, 192, 336, 720}</td><td>(5120, 665, 1422)</td><td>Daily</td><td>0.41</td><td>Weather</td></tr><tr><td>Weather</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>10 min</td><td>0.75</td><td>Weather</td></tr><tr><td>Solar-Energy</td><td>137</td><td>{96, 192, 336, 720}</td><td>(36601, 5161, 10417)</td><td>10min</td><td>0.33</td><td>Electricity</td></tr><tr><td rowspan="10">Short-term Forecasting</td><td>PEMS03</td><td>358</td><td>12</td><td>(15617,5135,5135)</td><td>5min</td><td>0.65</td><td>Transportation</td></tr><tr><td>PEMS04</td><td>307</td><td>12</td><td>(10172,3375,3375)</td><td>5min</td><td>0.45</td><td>Transportation</td></tr><tr><td>PEMS07</td><td>883</td><td>12</td><td>(16911,5622,5622)</td><td>5min</td><td>0.58</td><td>Transportation</td></tr><tr><td>PEMS08</td><td>170</td><td>12</td><td>(10690,3548,265)</td><td>5min</td><td>0.52</td><td>Transportation</td></tr><tr><td>M4-Yearly</td><td>1</td><td>6</td><td>(23000, 0, 23000)</td><td>Yearly</td><td>0.43</td><td>Demographic</td></tr><tr><td>M4-Quarterly</td><td>1</td><td>8</td><td>(24000, 0, 24000)</td><td>Quarterly</td><td>0.47</td><td>Finance</td></tr><tr><td>M4-Monthly</td><td>1</td><td>18</td><td>(48000, 0, 48000)</td><td>Monthly</td><td>0.44</td><td>Industry</td></tr><tr><td>M4-Weakly</td><td>1</td><td>13</td><td>(359, 0, 359)</td><td>Weakly</td><td>0.43</td><td>Macro</td></tr><tr><td>M4-Daily</td><td>1</td><td>14</td><td>(4227, 0, 4227)</td><td>Daily</td><td>0.44</td><td>Micro</td></tr><tr><td>M4-Hourly</td><td>1</td><td>48</td><td>(414, 0, 414)</td><td>Hourly</td><td>0.46</td><td>Other</td></tr></table>


* The forecastability is calculated by one minus the entropy of Fourier decomposition of time series (Goerg, 2013). A larger value indicates better predictability.



Table 9: Datasets and mapping details of UEA dataset (Bagnall et al., 2018).


<table><tr><td>Dataset</td><td>Sample Numbers(train set,test set)</td><td>Variable Number</td><td>Series Length</td></tr><tr><td>EthanolConcentration</td><td>(261, 263)</td><td>3</td><td>1751</td></tr><tr><td>FaceDetection</td><td>(5890, 3524)</td><td>144</td><td>62</td></tr><tr><td>Handwriting</td><td>(150, 850)</td><td>3</td><td>152</td></tr><tr><td>Heartbeat</td><td>(204, 205)</td><td>61</td><td>405</td></tr><tr><td>JapaneseVowels</td><td>(270, 370)</td><td>12</td><td>29</td></tr><tr><td>PEMSSF</td><td>(267, 173)</td><td>963</td><td>144</td></tr><tr><td>SelfRegulationSCP1</td><td>(268, 293)</td><td>6</td><td>896</td></tr><tr><td>SelfRegulationSCP2</td><td>(200, 180)</td><td>7</td><td>1152</td></tr><tr><td>SpokenArabicDigits</td><td>(6599, 2199)</td><td>13</td><td>93</td></tr><tr><td>UWaveGestureLibrary</td><td>(120, 320)</td><td>3</td><td>315</td></tr></table>


Table 10: Datasets and mapping details of anomaly detection dataset.


<table><tr><td>Dataset</td><td>Dataset sizes(train set, val set, test set)</td><td>Variable Number</td><td>Sliding Window Length</td></tr><tr><td>SMD</td><td>(566724, 141681, 708420)</td><td>38</td><td>100</td></tr><tr><td>MSL</td><td>(44653, 11664, 73729)</td><td>55</td><td>100</td></tr><tr><td>SMAP</td><td>(108146, 27037, 427617)</td><td>25</td><td>100</td></tr><tr><td>SWaT</td><td>(396000, 99000, 449919)</td><td>51</td><td>100</td></tr><tr><td>PSM</td><td>(105984, 26497, 87841)</td><td>25</td><td>100</td></tr></table>

Baseline Details. To assess the effectiveness of our method across various tasks, we select 27 advanced baseline models spanning a wide range of architectures. Specifically, we utilize CNN-based models: MICN (2023a), SCINet (2022a), and TimesNet (2023); MLP-based models: TimeMixer (2024b), LightTS (2022a), and DLinear (2023); RMLP&RLinear (2023a) and Transformer-based models: iTransformer (2024), PatchTST (2023), Crossformer (2023), FEDformer (2022b), Stationary (2022c), Autoformer (2021), and Informer (2021b). These models have demonstrated superior capabilities in temporal modeling and provide a robust framework for comparative analysis. For specific tasks, TiDE (2023b), FiLM (2022a), N-HiTS (2023), and N-BEATS (2019) address long- or short-term forecasting; Anomaly Transformer (2022) and MTS-Mixers (2023b) target anomaly detection; while Rocket (2023a), LSTNet (2018c), LSSL (2022a), and Flowformer (2022) are utilized for classification. Few/zero-shot forecasting tasks employ ETS-former (2022), Reformer (2020), and LLMTime (2023).

Metric Details. Regarding metrics, we utilize the mean square error (MSE) and mean absolute error (MAE) for long-term forecasting. In the case of short-term forecasting, we follow the metrics of SCINet (Liu et al., 2022a) on the PeMS datasets, including mean absolute error (MAE), mean absolute percentage error (MAPE), root mean squared error (RMSE). As for the M4 datasets, we follow the methodology of N-BEATS (Oreshkin et al., 2019) and implement the symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA) as metrics. It is worth noting that OWA is a specific metric utilized in the M4 competition. The calculations of these metrics are:

$$
\mathrm {R M S E} = \left(\sum_ {i = 1} ^ {F} \left(\mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i}\right) ^ {2}\right) ^ {\frac {1}{2}}, \quad \mathrm {M A E} = \sum_ {i = 1} ^ {F} \left| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} \right|,
$$

$$
\mathrm {S M A P E} = \frac {2 0 0}{F} \sum_ {i = 1} ^ {F} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} | + | \widehat {\mathbf {X}} _ {i} |}, \qquad \qquad \mathrm {M A P E} = \frac {1 0 0}{F} \sum_ {i = 1} ^ {F} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} |},
$$

$$
\mathrm {M A S E} = \frac {1}{F} \sum_ {i = 1} ^ {F} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{\frac {1}{F - s} \sum_ {j = s + 1} ^ {F} | \mathbf {X} _ {j} - \mathbf {X} _ {j - s} |}, \quad \mathrm {O W A} = \frac {1}{2} \left[ \frac {\mathrm {S M A P E}}{\mathrm {S M A P E} _ {\mathrm {N a i v e 2}}} + \frac {\mathrm {M A S E}}{\mathrm {M A S E} _ {\mathrm {N a i v e 2}}} \right],
$$

where  $s$  is the periodicity of the data.  $\mathbf{X},\widehat{\mathbf{X}}\in \mathbb{R}^{F\times C}$  are the ground truth and prediction results of the future with  $F$  time pints and  $C$  dimensions.  $\mathbf{X}_i$  means the  $i$ -th future time point.

Experiment Details. All experiments were run three times, implemented in Pytorch (Paszke et al., 2019), and conducted on multi NVIDIA A100 80GB GPUs. We set the initial learning rate as a range from  $10^{-3}$  to  $10^{-1}$  and used the ADAM optimizer (Kingma & Ba, 2015) with L2 loss for model optimization. And the batch size was set to be 512. We set the number of resolutions  $K$  to range from 1 to 5. Moreover, we set the number of MixerBlocks  $L$  to range from 1 to 3. We choose the number of scales  $M$  according to the time series length to balance performance and efficiency. To handle longer series in long-term forecasting, we usually set  $M$  to 3. As for short-term forecasting with limited series length, we usually set  $M$  to 1. We pretrained the model with learning rate decay after linear warm-up. For baselines under the same experimental settings as our main study, we directly report the results from TimesNet (Wu et al., 2023). In scenarios where experimental settings differed or tasks were not previously implemented, we reproduced the baseline results referring to the benchmark framework from the Time-Series Library<sup>2</sup>. The source code and pretrained model will be provided in GitHub (https://github.com/kwuking/TimeMixer).

# B DETAILS OF MODEL DESIGN

In this section, we present a comprehensive exposition of our model design, encompassing five key components: channel mixing and embedding (input projection), multi-resolution time imaging, time image decomposition, multi-scale mixing, and multi-resolution mixing. To enhance comprehension, we provide visual illustrations that afford an intuitive understanding of our structural design.

Channel Mixing and Embedding. We employ a channel mixing approach to effectively capture inter-variable dependencies crucial for uncovering rich temporal patterns. Our method first applies variate-wise self-attention, as formulated in Equation 2, at the coarsest temporal scale  $\mathbf{x}_M\in \mathbb{R}^{\lfloor \frac{T}{2^M}\rfloor \times C}$ , ensuring the preservation of global context. This mechanism fuses information across variables, enabling the extraction of inter-variable patterns. Subsequently, the multivariate time series is projected into an embedding space via the function  $\operatorname {Embed}(\cdot):\mathbb{R}^{\lfloor \frac{T}{2^M}\rfloor \times C}\to \mathbb{R}^{\lfloor \frac{T}{2^M}\rfloor \times d_{\mathrm{model}}}$ , capturing temporal structure at different scales and facilitating comprehensive pattern learning across the input time series.

![](images/03a3116f8eb3617f96ab6282449a73296ddd41fb776af598833af16c482b2913.jpg)



Figure 5: Illustration of the channel mixing approach and embedding function in the input projection process. This process highlights how variate-wise self-attention captures inter-variable dependencies at the coarsest scale, followed by the projection into an embedding space.


Multi-resolution Time Imaging. Starting from the coarsest scale  $\mathbf{x}_M^l$ , the fast fourier transform (FFT) is applied to extract the top- $K$  frequencies, corresponding to dominant periods in the series. These top- $K$  periods, which capture global patterns, are applied across all scales. At each scale  $m$ , the input time series  $\mathbf{x}_m^l$  is reshaped into  $K$  time images by padding the series according to the identified periods and reshaping it (Equation. 6). The size of each image, denoted as  $\mathbf{z}_m^{(l,k)}$ , is  $p_k \times f_{m,k}$ . As shown in Figure 6, this process produces multi-resolution time images that capture both temporal and frequency domain patterns, enabling the extraction of comprehensive periodic structures.

![](images/1fadde1bea60aa93bf58631557a1411c38bb15dc299710b4de0ee041b1b370c1.jpg)



Figure 6: Multi-resolution Time Imaging. We illustrate the generation of multi-resolution images using the top-3 frequencies and three scales.


Time Image Decomposition. As depicted in Figure 7, each time image  $\mathbf{z}_m^{(l,k)}\in \mathbb{R}^{p_k\times f_{\mathrm{m,k}}\times d_{\mathrm{model}}}$  corresponds to a specific scale and period. The columns represent time segments within each period, while the rows track consistent points across periods. This structure allows us to apply column-axis

![](images/19c5bcfca29e0491179f2738ecc06641ff3cfe4ec9885fbdf7b4979383bfb235.jpg)



Figure 7: Time Image Decomposition. We demonstrate how the identified periods are used to convert the time series into the time image, and how dual-axis attention is applied to extract both seasonal and trend patterns from this image.


attention, capturing seasonality within a period, and row-axis attention, capturing trend across periods. Column-axis attention processes temporal dependencies within periods using 2D convolution to compute the queries, keys, and values  $(\mathbf{Q}_{\mathrm{col}},\mathbf{K}_{\mathrm{col}},\mathbf{V}_{\mathrm{col}}\in \mathbb{R}^{f_{\mathrm{m,k}}\times d_{\mathrm{model}}})$  , with the row axis transposed into the batch dimension. Similarly, row-axis attention employs 2D convolution to compute queries, keys, and values  $(\mathbf{Q}_{\mathrm{row}},\mathbf{K}_{\mathrm{row}},\mathbf{V}_{\mathrm{row}}\in \mathbb{R}^{p_k\times d_{\mathrm{model}}})$  , where the column axis is transposed into the batch dimension. By leveraging this dual-axis attention, we disentangle seasonality and trends for each image. The seasonal image  $\mathbf{s}_m^{(l,k)}$  and the trend image  $\mathbf{t}_m^{(l,k)}$  effectively preserve key patterns, facilitating the extraction of long-range dependencies and enabling clearer temporal analysis.

Multi-scale Mixing. The Figure 8 illustrates the process of multi-scale mixing as formalized in Equations 8 and 9. In this approach, we hierarchically mix the multi-scale seasonal and trend patterns. For seasonal patterns, the mixing begins at the finest scale  $\mathbf{s}_0^{(l,k)}$  and proceeds in a bottom-up fashion through successive scales. Conversely, for trend patterns, the mixing starts from the coarsest scale  $\mathbf{t}_M^{(l,k)}$  and flows top-down to finer scales. This hierarchical flow enables effective integration of both long-term and short-term patterns, allowing finer patterns to be aggregated into seasonal representations, while coarser trend information is propagated downward to refine trend representations at finer scales. The effectiveness of this multi-scale mixing is further demonstrated by the representation analysis provided in Appendix D.

![](images/f6659b15e22a85bbed33cd42186d4a8943eaec7f46aeba69ea01e54673d79ebc.jpg)


![](images/426093c20d763e290b0612c8f7deb84139b25154d97fda5deecadd26c0d1a411.jpg)



Figure 8: Multi-Scale Mixing. We illustrate the hierarchical mixing of multi-scale seasonal and trend images. Each scale's output (red symbol) integrates all preceding information. 2D convolutions are used in the bottom-up path, while transposed convolutions are applied in the top-down path to accommodate changes in size.


![](images/bfe5402d7cfebb6e17e967a174cfa4eb829c28bf2a2243553676a7e972779e29.jpg)



Figure 9: Multi-resolution Mixing. At each scale,  $K$  period-based representations are fused after season-trend mixing to produce  $M$  scale-specific embeddings, which can be further ensembled for the final output.


Multi-resolution Mixing. As shown in Figure 9, at each scale  $m$ , the  $K$  period-based representations, denoted as  $\mathbf{z}_m^{(l,k)}$ , are first weighted by their corresponding FFT amplitudes  $\mathbf{A}_{f_k}$ , which capture the importance of each period, and then summed to produce the final representation for that scale. This process, repeated across all scales, yields a comprehensive embedding for each scale, capturing multi-resolution information.

# C ADDITIONAL ABLATION STUDIES

To verify the effectiveness of each component of TIMEMIXER++, we conducted a detailed ablation study by performing experiments that remove individual components (w/o) across various tasks, including univariate short-term forecasting, multivariate short-term forecasting, and anomaly detection. Based on the Table 11 12 provided, the conclusions are as follows: TIMEMIXER++ outperforms other configurations in short-term forecasting with the lowest average SMAPE and MAPE scores, indicating the importance of each component. As for PEMS datasets with large variable dimensions, the most significant improvement is observed with channel mixing, showing a  $14.95\%$  improvement. In anomaly detection, TIMEMIXER++ achieves the highest average F1 score, with time image decomposition contributing the most to performance, showing a  $9.8\%$  improvement. Other components like channel mixing, multi-scale mixing, and multi-resolution mixing also enhance performance. Overall, each component plays a crucial role in the effectiveness of TIMEMIXER++ for all tasks.


Table 11: SMAPE&MAPE for short-term forecasting across 5 benchmarks, evaluated with different model components.


<table><tr><td></td><td>M4</td><td>PEMS03</td><td>PEMS04</td><td>PEMS07</td><td>PEMS08</td><td>Average</td><td>Promotion</td></tr><tr><td>TIMEMIXER++</td><td>11.45</td><td>13.43</td><td>11.34</td><td>7.32</td><td>8.21</td><td>10.35</td><td>-</td></tr><tr><td>w/o channel mixing</td><td>11.44</td><td>15.57</td><td>13.31</td><td>9.74</td><td>10.78</td><td>12.17</td><td>14.95%</td></tr><tr><td>w/o time image decomposition</td><td>12.37</td><td>15.59</td><td>12.97</td><td>9.65</td><td>9.97</td><td>12.11</td><td>14.51%</td></tr><tr><td>w/o multi-scale mixing</td><td>11.98</td><td>14.97</td><td>13.02</td><td>9.17</td><td>9.69</td><td>11.79</td><td>12.06%</td></tr><tr><td>w/o multi-resolution mixing</td><td>11.87</td><td>15.02</td><td>13.14</td><td>8.72</td><td>9.53</td><td>11.68</td><td>11.23%</td></tr></table>


Table 12: F1 score for anomaly detection across 5 benchmarks, evaluated with different model components.


<table><tr><td></td><td>SMD</td><td>MSL</td><td>SMAP</td><td>SWAT</td><td>PSM</td><td>Average</td><td>Promotion</td></tr><tr><td>TIMEMIXER++</td><td>86.50</td><td>85.82</td><td>73.10</td><td>94.64</td><td>97.60</td><td>87.47</td><td>-</td></tr><tr><td>w/o channel mixing</td><td>84.51</td><td>74.03</td><td>70.91</td><td>90.41</td><td>96.17</td><td>83.21</td><td>4.94%</td></tr><tr><td>w/o time image decomposition</td><td>81.21</td><td>72.43</td><td>66.02</td><td>82.41</td><td>92.53</td><td>78.92</td><td>9.84%</td></tr><tr><td>w/o multi-scale mixing</td><td>82.37</td><td>75.12</td><td>92.79</td><td>86.48</td><td>94.53</td><td>86.26</td><td>1.46%</td></tr><tr><td>w/o multi-resolution mixing</td><td>83.37</td><td>79.24</td><td>77.49</td><td>88.46</td><td>96.02</td><td>86.26</td><td>2.99%</td></tr></table>

# D ADDITIONAL REPRESENTATION ANALYSIS

To evaluate the representational capabilities of TIMEMIXER++, we selected three datasets: Traffic, Electricity, and ETTm1, each exhibiting distinct periodic and trend characteristics. We conducted comprehensive analyses across various scales and resolutions. Initially, 1D convolution was employed to downsample the original time series, yielding different scales. Subsequently, frequency spectrum analysis was utilized to identify the primary frequency components within the time series, selecting the top three components by magnitude as the primary resolutions. This process transformed the original time series into a multi-scale time image. Time image decomposition was then applied to disentangle seasonality and trends from the original time image, resulting in distinct seasonality and trend images. Hierarchical mixing was performed to facilitate interactions across different scales. The visualization of the resulting representations is depicted in Figures 10, 11, and 13.

![](images/bee90198bd08739d6acb48ff84b9131e36f310f82b1131fba538913b3dec0677.jpg)



Figure 10: Visualization of representation on Time Image under Traffic dataset. We provide different scales and different resolutions in Time Image, seasonality, and trend.


As shown in the Figure 10, we visualized the representation of the time series in the Traffic dataset across three different scales (length: 192, 96, 48) and three resolutions (periods: 12, 8, 6; frequencies: 16, 24, 32). From the visualization, we can observe that we successfully disentangled the seasonality and trend within the traffic time series. The seasonal patterns vary across different scales and resolutions: at a fine scale (scale: 0), higher frequency seasonality (period:32; freq: 6) is more prominent, while at a coarse scale (scale: 2), lower frequency seasonality (period:16; freq: 12) is more evident. Additionally, we successfully isolated the upward trend across all scales and resolutions. This demonstrates the strong representational capability of TIMEMIXER++, highlighting its potential as a general time series pattern machine.

We can observe the visualization of the representation of the ETTm1 dataset, which exhibits multi-period characteristics and a downward trend, in Figure 11. We obtained findings consistent with previous observations: higher frequency components are more prominent at a fine scale, while lower frequency components are more easily observed at a coarse scale. This observation aligns perfectly with theoretical intuition (Harti, 1993), as the fine scale retains more detailed information, such as higher frequency seasonality, from a microscopic perspective, whereas the coarse scale provides a macroscopic view where more low-frequency global information is more evident. Moreover, it is important to note that the powerful representational capability of TIMEMIXER++ allows it to accurately capture the downward trend. These conclusions further underscore the necessity of our multi-scale and multi-resolution design.

![](images/9f9fab0549a68eb2693591c6e7fb824473debe4cc43be4fb885e769a83db45b2.jpg)



Figure 11: Visualization of representation on Time Image under ETTm1 dataset. We provide different scales and different resolutions in Time Image, seasonality, and trend.


In Figure 13, we present the visualization of the representation of the Electricity dataset. A prominent feature is that the trend is time-varying, initially declining and then rising over time. TIMEMIXER++ successfully captures this time-varying characteristic in its representation. From the figure, we can observe the gradient features in the trend image, which undoubtedly demonstrate the powerful representational capability of the TIMEMIXER++. Especially when combined with our multi-scale and multi-resolution design, the representation exhibits hierarchical performance across different scales and resolutions. This greatly enhances the richness of the representation, making TIMEMIXER++ adept at handling universal predictive analysis.

Through the preceding analyses, it is evident that the paradigm of multi-scale time series modeling (Mozer, 1991) and multi-resolution analysis (Harti, 1993) endows TIMEMIXER++ with robust representational capabilities. This enables proficient handling of high-frequency, low-frequency, seasonal, and trend components within time series data. Such capabilities are pivotal to achieving state-of-the-art performance across diverse and comprehensive time series analysis tasks, underscoring its essential role as a time series pattern machine.

As shown in Figure 12, TIMEMIXER++ consistently excels across all four tasks, achieving state-of-the-art performance in each scenario. The model effectively adapts its representation transformations to meet the demands of different tasks. Whether preserving representations or enabling significant

![](images/6b1ea0615ef0c4d58271088bf82bf8a64e08bf2cecea9807576eccb5b3d1c58d.jpg)


![](images/368fcf898b1b2c3b4899bc956b486743359a5656e3d81b635fa98cca2bbda2e3.jpg)


![](images/d2c48fa81bafa633a94ea296fd17ae95cc8c7261c5704308b44a0ed348c3ebc6.jpg)


![](images/549fd110079ab0edaab91685a56efaa5a22b1467ae1ebe1ecbbfdeebb1d27c1a.jpg)



Figure 12: Representation analysis in four tasks. For each model, the centered kernel alignment (CKA) similarity is computed between representations from the first and the last layers.


![](images/140a7d65585386acaa96cb0afb74eedfbec58091bf7cc4b5362afee20e9fb9aa.jpg)



Figure 13: Visualization of representation on Time Image under Electricity dataset. We provide different scales and different resolutions in Time Image, seasonality, and trend.


changes across layers, TimeMixer++ demonstrates its versatility and robustness in handling a wide range of time-series analysis challenges.

# E EFFICIENCY ANALYSIS

We comprehensively compare the forecasting and imputation in performance, training speed, and memory footprint of the following models: TIMEMIXER++, iTransformer(Liu et al., 2024), PatchTST(Nie et al., 2023), TimeMixer(Wang et al., 2024b), TIDE(Das et al., 2023b), Fedformer(Zhou et al., 2022b), TimesNet(Wu et al., 2023), MICN(Wang et al., 2023a), and SCINet(Liu et al., 2022a).

![](images/b2b3573b460775e2e6ba2965e76c9a99886ce321a6dc0fdff0f92f35ad450e57.jpg)


![](images/36142a4e6d88720f1ef2b5d69b46c0aaae6e7f4f7c7ea3d4e93e47fdf9e57069.jpg)



Figure 14: Model efficiency comparison under imputation and long-term forecasting.


As shown in Figure 14, TIMEMIXER++ achieves a comparable balance among memory footprint, training time, and performance in both the Weather imputation and long-term forecasting tasks on ETTm1, delivering the MSE scores.

# F HYPERPARAMETER SENSITIVITY

We conduct a hyperparameter sensitivity analysis focusing on the four important hyperparameters within TIMEMIXER++: namely, the number of scales  $M$ , the number of layers  $L$ , the time series input length  $T$ , and the selection of the top  $K$  periods with the highest amplitudes in the spectrogram. The related findings are presented in Figure 15. Based on our analysis, we have made the following observations: (1) As the number of scales increases, the MSE generally decreases. Increasing the number of scales benefits model performance across all prediction lengths, with noticeable improvements observed between 3 and 4 scales. Considering overall performance and efficiency, the

![](images/5dc1df141a496373eedf85d8eb7b6c826ee9d662c2360a7f6f365be2e2341050.jpg)


![](images/adc51bf001c1c90bff83ff11ab545e2fcc31ce74be40143a715667f4f483b1e0.jpg)


![](images/df03f71c33b954529db80bc2a707cd98c077c212ef55ab4a9e9349c1a5e9d628.jpg)


![](images/acb002652b231a7b367af192e152949caa5a110214eacad08a55d5ac1cf110d4.jpg)



Figure 15: Analysis of hyperparameter sensitivity on ETTm1 dataset.


marginal benefits of increasing  $M$  significantly diminish, so we can set  $M$  from 1 to 3. (2) Adding more layers typically reduces MSE, particularly between 1 and 2 layers where the change is most pronounced. For shorter prediction lengths (e.g., predict-96), increasing the number of layers results in more significant performance gains. (3) Increasing the selection of Top-K generally leads to a reduction in MSE. For longer prediction lengths, the choice of Top-K has a more substantial impact on the results. (4) As the input length increases, the MSE gradually decreases. Longer input lengths help improve prediction accuracy across all prediction lengths, which indicates that using longer inputs may achieve better predictive performance for the TIMEMIXER++.

# G ERROR BARS

In this paper, we repeat all the experiments three times. Here we report the standard deviation of our model and the second best model, as well as the statistical significance test in Table 13, 14, 15.


Table 13: Standard deviation and statistical tests for our TIMEMIXER++ and second-best method (iTransformer) on ETT, Weather, Solar-Energy, Electricity and Traffic datasets.


<table><tr><td>Model</td><td colspan="2">TimeMixer++</td><td colspan="2">iTransformer (2024)</td><td>Confidence</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>Level</td></tr><tr><td>Weather</td><td>0.226 ± 0.008</td><td>0.262 ± 0.007</td><td>0.258 ± 0.009</td><td>0.278 ± 0.006</td><td>99%</td></tr><tr><td>Solar-Energy</td><td>0.203 ± 0.027</td><td>0.238 ± 0.026</td><td>0.233 ± 0.009</td><td>0.262 ± 0.011</td><td>95%</td></tr><tr><td>Electricity</td><td>0.165 ± 0.017</td><td>0.253 ± 0.019</td><td>0.178 ± 0.002</td><td>0.270 ± 0.017</td><td>99%</td></tr><tr><td>Traffic</td><td>0.416 ± 0.027</td><td>0.264 ± 0.030</td><td>0.428 ± 0.008</td><td>0.282 ± 0.027</td><td>95%</td></tr><tr><td>ETTh1</td><td>0.419 ± 0.023</td><td>0.432 ± 0.021</td><td>0.454 ± 0.004</td><td>0.447 ± 0.007</td><td>99%</td></tr><tr><td>ETTh2</td><td>0.339 ± 0.020</td><td>0.380 ± 0.019</td><td>0.383 ± 0.004</td><td>0.407 ± 0.007</td><td>95%</td></tr><tr><td>ETTm1</td><td>0.369 ± 0.019</td><td>0.378 ± 0.026</td><td>0.407 ± 0.004</td><td>0.410 ± 0.009</td><td>99%</td></tr><tr><td>ETTm2</td><td>0.269 ± 0.021</td><td>0.320 ± 0.019</td><td>0.288 ± 0.010</td><td>0.332 ± 0.003</td><td>95%</td></tr></table>


Table 14: Standard deviation and statistical tests for our TimeMixer++ method and second-best method (TimeMixer) on PEMS dataset.


<table><tr><td>Model</td><td colspan="3">TimeMixer++</td><td colspan="3">TimeMixer (2024b)</td><td>Confidence</td></tr><tr><td>Dataset</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>Level</td></tr><tr><td>PEMS03</td><td>13.99 ± 0.271</td><td>13.43 ± 0.292</td><td>24.03 ± 0.269</td><td>14.63 ± 0.471</td><td>14.54 ± 0.502</td><td>23.28 ± 0.468</td><td>99%</td></tr><tr><td>PEMS04</td><td>17.46 ± 0.951</td><td>11.34 ± 0.970</td><td>28.83 ± 0.916</td><td>19.21 ± 0.511</td><td>12.53 ± 0.523</td><td>30.92 ± 0.519</td><td>95%</td></tr><tr><td>PEMS07</td><td>18.38 ± 0.991</td><td>7.32 ± 0.977</td><td>31.75 ± 0.890</td><td>20.57 ± 0.372</td><td>8.62 ± 0.399</td><td>33.59 ± 0.375</td><td>95%</td></tr><tr><td>PEMS08</td><td>13.81 ± 0.827</td><td>8.21 ± 0.836</td><td>23.62 ± 0.877</td><td>15.22 ± 0.311</td><td>9.67 ± 0.332</td><td>24.26 ± 0.317</td><td>99%</td></tr></table>


Table 15: Standard deviation and statistical tests for our TimeMixer++ method and second-best method (TimesMixer) on M4 dataset.


<table><tr><td>Model</td><td colspan="3">TimeMixer++</td><td colspan="3">TimesMixer (2024b)</td><td>Confidence</td></tr><tr><td>Dataset</td><td>SMAPE</td><td>MASE</td><td>OWA</td><td>SMAPE</td><td>MASE</td><td>OWA</td><td>Level</td></tr><tr><td>Yearly</td><td>13.179 ± 0.021</td><td>2.934 ± 0.012</td><td>0.769 ± 0.001</td><td>13.206 ± 0.121</td><td>2.916 ± 0.022</td><td>0.776 ± 0.002</td><td>95%</td></tr><tr><td>Quarterly</td><td>9.755 ± 0.001</td><td>1.159 ± 0.005</td><td>0.865 ± 0.009</td><td>9.996 ± 0.101</td><td>1.166 ± 0.015</td><td>0.825 ± 0.008</td><td>95%</td></tr><tr><td>Monthly</td><td>12.432 ± 0.015</td><td>0.904 ± 0.012</td><td>0.841 ± 0.001</td><td>12.605 ± 0.115</td><td>0.919 ± 0.011</td><td>0.869 ± 0.003</td><td>95%</td></tr><tr><td>Others</td><td>4.698 ± 0.114</td><td>2.931 ± 0.027</td><td>1.01 ± 0.011</td><td>4.564 ± 0.114</td><td>3.115 ± 0.027</td><td>0.982 ± 0.011</td><td>99%</td></tr><tr><td>Averaged</td><td>11.448 ± 0.007</td><td>1.487 ± 0.010</td><td>0.821 ± 0.002</td><td>11.723 ± 0.011</td><td>1.559 ± 0.022</td><td>0.840 ± 0.001</td><td>99%</td></tr></table>

# H FULL RESULTS

Due to the space limitation of the main text, we place the full results of all experiments in the following: long-term forecasting in Table 16, univariate short-term forecasting in Table 17, multivariate short-term forecasting in Table 18, classification in Table 19 and anomaly detection in Table 20.

# I SHOWCASES

To assess the performance of various models, we perform a qualitative comparison by visualizing the final dimension of the forecasting results derived from the test set of each dataset (Figures 16, 17, 18, 19, 20, 22, 23, 24, 25, 21). Among the various models, TIMEMIXER++ exhibits superior performance.

# J BROADER IMPACT

Real-world applications TIMEMIXER++ has achieved state-of-the-art (SOTA) performance as a general time series pattern machine in various time series analysis tasks, including forecasting,


Table 16: Full results for the long-term forecasting task. We compare extensive competitive models under different prediction lengths.  $Avg$  is averaged from all four prediction lengths, that  $\{96, 192, 336, 720\}$ .


<table><tr><td colspan="2">Models</td><td colspan="2">TimeMixer++(Ours)</td><td colspan="2">TimeMixer (2024b)</td><td colspan="2">iTransformer 2024</td><td colspan="2">PatchTST 2023</td><td colspan="2">Crossformer 2023</td><td colspan="2">TiDE 2023a</td><td colspan="2">TimesNet 2023</td><td colspan="2">DLinear 2023</td><td colspan="2">SCINet 2022a</td><td colspan="2">FEDformer 2022b</td><td colspan="2">Stationary 2022c</td><td colspan="2">Autoform 2021</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.155</td><td>0.205</td><td>0.163</td><td>0.209</td><td>0.174</td><td>0.214</td><td>0.186</td><td>0.227</td><td>0.195</td><td>0.271</td><td>0.202</td><td>0.261</td><td>0.172</td><td>0.220</td><td>0.195</td><td>0.252</td><td>0.221</td><td>0.306</td><td>0.217</td><td>0.296</td><td>0.173</td><td>0.223</td><td>0.266</td><td>0.336</td></tr><tr><td>192</td><td>0.201</td><td>0.245</td><td>0.208</td><td>0.250</td><td>0.221</td><td>0.254</td><td>0.234</td><td>0.265</td><td>0.209</td><td>0.277</td><td>0.242</td><td>0.298</td><td>0.219</td><td>0.261</td><td>0.237</td><td>0.295</td><td>0.261</td><td>0.340</td><td>0.276</td><td>0.336</td><td>0.245</td><td>0.285</td><td>0.307</td><td>0.367</td></tr><tr><td>336</td><td>0.237</td><td>0.265</td><td>0.251</td><td>0.287</td><td>0.278</td><td>0.296</td><td>0.284</td><td>0.301</td><td>0.273</td><td>0.332</td><td>0.287</td><td>0.335</td><td>0.280</td><td>0.306</td><td>0.282</td><td>0.331</td><td>0.309</td><td>0.378</td><td>0.339</td><td>0.380</td><td>0.321</td><td>0.338</td><td>0.359</td><td>0.395</td></tr><tr><td>720</td><td>0.312</td><td>0.334</td><td>0.339</td><td>0.341</td><td>0.358</td><td>0.347</td><td>0.356</td><td>0.349</td><td>0.379</td><td>0.401</td><td>0.351</td><td>0.386</td><td>0.365</td><td>0.359</td><td>0.345</td><td>0.382</td><td>0.377</td><td>0.427</td><td>0.403</td><td>0.428</td><td>0.414</td><td>0.410</td><td>0.419</td><td>0.428</td></tr><tr><td></td><td>Avg</td><td>0.226</td><td>0.262</td><td>0.240</td><td>0.271</td><td>0.258</td><td>0.278</td><td>0.265</td><td>0.285</td><td>0.264</td><td>0.320</td><td>0.271</td><td>0.320</td><td>0.259</td><td>0.287</td><td>0.265</td><td>0.315</td><td>0.292</td><td>0.363</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td></tr><tr><td rowspan="4">Solar-Energy</td><td>96</td><td>0.171</td><td>0.231</td><td>0.189</td><td>0.259</td><td>0.203</td><td>0.237</td><td>0.265</td><td>0.323</td><td>0.232</td><td>0.302</td><td>0.312</td><td>0.399</td><td>0.373</td><td>0.358</td><td>0.290</td><td>0.378</td><td>0.237</td><td>0.344</td><td>0.286</td><td>0.341</td><td>0.321</td><td>0.380</td><td>0.456</td><td>0.446</td></tr><tr><td>192</td><td>0.218</td><td>0.263</td><td>0.222</td><td>0.283</td><td>0.233</td><td>0.261</td><td>0.288</td><td>0.332</td><td>0.371</td><td>0.410</td><td>0.339</td><td>0.416</td><td>0.397</td><td>0.376</td><td>0.320</td><td>0.398</td><td>0.280</td><td>0.380</td><td>0.291</td><td>0.337</td><td>0.346</td><td>0.369</td><td>0.588</td><td>0.561</td></tr><tr><td>336</td><td>0.212</td><td>0.269</td><td>0.231</td><td>0.292</td><td>0.248</td><td>0.273</td><td>0.301</td><td>0.339</td><td>0.495</td><td>0.515</td><td>0.368</td><td>0.430</td><td>0.420</td><td>0.380</td><td>0.353</td><td>0.415</td><td>0.304</td><td>0.389</td><td>0.354</td><td>0.416</td><td>0.357</td><td>0.387</td><td>0.595</td><td>0.588</td></tr><tr><td>720</td><td>0.212</td><td>0.270</td><td>0.223</td><td>0.285</td><td>0.249</td><td>0.275</td><td>0.295</td><td>0.336</td><td>0.526</td><td>0.542</td><td>0.370</td><td>0.425</td><td>0.420</td><td>0.381</td><td>0.357</td><td>0.413</td><td>0.308</td><td>0.388</td><td>0.380</td><td>0.437</td><td>0.375</td><td>0.424</td><td>0.733</td><td>0.633</td></tr><tr><td></td><td>Avg</td><td>0.203</td><td>0.238</td><td>0.216</td><td>0.280</td><td>0.233</td><td>0.262</td><td>0.287</td><td>0.333</td><td>0.406</td><td>0.442</td><td>0.347</td><td>0.417</td><td>0.403</td><td>0.374</td><td>0.330</td><td>0.401</td><td>0.282</td><td>0.375</td><td>0.328</td><td>0.383</td><td>0.350</td><td>0.390</td><td>0.586</td><td>0.557</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.135</td><td>0.222</td><td>0.153</td><td>0.247</td><td>0.148</td><td>0.240</td><td>0.190</td><td>0.296</td><td>0.219</td><td>0.314</td><td>0.237</td><td>0.329</td><td>0.168</td><td>0.272</td><td>0.210</td><td>0.302</td><td>0.247</td><td>0.345</td><td>0.193</td><td>0.308</td><td>0.169</td><td>0.273</td><td>0.201</td><td>0.317</td></tr><tr><td>192</td><td>0.147</td><td>0.235</td><td>0.166</td><td>0.256</td><td>0.162</td><td>0.253</td><td>0.199</td><td>0.304</td><td>0.231</td><td>0.322</td><td>0.236</td><td>0.330</td><td>0.184</td><td>0.322</td><td>0.210</td><td>0.305</td><td>0.257</td><td>0.355</td><td>0.201</td><td>0.315</td><td>0.182</td><td>0.286</td><td>0.222</td><td>0.334</td></tr><tr><td>336</td><td>0.164</td><td>0.245</td><td>0.185</td><td>0.277</td><td>0.178</td><td>0.269</td><td>0.217</td><td>0.319</td><td>0.246</td><td>0.337</td><td>0.249</td><td>0.344</td><td>0.198</td><td>0.300</td><td>0.223</td><td>0.319</td><td>0.269</td><td>0.369</td><td>0.214</td><td>0.329</td><td>0.200</td><td>0.304</td><td>0.231</td><td>0.443</td></tr><tr><td>720</td><td>0.212</td><td>0.310</td><td>0.225</td><td>0.310</td><td>0.225</td><td>0.317</td><td>0.258</td><td>0.352</td><td>0.280</td><td>0.363</td><td>0.284</td><td>0.373</td><td>0.220</td><td>0.320</td><td>0.258</td><td>0.350</td><td>0.299</td><td>0.390</td><td>0.246</td><td>0.355</td><td>0.222</td><td>0.321</td><td>0.254</td><td>0.361</td></tr><tr><td></td><td>Avg</td><td>0.165</td><td>0.253</td><td>0.182</td><td>0.272</td><td>0.178</td><td>0.270</td><td>0.216</td><td>0.318</td><td>0.244</td><td>0.334</td><td>0.251</td><td>0.344</td><td>0.192</td><td>0.304</td><td>0.225</td><td>0.319</td><td>0.268</td><td>0.365</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.392</td><td>0.253</td><td>0.462</td><td>0.285</td><td>0.395</td><td>0.268</td><td>0.526</td><td>0.347</td><td>0.644</td><td>0.429</td><td>0.805</td><td>0.493</td><td>0.593</td><td>0.321</td><td>0.650</td><td>0.396</td><td>0.788</td><td>0.499</td><td>0.587</td><td>0.366</td><td>0.612</td><td>0.338</td><td>0.613</td><td>0.388</td></tr><tr><td>192</td><td>0.402</td><td>0.258</td><td>0.473</td><td>0.296</td><td>0.417</td><td>0.276</td><td>0.522</td><td>0.332</td><td>0.665</td><td>0.431</td><td>0.756</td><td>0.474</td><td>0.617</td><td>0.336</td><td>0.598</td><td>0.370</td><td>0.789</td><td>0.505</td><td>0.604</td><td>0.373</td><td>0.613</td><td>0.340</td><td>0.616</td><td>0.382</td></tr><tr><td>336</td><td>0.428</td><td>0.263</td><td>0.498</td><td>0.296</td><td>0.433</td><td>0.283</td><td>0.517</td><td>0.334</td><td>0.674</td><td>0.420</td><td>0.762</td><td>0.477</td><td>0.629</td><td>0.336</td><td>0.605</td><td>0.373</td><td>0.797</td><td>0.508</td><td>0.621</td><td>0.383</td><td>0.618</td><td>0.328</td><td>0.622</td><td>0.337</td></tr><tr><td>720</td><td>0.441</td><td>0.282</td><td>0.506</td><td>0.313</td><td>0.467</td><td>0.302</td><td>0.552</td><td>0.352</td><td>0.683</td><td>0.424</td><td>0.719</td><td>0.449</td><td>0.640</td><td>0.350</td><td>0.645</td><td>0.394</td><td>0.841</td><td>0.523</td><td>0.626</td><td>0.382</td><td>0.653</td><td>0.355</td><td>0.660</td><td>0.408</td></tr><tr><td></td><td>Avg</td><td>0.416</td><td>0.264</td><td>0.484</td><td>0.297</td><td>0.428</td><td>0.282</td><td>0.529</td><td>0.341</td><td>0.667</td><td>0.426</td><td>0.760</td><td>0.473</td><td>0.620</td><td>0.336</td><td>0.625</td><td>0.383</td><td>0.804</td><td>0.509</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.085</td><td>0.214</td><td>0.090</td><td>0.235</td><td>0.086</td><td>0.206</td><td>0.088</td><td>0.205</td><td>0.256</td><td>0.367</td><td>0.094</td><td>0.218</td><td>0.107</td><td>0.234</td><td>0.088</td><td>0.218</td><td>0.267</td><td>0.396</td><td>0.148</td><td>0.278</td><td>0.111</td><td>0.237</td><td>0.197</td><td>0.323</td></tr><tr><td>192</td><td>0.175</td><td>0.313</td><td>0.187</td><td>0.343</td><td>0.177</td><td>0.299</td><td>0.176</td><td>0.299</td><td>0.470</td><td>0.509</td><td>0.184</td><td>0.307</td><td>0.226</td><td>0.344</td><td>0.176</td><td>0.315</td><td>0.351</td><td>0.459</td><td>0.271</td><td>0.315</td><td>0.219</td><td>0.335</td><td>0.300</td><td>0.369</td></tr><tr><td>336</td><td>0.316</td><td>0.420</td><td>0.353</td><td>0.473</td><td>0.331</td><td>0.417</td><td>0.301</td><td>0.397</td><td>1.268</td><td>0.883</td><td>0.349</td><td>0.431</td><td>0.367</td><td>0.448</td><td>0.313</td><td>0.427</td><td>1.324</td><td>0.853</td><td>0.460</td><td>0.427</td><td>0.421</td><td>0.476</td><td>0.509</td><td>0.524</td></tr><tr><td>720</td><td>0.851</td><td>0.689</td><td>0.934</td><td>0.761</td><td>0.847</td><td>0.691</td><td>0.901</td><td>0.714</td><td>1.767</td><td>1.068</td><td>0.852</td><td>0.698</td><td>0.964</td><td>0.746</td><td>0.839</td><td>0.695</td><td>1.058</td><td>0.797</td><td>1.195</td><td>0.695</td><td>1.092</td><td>0.769</td><td>1.447</td><td>0.941</td></tr><tr><td></td><td>Avg</td><td>0.357</td><td>0.391</td><td>0.391</td><td>0.453</td><td>0.360</td><td>0.403</td><td>0.367</td><td>0.404</td><td>0.940</td><td>0.707</td><td>0.370</td><td>0.413</td><td>0.416</td><td>0.443</td><td>0.354</td><td>0.414</td><td>0.750</td><td>0.626</td><td>0.519</td><td>0.429</td><td>0.461</td><td>0.454</td><td>0.613</td><td>0.539</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.361</td><td>0.403</td><td>0.375</td><td>0.400</td><td>0.386</td><td>0.405</td><td>0.460</td><td>0.447</td><td>0.423</td><td>0.448</td><td>0.479</td><td>0.464</td><td>0.384</td><td>0.402</td><td>0.397</td><td>0.412</td><td>0.654</td><td>0.599</td><td>0.395</td><td>0.424</td><td>0.513</td><td>0.491</td><td>0.449</td><td>0.459</td></tr><tr><td>192</td><td>0.416</td><td>0.441</td><td>0.429</td><td>0.421</td><td>0.441</td><td>0.512</td><td>0.477</td><td>0.429</td><td>0.471</td><td>0.474</td><td>0.525</td><td>0.492</td><td>0.436</td><td>0.429</td><td>0.446</td><td>0.441</td><td>0.719</td><td>0.631</td><td>0.469</td><td>0.470</td><td>0.534</td><td>0.504</td><td>0.500</td><td>0.482</td></tr><tr><td>336</td><td>0.430</td><td>0.434</td><td>0.484</td><td>0.458</td><td>0.487</td><td>0.458</td><td>0.546</td><td>0.496</td><td>0.570</td><td>0.546</td><td>0.565</td><td>0.515</td><td>0.491</td><td>0.469</td><td>0.489</td><td>0.467</td><td>0.778</td><td>0.659</td><td>0.530</td><td>0.499</td><td>0.588</td><td>0.535</td><td>0.521</td><td>0.496</td></tr><tr><td>720</td><td>0.467</td><td>0.451</td><td>0.498</td><td>0.482</td><td>0.503</td><td>0.491</td><td>0.544</td><td>0.517</td><td>0.653</td><td>0.621</td><td>0.594</td><td>0.558</td><td>0.521</td><td>0.500</td><td>0.513</td><td>0.510</td><td>0.836</td><td>0.699</td><td>0.598</td><td>0.544</td><td>0.643</td><td>0.616</td><td>0.514</td><td>0.512</td></tr><tr><td></td><td>Avg</td><td>0.419</td><td>0.432</td><td>0.447</td><td>0.440</td><td>0.454</td><td>0.447</td><td>0.516</td><td>0.484</td><td>0.529</td><td>0.522</td><td>0.541</td><td>0.507</td><td>0.458</td><td>0.450</td><td>0.461</td><td>0.457</td><td>0.747</td><td>0.647</td><td>0.498</td><td>0.484</td><td>0.570</td><td>0.537</td><td>0.496</td><td>0.487</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.276</td><td>0.328</td><td>0.289</td><td>0.341</td><td>0.297</td><td>0.349</td><td>0.308</td><td>0.355</td><td>0.745</td><td>0.584</td><td>0.400</td><td>0.440</td><td>0.340</td><td>0.374</td><td>0.340</td><td>0.394</td><td>0.707</td><td>0.621</td><td>0.358</td><td>0.397</td><td>0.476</td><td>0.458</td><td>0.346</td><td>0.388</td></tr><tr><td>192</td><td>0.342</td><td>0.379</td><td>0.372</td><td>0.392</td><td>0.380</td><td>0.400</td><td>0.393</td><td>0.405</td><td>0.877</td><td>0.656</td><td>0.528</td><td>0.509</td><td>0.402</td><td>0.414</td><td>0.482</td><td>0.479</td><td>0.860</td><td>0.689</td><td>0.429</td><td>0.439</td><td>0.512</td><td>0.493</td><td>0.456</td><td></td></tr><tr><td>336</td><td>0.346</td><td>0.398</td><td>0.386</td><td>0.414</td><td>0.428</td><td>0.432</td><td>0.427</td><td>0.436</td><td>0.546</td><td>1.043</td><td>0.731</td><td>0.643</td><td>0.571</td><td>0.452</td><td>0.591</td><td>0.541</td><td>1.000</td><td>0.774</td><td>0.444</td><td>0.496</td><td>0.487</td><td>0.552</td><td>0.551</td><td></td></tr><tr><td>720</td><td>0.392</td><td>0.415</td><td>0.412</td><td>0.434</td><td>0.427</td><td>0.445</td><td>0.436</td><td>0.450</td><td>1.104</td><td>0.763</td><td>0.874</td><td>0.679</td><td>0.462</td><td>0.468</td><td>0.839</td><td>0.661</td><td>1.249</td><td>0.838</td><td>0.463</td><td>0.474</td><td>0.562</td><td>0.560</td><td></td><td></td></tr><tr><td></td><td>Avg</td><td>0.339</td><td>0.380</td><td>0.364</td><td>0.395</td><td>0.383</td><td>0.407</td><td>0.391</td><td>0.411</td><td>0.942</td><td>0.684</td><td>0.611</td><td>0.550</td><td>0.414</td><td>0.427</td><td>0.563</td><td>0.519</td><td>0.954</td><td>0.723</td><td>0.437</td><td>0.449</td><td>0.526</td><td>0.516</td><td>0.450</td><td></td></tr><tr><td rowspan="4">ETm1m2</td><td>96</td><td>0.310</td><td>0.334</td><td>0.320</td><td>0.357</td><td>0.334</td><td>0.368</td><td>0.352</td><td>0.374</td><td>0.404</td><td>0.426</td><td>0.364</td><td>0.387</td><td>0.338</td><td>0.375</td><td>0.346</td><td>0.374</td><td>0.418</td><td>0.438</td><td>0.286</td><td>0.377</td><td>0.379</td><td>0.484</td><td>0.386</td><td>0.398</td></tr><tr><td>192</td><td>0.348</td><td>0.362</td><td>0.361</td><td>0.381</td><td>0.390</td><td>0.393</td><td>0.374</td><td>0.387</td><td>0.450</td><td>0.451</td><td>0.398</td><td>0.404</td><td>0.374</td><td>0.387</td><td>0.382</td><td>0.391</td><td>0.451</td><td>0.415</td><td>0.415</td><td>0.490</td><td>0.485</td><td>0.459</td><td>0.553</td><td>0.496</td></tr><tr><td>336</td><td>0.376</td><td>0.391</td><td>0.390</td><td>0.404</td><td>0.426</td><td>0.421</td><td>0.414</td><td>0.532</td><td>0.515</td><td>0.532</td><td>0.428</td><td>0.425</td><td>0.411</td><td>0.415</td><td>0.415</td><td>0.492</td><td>0.489</td><td>0.545</td><td>0.515</td><td>0.543</td><td>0.495</td><td>0.464</td><td>0.621</td><td>0.537</td></tr><tr><td>720</td><td>0.440</td><td>0.423</td><td>0.454</td><td>0.441</td><td>0.491</td><td>0.459</td><td>0.462</td><td>0.449</td><td>0.666</td><td>0.589</td><td>0.487</td><td>0.461</td><td>0.478</td><td>0.450</td><td>0.473</td><td>0.558</td><td>0.525</td><td>0.595</td><td>0.550</td><td>0.543</td><td>0.490</td><td>0.585</td><td>0.516</td><td></td></tr><tr><td></td><td>Avg</td><td>0.369</td><td>0.378</td><td>0.381</td><td>0.395</td><td>0.407</td><td>0.410</td><td>0.406</td><td>0.407</td><td>0.513</td><td>0.495</td><td>0.419</td><td>0.419</td><td>0.400</td><td>0.406</td><td>0.404</td><td>0.485</td><td>0.485</td><td>0.481</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td></td></tr></table>


Table 17: Short-term forecasting results in the M4 dataset with a single variate. All prediction lengths are in [6, 48]. A lower SMAPE, MASE or OWA indicates a better prediction. *, in the Transformers indicates the name of *former. Stationary means the Non-stationary Transformer.


<table><tr><td colspan="2">Models</td><td>TimeMixer++ (Ours)</td><td>TimeMixer iTransformer (2024b)</td><td>TiDE (2023a)</td><td>TimesNet N-HiTS (2023)</td><td>N-BEATS* (2019)</td><td>PatchTST (2023)</td><td>MICN (2023a)</td><td>FiLM (2022a)</td><td>LightTS DLinear (2022)</td><td>FED. (2022b)</td><td>Stationary (2022c)</td><td>Auto. (2021)</td><td></td><td></td><td></td></tr><tr><td rowspan="3">Yearly</td><td>SMAPE</td><td>13.179</td><td>13.206</td><td>13.923</td><td>15.320</td><td>13.387</td><td>13.418</td><td>13.436</td><td>16.463</td><td>25.022</td><td>17.431</td><td>14.247</td><td>16.965</td><td>13.728</td><td>13.717</td><td>13.974</td></tr><tr><td>MASE</td><td>2.934</td><td>2.916</td><td>3.214</td><td>3.540</td><td>2.996</td><td>3.045</td><td>3.043</td><td>3.967</td><td>7.162</td><td>4.043</td><td>3.109</td><td>4.283</td><td>3.048</td><td>3.078</td><td>3.134</td></tr><tr><td>OWA</td><td>0.769</td><td>0.776</td><td>0.830</td><td>0.910</td><td>0.786</td><td>0.793</td><td>0.794</td><td>1.003</td><td>1.667</td><td>1.042</td><td>0.827</td><td>1.058</td><td>0.803</td><td>0.807</td><td>0.822</td></tr><tr><td rowspan="3">Quarterly</td><td>SMAPE</td><td>9.755</td><td>9.996</td><td>10.757</td><td>11.830</td><td>10.100</td><td>10.202</td><td>10.124</td><td>10.644</td><td>15.214</td><td>12.925</td><td>11.364</td><td>12.145</td><td>10.792</td><td>10.958</td><td>11.338</td></tr><tr><td>MASE</td><td>1.159</td><td>1.166</td><td>1.283</td><td>1.410</td><td>1.182</td><td>1.194</td><td>1.169</td><td>1.278</td><td>1.963</td><td>1.664</td><td>1.328</td><td>1.520</td><td>1.283</td><td>1.325</td><td>1.365</td></tr><tr><td>OWA</td><td>0.865</td><td>0.825</td><td>0.956</td><td>1.050</td><td>0.890</td><td>0.899</td><td>0.886</td><td>0.949</td><td>1.407</td><td>1.193</td><td>1.000</td><td>1.106</td><td>0.958</td><td>0.981</td><td>1.012</td></tr><tr><td rowspan="3">Monthly</td><td>SMAPE</td><td>12.432</td><td>12.605</td><td>13.796</td><td>15.180</td><td>12.670</td><td>12.791</td><td>12.677</td><td>13.399</td><td>16.943</td><td>15.407</td><td>14.014</td><td>13.514</td><td>14.260</td><td>13.917</td><td>13.958</td></tr><tr><td>MASE</td><td>0.904</td><td>0.919</td><td>1.083</td><td>1.190</td><td>0.933</td><td>0.969</td><td>0.937</td><td>1.031</td><td>1.442</td><td>1.298</td><td>1.053</td><td>1.037</td><td>1.102</td><td>1.097</td><td>1.103</td></tr><tr><td>OWA</td><td>0.841</td><td>0.869</td><td>0.987</td><td>1.090</td><td>0.878</td><td>0.899</td><td>0.880</td><td>0.949</td><td>1.265</td><td>1.144</td><td>0.981</td><td>0.956</td><td>1.012</td><td>0.998</td><td>1.002</td></tr><tr><td rowspan="3">Others</td><td>SMAPE</td><td>4.698</td><td>4.564</td><td>5.569</td><td>6.120</td><td>4.891</td><td>5.061</td><td>4.925</td><td>6.558</td><td>41.985</td><td>7.134</td><td>15.880</td><td>6.709</td><td>4.954</td><td>6.302</td><td>5.485</td></tr><tr><td>MASE</td><td>2.931</td><td>3.115</td><td>3.940</td><td>4.330</td><td>3.302</td><td>3.216</td><td>3.391</td><td>4.511</td><td>62.734</td><td>5.09</td><td>11.434</td><td>4.953</td><td>3.264</td><td>4.064</td><td>3.865</td></tr><tr><td>OWA</td><td>1.01</td><td>0.982</td><td>1.207</td><td>1.330</td><td>1.035</td><td>1.040</td><td>1.053</td><td>1.401</td><td>14.313</td><td>1.553</td><td>3.474</td><td>1.487</td><td>1.036</td><td>1.304</td><td>1.187</td></tr><tr><td rowspan="3">Weighted Average</td><td>SMAPE</td><td>11.448</td><td>11.723</td><td>12.684</td><td>13.950</td><td>11.829</td><td>11.927</td><td>11.851</td><td>13.152</td><td>19.638</td><td>14.863</td><td>13.525</td><td>13.639</td><td>12.840</td><td>12.780</td><td>12.909</td></tr><tr><td>MASE</td><td>1.487</td><td>1.559</td><td>1.764</td><td>1.940</td><td>1.585</td><td>1.613</td><td>1.559</td><td>1.945</td><td>5.947</td><td>2.207</td><td>2.111</td><td>2.095</td><td>1.701</td><td>1.756</td><td>1.771</td></tr><tr><td>OWA</td><td>0.821</td><td>0.840</td><td>0.929</td><td>1.020</td><td>0.851</td><td>0.861</td><td>0.855</td><td>0.998</td><td>2.279</td><td>1.125</td><td>1.051</td><td>1.051</td><td>0.918</td><td>0.930</td><td>0.939</td></tr></table>


* The original paper of N-BEATS (2019) adopts a special ensemble method to promote the performance. For fair comparisons, we remove the ensemble and only compare the pure forecasting models.



Table 18: Short-term forecasting results in the PEMS datasets with multiple variates. All input lengths are 96 and prediction lengths are 12. A lower MAE, MAPE or RMSE indicates a better prediction.


<table><tr><td colspan="2">Models</td><td>TimeMixer++ (Ours)</td><td>TimeMixer iTransformer (2024b)</td><td>TiDE (2023a)</td><td>SCINet (2022a)</td><td>Crossformer (2023)</td><td>PatchTST (2023)</td><td>TimesNet (2023)</td><td>MICN (2023a)</td><td>DLinear FEDformer (2022b)</td><td>Stationary Autoformer (2022c)</td><td>(2021)</td><td></td><td></td></tr><tr><td rowspan="3">PEMS03</td><td>MAE</td><td>13.99</td><td>14.63</td><td>16.72</td><td>18.39</td><td>15.97</td><td>15.64</td><td>18.95</td><td>16.41</td><td>15.71</td><td>19.70</td><td>19.00</td><td>17.64</td><td>18.08</td></tr><tr><td>MAPE</td><td>13.43</td><td>11.54</td><td>15.81</td><td>17.39</td><td>15.89</td><td>15.74</td><td>17.29</td><td>15.17</td><td>15.67</td><td>18.35</td><td>18.57</td><td>17.56</td><td>18.75</td></tr><tr><td>RMSE</td><td>24.03</td><td>23.28</td><td>27.81</td><td>30.59</td><td>25.20</td><td>25.56</td><td>30.15</td><td>26.72</td><td>24.55</td><td>32.35</td><td>30.05</td><td>28.37</td><td>27.82</td></tr><tr><td rowspan="3">PEMS04</td><td>MAE</td><td>17.46</td><td>19.21</td><td>21.81</td><td>23.99</td><td>20.35</td><td>20.38</td><td>24.86</td><td>21.63</td><td>21.62</td><td>24.62</td><td>26.51</td><td>22.34</td><td>25.00</td></tr><tr><td>MAPE</td><td>11.34</td><td>12.53</td><td>13.42</td><td>14.76</td><td>12.84</td><td>12.84</td><td>16.65</td><td>13.15</td><td>13.53</td><td>16.12</td><td>16.76</td><td>14.85</td><td>16.70</td></tr><tr><td>RMSE</td><td>28.83</td><td>30.92</td><td>33.91</td><td>37.30</td><td>32.31</td><td>32.41</td><td>40.46</td><td>34.90</td><td>34.39</td><td>39.51</td><td>41.81</td><td>35.47</td><td>38.02</td></tr><tr><td rowspan="3">PEMS07</td><td>MAE</td><td>18.38</td><td>20.57</td><td>23.01</td><td>25.31</td><td>22.79</td><td>22.54</td><td>27.87</td><td>25.12</td><td>22.28</td><td>28.65</td><td>27.92</td><td>26.02</td><td>26.92</td></tr><tr><td>MAPE</td><td>7.32</td><td>8.62</td><td>10.02</td><td>11.02</td><td>9.41</td><td>9.38</td><td>12.69</td><td>10.60</td><td>9.57</td><td>12.15</td><td>12.29</td><td>11.75</td><td>11.83</td></tr><tr><td>RMSE</td><td>31.75</td><td>33.59</td><td>35.56</td><td>39.12</td><td>35.61</td><td>35.49</td><td>42.56</td><td>40.71</td><td>35.40</td><td>45.02</td><td>42.29</td><td>42.34</td><td>40.60</td></tr><tr><td rowspan="3">PEMS08</td><td>MAE</td><td>13.81</td><td>15.22</td><td>17.94</td><td>19.73</td><td>17.38</td><td>17.56</td><td>20.35</td><td>19.01</td><td>17.76</td><td>20.26</td><td>20.56</td><td>19.29</td><td>20.47</td></tr><tr><td>MAPE</td><td>8.21</td><td>9.67</td><td>10.93</td><td>12.02</td><td>10.80</td><td>10.92</td><td>13.15</td><td>11.83</td><td>10.76</td><td>12.09</td><td>12.41</td><td>12.21</td><td>12.27</td></tr><tr><td>RMSE</td><td>23.62</td><td>24.26</td><td>27.88</td><td>30.67</td><td>27.34</td><td>27.21</td><td>31.04</td><td>30.65</td><td>27.26</td><td>32.38</td><td>32.97</td><td>38.62</td><td>31.52</td></tr></table>


Table 19: Full results for the anomaly detection task. The P, R and F1 represent the precision, recall and F1-score (\%) respectively. F1-score is the harmonic mean of precision and recall. A higher value of P, R and F1 indicates a better performance.


<table><tr><td rowspan="2" colspan="2">Datasets Metrics</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td><td rowspan="2">Avg F1 (%)</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>LSTM</td><td>(1997)</td><td>78.52</td><td>65.47</td><td>71.41</td><td>78.04</td><td>86.22</td><td>81.93</td><td>91.06</td><td>57.49</td><td>70.48</td><td>78.06</td><td>91.72</td><td>84.34</td><td>69.24</td><td>99.53</td><td>81.67</td><td>77.97</td></tr><tr><td>Transformer</td><td>(2017)</td><td>83.58</td><td>76.13</td><td>79.56</td><td>71.57</td><td>87.37</td><td>78.68</td><td>89.37</td><td>57.12</td><td>69.70</td><td>68.84</td><td>96.53</td><td>80.37</td><td>62.75</td><td>96.56</td><td>76.07</td><td>76.88</td></tr><tr><td>LogTrans</td><td>(2019)</td><td>83.46</td><td>70.13</td><td>76.21</td><td>73.05</td><td>87.37</td><td>79.57</td><td>89.15</td><td>57.59</td><td>69.97</td><td>68.67</td><td>97.32</td><td>80.52</td><td>63.06</td><td>98.00</td><td>76.74</td><td>76.60</td></tr><tr><td>TCN</td><td>(2019)</td><td>84.06</td><td>79.07</td><td>81.49</td><td>75.11</td><td>82.44</td><td>78.60</td><td>86.90</td><td>59.23</td><td>70.45</td><td>76.59</td><td>95.71</td><td>85.09</td><td>54.59</td><td>99.77</td><td>70.57</td><td>77.24</td></tr><tr><td>Reformer</td><td>(2020)</td><td>82.58</td><td>69.24</td><td>75.32</td><td>85.51</td><td>83.31</td><td>84.40</td><td>90.91</td><td>57.44</td><td>70.40</td><td>72.50</td><td>96.53</td><td>82.80</td><td>59.93</td><td>95.38</td><td>73.61</td><td>77.31</td></tr><tr><td>Informer</td><td>(2021a)</td><td>86.60</td><td>77.23</td><td>81.65</td><td>81.77</td><td>86.48</td><td>84.06</td><td>90.11</td><td>57.13</td><td>69.92</td><td>70.29</td><td>96.75</td><td>81.43</td><td>64.27</td><td>96.33</td><td>77.10</td><td>78.83</td></tr><tr><td>Anomaly*</td><td>(2022)</td><td>88.91</td><td>82.23</td><td>85.49</td><td>79.61</td><td>87.37</td><td>83.31</td><td>91.85</td><td>58.11</td><td>71.18</td><td>72.51</td><td>97.32</td><td>83.10</td><td>68.35</td><td>94.72</td><td>79.40</td><td>80.50</td></tr><tr><td>Pyraformer</td><td>(2022b)</td><td>85.61</td><td>80.61</td><td>83.04</td><td>83.81</td><td>85.93</td><td>84.86</td><td>92.54</td><td>57.71</td><td>71.09</td><td>87.92</td><td>96.00</td><td>91.78</td><td>71.67</td><td>96.02</td><td>82.08</td><td>82.57</td></tr><tr><td>Autoformer</td><td>(2021)</td><td>88.06</td><td>82.35</td><td>85.11</td><td>77.27</td><td>80.92</td><td>79.05</td><td>90.40</td><td>58.62</td><td>71.12</td><td>89.85</td><td>95.81</td><td>92.74</td><td>99.08</td><td>88.15</td><td>93.29</td><td>84.26</td></tr><tr><td>LSSL</td><td>(2022b)</td><td>78.51</td><td>65.32</td><td>71.31</td><td>77.55</td><td>88.18</td><td>82.53</td><td>89.43</td><td>53.43</td><td>66.90</td><td>79.05</td><td>93.72</td><td>85.76</td><td>66.02</td><td>92.93</td><td>77.20</td><td>76.74</td></tr><tr><td>Stationary</td><td>(2022c)</td><td>88.33</td><td>81.21</td><td>84.62</td><td>68.55</td><td>89.14</td><td>77.50</td><td>89.37</td><td>59.02</td><td>71.09</td><td>68.03</td><td>96.75</td><td>79.88</td><td>97.82</td><td>96.76</td><td>97.29</td><td>82.08</td></tr><tr><td>DLinear</td><td>(2023)</td><td>83.62</td><td>71.52</td><td>77.10</td><td>84.34</td><td>85.42</td><td>84.88</td><td>92.32</td><td>55.41</td><td>69.26</td><td>80.91</td><td>95.30</td><td>87.52</td><td>98.28</td><td>89.26</td><td>93.55</td><td>82.46</td></tr><tr><td>ETSformer</td><td>(2022)</td><td>87.44</td><td>79.23</td><td>83.13</td><td>85.13</td><td>84.93</td><td>85.03</td><td>92.25</td><td>55.75</td><td>69.50</td><td>90.02</td><td>80.36</td><td>84.91</td><td>99.31</td><td>85.28</td><td>91.76</td><td>82.87</td></tr><tr><td>LightTS</td><td>(2022a)</td><td>87.10</td><td>78.42</td><td>82.53</td><td>82.40</td><td>75.78</td><td>78.95</td><td>92.58</td><td>55.27</td><td>69.21</td><td>91.98</td><td>94.72</td><td>93.33</td><td>98.37</td><td>95.97</td><td>97.15</td><td>84.23</td></tr><tr><td>FEDformer</td><td>(2022b)</td><td>87.95</td><td>82.39</td><td>85.08</td><td>77.14</td><td>80.07</td><td>78.57</td><td>90.47</td><td>58.10</td><td>70.76</td><td>90.17</td><td>96.42</td><td>93.19</td><td>97.31</td><td>97.16</td><td>97.23</td><td>84.97</td></tr><tr><td>TimesNet</td><td>(2023)</td><td>88.66</td><td>83.14</td><td>85.81</td><td>83.92</td><td>86.42</td><td>85.15</td><td>92.52</td><td>58.29</td><td>71.52</td><td>86.76</td><td>97.32</td><td>91.74</td><td>98.19</td><td>96.76</td><td>97.47</td><td>86.34</td></tr><tr><td>TiDE</td><td>(2023a)</td><td>76.00</td><td>63.00</td><td>68.91</td><td>84.00</td><td>60.00</td><td>70.18</td><td>88.00</td><td>50.00</td><td>64.00</td><td>98.00</td><td>63.00</td><td>76.73</td><td>93.00</td><td>92.00</td><td>92.50</td><td>74.46</td></tr><tr><td>iTransformer</td><td>(2024)</td><td>78.45</td><td>65.10</td><td>71.15</td><td>86.15</td><td>62.65</td><td>72.54</td><td>90.67</td><td>52.96</td><td>66.87</td><td>99.96</td><td>65.55</td><td>79.18</td><td>95.65</td><td>94.69</td><td>95.17</td><td>76.98</td></tr><tr><td>TimesMixer++</td><td>(Ours)</td><td>88.59</td><td>84.50</td><td>86.50</td><td>89.73</td><td>82.23</td><td>85.82</td><td>93.47</td><td>60.02</td><td>73.10</td><td>92.96</td><td>94.33</td><td>94.64</td><td>98.33</td><td>96.90</td><td>97.60</td><td>87.47</td></tr></table>


* The original paper of Anomaly Transformer (Xu et al., 2022) adopts the temporal association and reconstruction error as a joint anomaly criterion. For fair comparisons, we only use reconstruction error here.



Table 20: Full results for the classification task. \*. in the Transformers indicates the name of \*former. We report the classification accuracy  $(\%)$  as the result. The standard deviation is within  $0.1\%$ .


<table><tr><td rowspan="4">Datasets / Models</td><td colspan="3">Classical methods</td><td colspan="3">RNN</td><td colspan="3">TCN</td><td colspan="8">Transformers</td><td colspan="2">MLP</td><td colspan="2">CNN</td><td></td></tr><tr><td colspan="3">DTWxGBoost Rocket LSTM LSTM LSTNet LSSL</td><td colspan="17">TCN Trans. Re. In. Pyra. Auto.Station. FED. ETS. Flow.iTrans.DLinearLightTS.TiDE</td><td></td><td></td></tr><tr><td>(1994)</td><td>(2016)</td><td>(2020)(1997)(2018a)(2022b)(2019)</td><td>(2017)</td><td>(2020)(2021a)(2022b)(2021)</td><td colspan="15">(2022c)(2022b)(2022)(2024)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EthanolConcentration</td><td>32.3</td><td>43.7</td><td>45.2</td><td>32.3</td><td>39.9</td><td>31.1</td><td>28.9</td><td>32.7</td><td>31.9</td><td>31.6</td><td>30.8</td><td>31.6</td><td>32.7</td><td>28.1</td><td>31.2</td><td>33.8</td><td>28.1</td><td>32.6</td><td>29.7</td><td>27.1</td><td>35.7</td><td>39.9</td></tr><tr><td>FaceDetection</td><td>52.9</td><td>63.3</td><td>64.7</td><td>57.7</td><td>65.7</td><td>66.7</td><td>52.8</td><td>67.3</td><td>68.6</td><td>67.0</td><td>65.7</td><td>68.4</td><td>68.0</td><td>66.6</td><td>66.3</td><td>67.6</td><td>66.3</td><td>68.0</td><td>67.5</td><td>65.3</td><td>68.6</td><td>71.8</td></tr><tr><td>Handwriting</td><td>28.6</td><td>15.8</td><td>58.8</td><td>15.2</td><td>25.8</td><td>24.6</td><td>53.3</td><td>32.0</td><td>27.4</td><td>32.8</td><td>29.4</td><td>36.7</td><td>31.6</td><td>28.0</td><td>32.5</td><td>33.8</td><td>24.2</td><td>27.0</td><td>26.1</td><td>23.2</td><td>32.1</td><td>26.5</td></tr><tr><td>Heartbeat</td><td>71.7</td><td>73.2</td><td>75.6</td><td>72.2</td><td>77.1</td><td>72.7</td><td>75.6</td><td>76.1</td><td>77.1</td><td>80.5</td><td>75.6</td><td>74.6</td><td>73.7</td><td>73.7</td><td>71.2</td><td>77.6</td><td>75.6</td><td>75.1</td><td>75.1</td><td>74.6</td><td>78.0</td><td>79.1</td></tr><tr><td>JapaneseVowels</td><td>94.9</td><td>86.5</td><td>96.2</td><td>79.7</td><td>98.1</td><td>98.4</td><td>98.9</td><td>98.7</td><td>97.8</td><td>98.9</td><td>98.4</td><td>96.2</td><td>99.2</td><td>98.4</td><td>95.9</td><td>98.9</td><td>96.6</td><td>96.2</td><td>96.2</td><td>95.6</td><td>98.4</td><td>97.9</td></tr><tr><td>PEMS-SF</td><td>71.1</td><td>98.3</td><td>75.1</td><td>39.9</td><td>86.7</td><td>86.1</td><td>68.8</td><td>82.1</td><td>82.7</td><td>81.5</td><td>83.2</td><td>82.7</td><td>87.3</td><td>80.9</td><td>86.0</td><td>83.8</td><td>87.9</td><td>75.1</td><td>88.4</td><td>86.9</td><td>89.6</td><td>91.0</td></tr><tr><td>SelfRegulationSCP1</td><td>77.7</td><td>84.6</td><td>90.8</td><td>68.9</td><td>84.0</td><td>90.8</td><td>84.6</td><td>92.2</td><td>90.4</td><td>90.1</td><td>88.1</td><td>84.0</td><td>89.4</td><td>88.7</td><td>89.6</td><td>92.5</td><td>90.2</td><td>87.3</td><td>89.8</td><td>89.2</td><td>91.8</td><td>93.1</td></tr><tr><td>SelfRegulationSCP2</td><td>53.9</td><td>48.9</td><td>53.3</td><td>46.6</td><td>52.8</td><td>52.2</td><td>55.6</td><td>53.9</td><td>56.7</td><td>53.3</td><td>53.3</td><td>50.6</td><td>57.2</td><td>54.4</td><td>55.0</td><td>56.1</td><td>54.4</td><td>50.5</td><td>51.1</td><td>53.4</td><td>57.2</td><td>65.6</td></tr><tr><td>SpokenArabicDigits</td><td>96.3</td><td>69.6</td><td>71.2</td><td>31.9</td><td>100.0</td><td>100.0</td><td>95.6</td><td>98.4</td><td>97.0</td><td>100.0</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>98.8</td><td>96.0</td><td>81.4</td><td>100.0</td><td>95.0</td><td>99.0</td><td>99.8</td></tr><tr><td>UWaveGestureLibrary</td><td>90.3</td><td>75.9</td><td>94.4</td><td>41.2</td><td>87.8</td><td>85.9</td><td>88.4</td><td>85.6</td><td>85.6</td><td>85.6</td><td>83.4</td><td>85.9</td><td>87.5</td><td>85.3</td><td>85.0</td><td>86.6</td><td>85.9</td><td>82.1</td><td>80.3</td><td>84.9</td><td>85.3</td><td>88.2</td></tr><tr><td>Average Accuracy</td><td>67.0</td><td>66.0</td><td>72.5</td><td>48.6</td><td>71.8</td><td>70.9</td><td>70.3</td><td>71.9</td><td>71.5</td><td>72.1</td><td>70.8</td><td>71.1</td><td>72.7</td><td>70.7</td><td>71.0</td><td>73.0</td><td>70.5</td><td>67.5</td><td>70.4</td><td>69.5</td><td>73.6</td><td>75.3</td></tr></table>

classification, anomaly detection, as well as few-shot and zero-shot tasks, demonstrating remarkable capabilities. This gives it broad prospects in various real-world applications, such as energy and power forecasting with significant seasonal fluctuations, complex and variable weather forecasting, rapidly changing financial market predictions, and demand forecasting in supply chains, all of which it is highly applicable to. It can also excel in various anomaly detection scenarios commonly found in the industry. By leveraging the capabilities of TIMEMIXER++, we can effectively promote the development of various real-world applications related to time series analysis tasks.

Academic research As the pioneering study on a general time series pattern machine, we posit that TIMEMIXER++ holds significant potential to advance research in the domain of time series analysis. Our innovative approach involves converting time series data into time images and implementing hierarchical mixing across different scales and resolutions, which can provide substantial inspiration for future research endeavors in this field. Furthermore, it is noteworthy that our method of employing axial attention within the depth space to extract seasonality and trends from time images surpasses traditional shallow decomposition techniques, such as moving averages and FFT-based decomposition. This represents the first effective methodology for decomposing time series within the deep embedding, promising to catalyze further scholarly investigation.

Model Robustness The robustness of TIMEMIXER++ is evidenced by its performance across a diverse range of time series analysis tasks. In our extensive evaluation, TIMEMIXER++ was tested on 8 different types of tasks and over 30 well-known benchmarks, competing against 27 advanced baseline models. The results highlight its ability to consistently deliver high performance, demonstrating resilience to the variability and complexity inherent in time series data. This robustness is indicative of TIMEMIXER++'s capability to maintain accuracy and reliability across various scenarios, making it a versatile tool in the field of time series analysis. Its robust nature ensures that it can effectively handle noise and fluctuations within data, providing stable and dependable outcomes even in challenging conditions.

# K LIMITATIONS AND FUTURE WORK

TIMEMIXER++ consistently delivers state-of-the-art performance across a wide range of tasks, including long-term and short-term forecasting, classification, anomaly detection, as well as few-shot and zero-shot learning. This underscores its exceptional representational capacity and robust generalization as a time series pattern machine. However, it is important to acknowledge the recent shift in focus toward large time series language and foundation models, which emphasize continuous scaling of data and parameters, becoming a dominant paradigm in the field. In contrast, due to limitations in the quality and scale of available time series data, the parameter sizes of current advanced deep time series models remain relatively modest. Addressing the challenge of applying scaling laws to time series pattern machines is therefore essential. In this study, we introduced an effective backbone model as a first step toward building a universal time series pattern machine. Future research will focus on constructing large-scale time series datasets to further explore scaling laws for TIMEMIXER++, an exciting and promising direction for continued investigation.

![](images/ebb75601f33cb77c255049771c1057c915282046d9f7b13ebb7294486901df59.jpg)



(a)TimeMixer++


![](images/a65cb7e4eb3f2bfc2926eb6bf5d48c0c0f4f8c8b66aa24ae219ae2603858c9d0.jpg)



(b) iTransformer


![](images/596aa4a1e0d84375865972585ea825824b1521b3e6ee15370fba5fe832a6d8e9.jpg)



(c) PatchTST


![](images/60a85c90d5ead6cd798f44150118bfe9f25a97503d2a9a3b182012996a77e8c2.jpg)



(d) Crossformer


![](images/71594bb51fe207afc4860f127f79bb4430d0f7c70a2c50aa81b39c2029a01756.jpg)



(e) TiDE


![](images/198d7cfd8547addfb8147680d66fbabb884a0eb2fcfbeb2a25ef402ecbe4a595.jpg)



(f) Stationary


![](images/646290bb56d8c4aede9b519dcb86ea74cf15521533700e868cfdf311f307d189.jpg)



(g) Autoformer


![](images/ae8c2ae62100b06394638139b74a168bf46f51b8a85586cf124955361b3d2bf3.jpg)



(h) DLinear


![](images/678aa198572eaec301d7fa2e3b3492c651858c7862f562e8cc33300114500cad.jpg)



(i) FEDformer



Figure 16: Prediction cases from ETTh1 by different models under the input-96-predict-96 settings. Blue lines are the ground truths and orange lines are the model predictions.


![](images/4b8c0c5aee33f598d6ab3cc76fb9ef6368fe027493f4991f557bb40490a582b0.jpg)


![](images/e66612fedf414447f08729c15178f0d426766d01c46df1a652d9da844c7c301e.jpg)


![](images/cbeb3171628cc14b3f9705d462dae5d6e0a45e0b89b37809e32f31333dba1512.jpg)


![](images/5ced92f71e45e6567b24cd744f6e5225719e0af09eab6e73a0aafb7f4837ce7c.jpg)


![](images/71c09e2e84d862fb0343ce7782c9da295be66c746f988848a673601c2f47db23.jpg)


![](images/49fbfb19a12eefb662078d10a883af01187f9fd28810986e7890420272bed170.jpg)


![](images/aeb9a5020338554716db5769bfdf82a9673b160efe546b16495c7ed1e53bf2bf.jpg)


![](images/23a3c9951def002bde3d26f3d2b0c4cd8556fce83a728e6f048311ac2a1e5ddc.jpg)


![](images/0d762e444d482d7453301596f9a6196e405988dcdb3d0bb2f8ec152115c89032.jpg)



Figure 17: Prediction cases from Electricity by different models under input-96-predict-96 settings.


![](images/39566625de31703b3d61cd399cef5729145e981bccb2ec3bc82ba99ce495ae28.jpg)



(a)TimeMixer  $+ +$


![](images/f8f9c2e64541c9ce8783d3fef5c7a093e6b51fef7aaffc790d38bddebe596e13.jpg)



(b) iTransformer


![](images/f4d84d2de3dff8e2d7db1473b71195a64eaaacb6b2df7f9179a47ec6fe9ad297.jpg)



(c) PatchTST


![](images/b5b459a0177ad5e9745e48cf50c47ba80f25da285d7eca7b75518e1c769b3757.jpg)



(d) Crossformer


![](images/b5f573fd9824a16a8af792f81525ea25feeaadb5ace2cb6884975299b84744d7.jpg)



(e) TiDE


![](images/ec8729ea2c85cdc3990596d67dfab1ae6e471450423b7c0c43f9bcc4de444c85.jpg)



(f) Stationary


![](images/2f871b04719819882ebbfe8679929b23f0d991feb6f9f3407eb3677e2d65aaaf.jpg)



(g) Autoformer


![](images/382f781327b5833c7c285713b2bc6316e389311d4271fe19f8e5ab4125dd970f.jpg)



(h) DLinear


![](images/16a80fed67740bd82fe80b44b8ab0fe849674bc3dd7b7e034fe4799b5199f1ac.jpg)



(i) FEDformer



Figure 18: Prediction cases from Traffic by different models under the input-96-predict-96 settings.


![](images/541bd7137d8901c7dec96832e7e37679d9b7547863d58c24ba0cc58bf37bb332.jpg)



(a)TimeMixer++


![](images/47b1e543e32d8b202f37189fdf412badef39fff5fc6b0e16a6aad045ec29d456.jpg)



(b) iTransformer


![](images/69941fecac9f92ec256e1bb014d0a958d625cf4eefffb07074d598cc932cb2ea.jpg)



(c) PatchTST


![](images/6939d0a547b7a582a959fafbbf2b4721fbec513f2e4e00f295e24b9d5913db14.jpg)



(d) Crossformer


![](images/64cb2085fc7d3b07868eccfc3d825c78003516993f8cac812dea915946fdefa9.jpg)



(e) TiDE


![](images/b52a9cc035c231822f463c97706530651fa3eaf74cc6ebb932ac929678d3b9f5.jpg)



(f) Stationary


![](images/59d67442a73fab721c9718819e00f4844bf0bc1bb69cd1fc4bb115436e494b9f.jpg)



(g) Autoformer


![](images/c3c82963609210d4f01d2594f1a30e6bee26a1ac73078585c664675dee193478.jpg)



(h) DLinear


![](images/95827c5e3d1c1c8edf0f31196f6d4e86c22acfc9fd2c7c8b3c7e7dbc0fc082aa.jpg)



(i) FEDformer



Figure 19: Prediction cases from Weather by different models under the input-96-predict-96 settings.


![](images/b4997c024aaf54526de477179abb96fba1d204abd5dd73b325493fa1f3e4e543.jpg)


![](images/85e75ced5e4d3c224f4d32c73c9b8072c920386c50de40b62feb3c34915b9e68.jpg)


![](images/9bdb939cb8a8bf2ca2fc12197dfc20f459e8a29a7cbc01f57f5605707a130ff2.jpg)


![](images/548afa4b93e612dac22f1a0986ab4167859b57453fc93ecf2e2e7e348f64db3a.jpg)


![](images/c99daa24ce8c110765396a66e9d9bea1071e80fb5b17bfa725f655df520b4060.jpg)


![](images/263a69298be3016e2c22b73f9eab1c5408d0a66d51b493478ccf329702b8976a.jpg)


![](images/a544a28e5121622b54206d3a3ca99d3d8f35fbd02f38f79335b6b2a56e33c5f1.jpg)


![](images/e6c3963e981dc5c3b3821f70c8fb0f73ab43a3ee82154996aa8ab6d599996bf9.jpg)


![](images/de391f0eded8532c2b0d2fdf00168b47207f1af24c610ca2d73f57a28c56e2e5.jpg)



Figure 20: Showcases from Solar-Energy by different models under the input-96-predict-96 settings.


![](images/60dca6ebbe60759c2800e210a03588aaf7bb66df9ff5e265a3543e21aa33d56c.jpg)



(a) TimeMixer++


![](images/2fa205859b99a22d12b29bd9b29240cfbdfee0f0941913719c2bac31cf989ce0.jpg)



(b) iTransformer


![](images/56371a842202f56d5532d11eee3b34882789a66d9eb56d83cfffeadd352f202b.jpg)



(c) PatchTST


![](images/62a0cc41255cfba419c86ee1455fc15d98ac3bb0fbd3d914c693062910b61166.jpg)



(d) TimesNet


![](images/e42ed3e4a74886d0a22843fb4d9ab9a7a123f04de42e88b6039df1bd99ee777c.jpg)



(e) TiDE


![](images/791dc380d45dbf0332b0538462b9ea42763fcf1844e66c958a16d2f834cc3f60.jpg)



(f) Stationary


![](images/5c4679f92e14573653e75e1edd522b5dc3e6173ae279d912d8ae9134c4497f49.jpg)



(g) Autoformer


![](images/0c79b389b043584716206e7464da05647eda177a786a0c6914934b224392cd53.jpg)



(h) DLinear


![](images/af05b4cbd0e2cd63af25626e7fae8c597c3215b3db3fbad56d934cd30afa19c0.jpg)



(i) FEDformer



Figure 21: Showcases from the M4 dataset by different models under the input-36-predict-18 settings.


![](images/ab13104756e836f090bff656d21fb7348f3cc54fa3de424ed00b928351c6e400.jpg)



(a) TimeMixer++


![](images/dfe062aef042a0e0885af33a99f5cc1647619822f827cf06e89a3725eecc594b.jpg)



(b) iTransformer


![](images/185567290ed3d9f88b48668093a215eaf1cb664c8fc1e584498fcf91e8d0dbe6.jpg)



(c) PatchTST


![](images/4da99588e48e3948c6c2946e5da480569cd5a56cf7b5f060d36365f7ff8564c2.jpg)



(d) Crossformer


![](images/0d1883de11dc0e07f7c7373c0c8c556ece024e0bf13232070a45242756145086.jpg)



(e) TiDE


![](images/e63c9ba907af497142158bace5e7a8215874b058808311b13cf58ba4ad17ee13.jpg)



(f) Stationary


![](images/db91e6d6e218540ef47bc975913ccebeedad7f341e5d345bb7a74e287431314d.jpg)



(g) Autoformer


![](images/8ee33a77dfaa67ca07c583646dacca07cf5a04b73e2fe02250bb7640197d2764.jpg)



(h) DLinear


![](images/0b602318df5e1a10911c9e71a73a90e80fff2dea7de8c76cc8af5d1e3b812047.jpg)



(i) FEDformer



Figure 22: Showcases from PEMS03 by different models under the input-96-predict-12 settings.


![](images/1bec1a1c48e94f2cae3765f58fc81d3e067549feb7160335f5e215d5cfae20c5.jpg)



(a)TimeMixer++


![](images/9b78b9c2209dfb4e85f88332b3aef8009c2b3759e0a815cc9b3ecb1c5cea10f8.jpg)



(b) iTransformer


![](images/385136cee93d51d0e84a995802e2c315b95e4eab04a7de571c082f65bc1fa3e2.jpg)



(c) PatchTST


![](images/5e57c5705fb883c741d30c0f2d5ec7a45e6ddc4772433c8dbb10d1dfdf57e085.jpg)



(d) Crossformer


![](images/3457e09140aff9ed42aea0aaa4d8485dffb973f622edb39776e81deb7d971a8a.jpg)



(e) TiDE


![](images/78eddcfa67ffd878cafdf32a038fd6607746be716beb3b8e766603dab16a8acd.jpg)



(f) Stationary


![](images/2b77f18dc0bca2c87e096ca9f7a0b14e4dd98f0a78f75d1641fe339f0db4c275.jpg)



(g) Autoformer


![](images/b692b7a12679a1dec3e3f79631f9981cd20717d8a07f75f7aef7452a55fd711d.jpg)



(h) DLinear


![](images/afb5f62d60acb2464039013dec865f5d24586a06888aaaa19b05325c2fd398d4.jpg)



(i) FEDformer



Figure 23: Showcases from PEMS04 by different models under the input-96-predict-12 settings.


![](images/1fd1c13ee21289af3fb4d9431370759dc24faebe37bfd15c1ae42e0f2e57ccd2.jpg)



(a) TimeMixer++


![](images/fcc8a71fb2b212be23389265b78aa8ce7b66b734c748917c2f24d00458bfd213.jpg)



(b) iTransformer


![](images/5240d82215e8a9085bff9080440b8ce936f66fe7ab6e44b31591ccfb9ad9cbd4.jpg)



(c) PatchTST


![](images/0bef476387900c7ffa0013d9da1e1be95c5af88488c5b75eea7e553a95059fd7.jpg)



(d) Crossformer


![](images/1f9ef1e40e70b7db7ab3046d7e5d6b160589f68ac920d2ec07499e3865511c26.jpg)



(e) TiDE


![](images/01e23a95f9af15b4c602e6dbd7e6aa58310b14d0580f5c3453e7e8ace3381a56.jpg)



(f) Stationary


![](images/ca92493619886789fa1e118e2bf1c09a6deadc3f21370d0147e7b2e4abbeb62b.jpg)



(g) Autoformer


![](images/998c7148a16a0ddadb71f85785eada5a24100b4b4914722505cb75997071cf32.jpg)



(h) DLinear


![](images/f07cf915e5f8803bb95584360afc8281f8216b394477e272b19d1f14aee39368.jpg)



(i) FEDformer



Figure 24: Showcases from PEMS07 by different models under the input-96-predict-12 settings.


![](images/e90a7a60768c87db79d1370d5b39574a7cf1162fd5693207fd774d7defd10071.jpg)



(a)TimeMixer  $+ +$


![](images/843917dd0d85f72621afda513731173d367a09aaed98e5426288e880cf962f03.jpg)



(b) iTransformer


![](images/a89a025bc2a0d93598e880faa6cd6a15f9c81ab151e406937dab36ac172372c0.jpg)



(c) PatchTST


![](images/dc0f326d3facdc492125c36eee46f2944b9c4c289baf07190f0ec7821b617dce.jpg)



(d) Crossformer


![](images/f2383cb03b6880047f7676684922cb429a4d2af5a24e5211cf9c1a44f9ec8013.jpg)



(e) TiDE


![](images/49d462a162e522e5c62b6a5d7fe9a39f7e899976ed4ffa7997dec5ef338c181d.jpg)



(f) Stationary


![](images/200da10f53ce48a9966167e36ba522aac2477cedfb7b1b81d923cdad2c36b590.jpg)



(g) Autoformer


![](images/1b4050a1dbe43790fe3c788ec6d36cee8b6537e9d5989e09a5bd89a6ed0ebb75.jpg)



(h) DLinear


![](images/4d1c81ddf24739597f87f4d1c9b649c130179909901f50b920ffb826f92ff1c7.jpg)



(i) FEDformer



Figure 25: Showcases from PEMS08 by different models under the input-96-predict-12 settings.
