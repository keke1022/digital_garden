---
{"dg-publish":true,"permalink":"/GPU_PMC/Feb. 18/"}
---

### Overview
利用内存数据估计GPT2模型的参数总数
### Recall Flops Calculations

| flops(dp)  | seqlen | flops(fp)   | ($B×H×(2×L^2×d_h​+3×L^2)$) | $B×H×4×L^2×d_h$ |
| ---------- | ------ | ----------- | -------------------------- | --------------- |
| 8650768    | 64     | 643694621   | 6438912                    |                 |
| 25952264   | 128    | 1873521678  | 25755648                   |                 |
| 86507544   | 256    | 6091597777  | 103022592                  |                 |
| 311427031  | 512    | 21560524758 | 412090368                  |                 |
| 1176502234 | 1024   | 80625502255 | 1648361472                 |                 |
#### Estimation Calculations
$QK^T$(both in the shape of $[B, H, L, d_h])$: $B × H × L × L × (2×d_h)$
$Attn\times v$(v is of the shape $[B, H, L, d_v]$): $B×H×L×d_v​×2L$

### dram_read_transactions建模

| seqlen | dram_read_transactions | Estimated($9\times B \times L \times H \times 4/128$) | 1.4x Esti. |
| ------ | ---------------------- | ----------------------------------------------------- | ---------- |
| 64     | 20556                  | 13824                                                 | 19353.6    |
| 128    | 39012                  | 27648                                                 | 38707.2    |
| 256    | 75912                  | 55296                                                 | 77414.4    |
| 512    | 149692                 | 110592                                                | 154828.8   |
| 1024   | 297132                 | 221184                                                | 309657.6   |

B: Batch size, L: seq_len, H: hidden_dim (embed size)
只关注每个Transformer层中attention和前馈网络
- 对于attention模块，假设读取数据量为
    $\text{Bytes}_{\text{attn}} = \beta_1 \times (B \times L \times H \times 3) + \beta_2 \times (B \times L \times H)$
    其中第一项代表Q、K、V的输入激活（可能被重复加载多次）
    第二项代表后续的softmax、加权求和
    $\beta$是放大系数，反映由于缓存未命中和额外开销
- 对于前馈模块，数据读取量可以估计为
    $\text{Bytes}_{\text{ff}} = \beta_3 \times (B \times L \times H) + \beta_4 \times (B \times L \times 4H)$
    这里4H是前馈层常见的扩展尺寸
- 整个模型的总字节读取为每层各模块字节之和，再乘以层数 N。再将总字节数除以事务大小 T（例如128字节），再向上取整，即得到理论上最少的事务数：
	$\text{Transactions} = N \times \left\lceil \frac{\text{Bytes}_{\text{attn}} + \text{Bytes}_{\text{ff}}}{T} \right\rceil.$


### gst_transactions建模
Definition: 全局存储（写入）请求的次数

| seqlen | gst_transactions | Estimated($B \times L \times H \times d_h \times 4/32$) |
| ------ | ---------------- | ------------------------------------------------------- |
| 64     | 6144             | 6144                                                    |
| 128    | 12288            | 12288                                                   |
| 256    | 24576            | 24576                                                   |
| 512    | 49152            | 49152                                                   |
| 1024   | 98304            | 98304                                                   |

| hidden layers | gst_transactions | Estimated($B \times L \times H \times d_h \times 4/32$) |
| ------------- | ---------------- | ------------------------------------------------------- |
| 192           | 3072             | 3072                                                    |
| 384           | 6144             | 6144                                                    |
| 768           | 12288            | 12288                                                   |
| 1526          | 24576            | 24576                                                   |
| 3072          | 49152            | 49152                                                   |
- 这里假设全局写入事务的有效大小是32字节，和一个sector大小一样
- gst_transactions与layers无关
- Estimation Reason: 输出张量的形状为$[B, H, L, d_v]$

---

### Note
smsp__sass_inst_executed_op_shared_ld.sum 可能比较重要