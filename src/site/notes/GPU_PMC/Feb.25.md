---
{"dg-publish":true,"permalink":"/GPU_PMC/Feb.25/"}
---

### Overview
### dram_read_transactions建模

| seqlen | dram_read_transactions | Estimated($9\times B \times L \times H \times 4/128$) | 1.4x Esti. |
| ------ | ---------------------- | ----------------------------------------------------- | ---------- |
| 64     | 20556                  | 13824                                                 | 19353.6    |
| 128    | 39012                  | 27648                                                 | 38707.2    |
| 256    | 75912                  | 55296                                                 | 77414.4    |
| 512    | 149692                 | 110592                                                | 154828.8   |
| 1024   | 297132                 | 221184                                                | 309657.6   |

| hidden_dim | dram_read_transactions | Estimated($9\times B \times L \times H \times 4/128$) | 1.4x Esti. |
| ---------- | ---------------------- | ----------------------------------------------------- | ---------- |
| 192        | 11372                  | 6912                                                  | 9676.8     |
| 384        | 20588                  | 13824                                                 | 19353.6    |
| 768        | 39004                  | 27648                                                 | 38707.2    |
| 1536       | 75768                  | 55296                                                 | 77414.4    |
| 3072       | 149784                 | 110592                                                | 154828.8   |
| 6144       | 297228                 | 221184                                                | 309657.6   |
- n_layer 无关
- 拟合公式:$1.4\times 9\times B \times L \times H \times 4/128$
- L: seq_len; H: hidden_dim

- Story for the BatchSize

### inst_executed_shared_loads
shared memory加载数据到寄存器的指令执行次数

| seqlen | inst_executed_shared_loads |
| ------ | -------------------------- |
| 64     | 384                        |
| 128    | 768                        |
| 256    | 1536                       |
| 512    | 3072                       |
| 1024   | 6144                       |

| hidden_dim | inst_executed_shared_loads |
| ---------- | -------------------------- |
| 192        | 768                        |
| 384        | 768                        |
| 768        | 768                        |
| 1536       | 1536                       |
| 3072       | 3072                       |
| 6144       | 6144                       

- 与n_layer无关
- 每个 token 大致固定触发 6 次共享内存加载指令
- 当 p.head_dim 小于或等于 768 时，gemm_k_iterations(就是hidden_dim 的 tile 数) 可能被固定为一个常数（例如 4），因此总的事务数保持 768。
- dram_read_transactions 在hidden_dim超过一定阈值（此处为 768）后才会线性增长(这里函数名称也不同)；当 hidden_dim 增大时，需要更多 tile 来完成矩阵乘法，导致全局内存的读取事务数随之增加。