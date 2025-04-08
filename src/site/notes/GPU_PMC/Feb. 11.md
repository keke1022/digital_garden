---
{"dg-publish":true,"permalink":"/GPU_PMC/Feb. 11/"}
---

## Overview
利用FLOPs数据估计GPT2模型的参数总数
## Important Metrics
NVIDIA Compute: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#metric-comparison
## Important Kernel Function
`fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::Sm80, 1, 64, 64, 64, 1, 1>::Params)`

## Kernel Name解读
fhma: fused multi head attention
aligned_64x64：暗示该 kernel 针对 64×64 大小的数据块做了对齐优化，可能与线程块（block）或 tile 的大小有关。

```c
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_64x64_rf_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
```

## FLOPs Table

| flops(dp)  | seqlen | flops(fp)   | Infer($B×H×(2×L^2×d_h​+3×L^2)$) | $B×H×4×L^2×d_h$ |
| ---------- | ------ | ----------- | ------------------------------- | --------------- |
| 8650768    | 64     | 643694621   | 6438912                         | 12582912        |
| 25952264   | 128    | 1873521678  | 25755648                        | 50331648        |
| 86507544   | 256    | 6091597777  | 103022592                       | 201326592       |
| 311427031  | 512    | 21560524758 | 412090368                       | 805306368       |
| 1176502234 | 1024   | 80625502255 | 1648361472                      | 3221225472      |

| fma flops (fp) | fma flops (dp) | seqlen | $B×H×2×L^2×d_h$ |
| -------------- | -------------- | ------ | --------------- |
| 37748744       | 3932164        | 64     | 6291456         |
| 100663295      | 11796486       | 128    | 25165824        |
| 301989882      | 39321616       | 256    | 100663296       |
| 1006632919     |                | 512    | 402653184       |

| flops(dp) | n_layer | flops(fp)  | Infer($B×H×(2×L^2×d_h​+3×L^2)$) |
| --------- | ------- | ---------- | ------------------------------- |
| 25952253  | 6       | 1873518609 | 25755648                        |
| 25952235  | 12      | 1873516571 | 25755648                        |
| 25952253  | 96      | 1873526810 | 25755648                        |

| flops(dp) | hidden_dim | flops(fp)  | Infer($B×H×(2×L^2×d_h​+3×L^2)$) |
| --------- | ---------- | ---------- | ------------------------------- |
| 25952290  | 192        | 1647049723 | 6881280                         |
| 25952265  | 384        | 1647031317 | 13172736                        |
| 25952235  | 768        | 1873502200 | 25755648                        |
| 34603020  | 1536       | 3289645067 | 50921472                        |
| 34603004  | 3072       | 5277745136 | 101253120                       |
| 34603067  | 6144       | 9732096043 | 201916416                       |
guess: attention计算的时候主要使用了类似(a\*b+c)之类的运算，就被当作一次dp
## Parameters
- Embedding Layers
	- Token Embedding: $(\text{vocab\_size},\, n_{\text{embd}})$
	- Positional Embedding: $(n_{positions}, n_{embd})$
- Transformer Block $(n_{layer})$
	- Layer Normalization: 
		- 通常有两个：一个位于 Self-Attention 之前（记作 $ln_1$），一个位于 MLP之前（记作 $ln_2$）。  
		- 每个 LayerNorm 通常有两个参数：
			- $\gamma$：形状为 $(n_{\text{embd}},)$
			- $\beta$：形状为 $(n_{\text{embd}},)$
	- Self-Attention:
		- 在 GPT2 中，自注意力部分通常将 Query、Key 和 Value 的线性变换合并为一个操作，记作 c_attn，再经过一个输出投影层 c_proj。
		- $c_{attn}: (n_{embd}​,3\times n_{embd}​) + (3\times n_{embd})$
		- $c_{proj}: (n_{embd}​,\times n_{embd}​) + (\times n_{embd})$
	- Feed-Forward Network
		- c_fc: $(n_{embd}​,4\times n_{embd}​)$
		- c_proj: $(4\times n_{embd}​,n_{embd}​)$
	- 在所有 Transformer Block 之后，还会加一个最终的 LayerNorm，其参数同样为$(n_{\text{embd}},)$

---

Possible Parameter: Group Query Attention (save KV cache)

Double Check: 断崖上涨、实际值比理论值少

reliability of the detection? How to bypass some of the metrics? 

为什么云不可以修改report？security key in NVIDIA

Benefits: lower overheads

## Paper
Verification Related work

还原目标确立
主体：找到那些kernel, 可以判断模型哪些内容

或者先找到PCA影响最大的，加上理论分析

计算量（指令数）/内存/时间

tensor core/cuda core：可能是`sm__inst_executed_pipe_fp64_op_dmma.avg.pct_of_peak_sustained_active`

对模型做hash可不可以拿到模型的数据？文件怎么读给GPU？好处可能是模型的初始状态。

GPU TE是什么。。。

内存怎么看：throughput, hit rate。。。就看那个kernel