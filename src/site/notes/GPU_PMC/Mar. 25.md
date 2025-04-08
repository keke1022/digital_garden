---
{"dg-publish":true,"permalink":"/GPU_PMC/Mar. 25/"}
---

[[Paper/GPU/LLM Inference Unveiled\|LLM Inference Unveiled]]

## Another Metrics
`gst_transactions`, recorded as `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`

| seqlen | Collected | Expected ($seqlen \times headDim/8$) |
| ------ | --------- | ------------------------------------ |
| 64     | 6144      | 6144                                 |
| 128    | 12288     | 12288                                |
| 256    | 24576     | 24576                                |
| 512    | 49152     | 49152                                |
| 1024   | 98304     | 98304                                |

| headDim | Collected | Expected ($seqlen \times headDim/8$) |
| ------- | --------- | ------------------------------------ |
| 384     | 6144      | 6144                                 |
| 768     | 12288     | 12288                                |
| 1536    | 24576     | 24576                                |
| 3072    | 49152     | 49152                                |
| 6144    | 98304     | 98304                                |
- 最终写入全局内存的数据构成了一个二维张量，其尺寸为$\text{[seqlen, headDim]}$
- 存储次数除以8: 一个float32数据大小4 Byte, 一次最多写入32B, 最多八个元素.

在源代码写回全局:
```python
auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
  using OutputTileIterator = typename MM1::OutputTileIterator;
  return OutputTileIterator(
	  typename OutputTileIterator::Params{(int32_t)p.o_strideM},
	  p.output_ptr,
	  typename OutputTileIterator::TensorCoord{
		  p.num_queries, p.head_dim_value},
	  thread_id(),
	  {0, col});
};
...

auto dest_iter = createOutputIter(col);
EpilogueOutputOp rescale(s_prime, out_rescale);
Epilogue epilogue(
    shared_storage.epilogue_shared_storage(),
    thread_id(),
    my_warp_id,
    my_lane_id);
epilogue(rescale, dest_iter, accum_o, source_iter);
```

作为上下限估计:
- 写回元素个数有最大上限
	- 最多八个
- 至少写回一个

let `memory sector=S`, `data size = D`, then 
$\frac{seqlen \times headDim \times D}{S} \leq \text{gst\_transactions} \leq seqlen \times headDim$

A good model should satisfy: 
$\text{gst\_transactions} = \frac{seqlen \times headDim \times D}{S}$

---

### Microbench
写一套基准测试?
Microbench 说明最多一次写回128 bits.

---

### Next Step
- 尝试检测模型量化压缩(可能)
	- 理论$\text{gst\_transactions} \approx \frac{seqlen \times headDim \times D}{S}$

---

### Regulation
Define efficiency
Deepseek 
Use Llama PMC data
Check 128bits and a sector size 32B
155T data check
### Recover Parameters
