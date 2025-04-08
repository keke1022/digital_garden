---
{"dg-publish":true,"permalink":"/GPU_PMC/Apr. 1/"}
---

## Workflow
1. Identify PMC Metrics:  
   - Global Store Requests: `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum`  
   - Sectors Written: `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`

2. Benchmark Approach:  
   - Write two simple kernels in Python (using Numba):
     - **Coalesced kernel:** Each thread writes to consecutive memory (`arr[idx] = idx`).
     - **Uncoalesced kernel:** Each thread writes with a stride (`arr[idx * 2] = idx`).

3. Data Collection:
   - Run the benchmark and profile with Nsight Compute.

### Essential Code Snippet
```python
@cuda.jit
def coalesced_store_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = idx

@cuda.jit
def uncoalesced_store_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx * 2] = idx
```

### Observed Results
| Type               | Global Store Requests | Sectors Written |
| ------------------ | --------------------- | --------------- |
| Coalesced Kernel   | 256                   | 1024            |
| Uncoalesced Kernel | 256                   | 2048            |
Coalesced Kernel → Each warp (32 threads) writes 4 sectors (32B each).
Uncoalesced Kernel → Each warp writes 8 sectors.

### Llama Result
seqlen = 128, HeadDim = 4096

| Type       | Global Store Requests | Sectors Written | Expected Writes                               |
| ---------- | --------------------- | --------------- | --------------------------------------------- |
| llama_fp32 | 4096                  | 65536           | $\frac{128\times 4096\times 4B}{32B} = 65536$ |
| llama_bf16 | 4096                  | 32768           | $\frac{128\times 4096\times 2B}{32B} = 32768$ |
Note that for `llama_fp32`, the recorded kernel is the same as GPT2
For `llama_bf16`, the recorded kernel is `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
However, for kernel `void pytorch_flash::flash_fwd_kernel...`, values are: 524800, 33280

This might be illuminating for being a "workflow", which will be explained in "How to find Critical Kernel"
### Conclusion

let `memory sector=S`, `data size = D`, then 
$\frac{seqlen \times headDim \times D}{S} \leq \text{gst\_transactions} \leq seqlen \times headDim$

A good model should satisfy: 
$\text{gst\_transactions} = \frac{seqlen \times headDim \times D}{S}$

---

## How to find Critical Kernel

`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`is related to Tensor Core Calculations.

For GPT2, only one kernel is non-zero:
- `fmha_cutlassF_f32_aligned_64x64_rf_sm80`
However, for Llama_bf16, two metrics are non-zero:
- `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
- `void pytorch_flash::flash_fwd_kernel...`

How to find the correct "Critical Kernel"? 
- use `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` to determine the kernel that uses Tensor Core
- Use `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` and `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` to determine which kernel is doing Matrix Multiplication.
	- If "Sectors Written" (`l1tex__t_sectors`) is equal to $\frac{seq\_length\times n\_embd\times data\_size}{sector\_size}$

---

## Next Step
- Find out the source code for `pytorch_flash::flash_fwd_kernel`. 
	- Temporarily it is hard to find. No obvious record on Github
	- If no, find out what this kernel is doing
- Other possible Metrics
	- `sm__sass_thread_inst_executed_op_ffma_pred_on.avg.peak_sustained` and `sm__sass_thread_inst_executed_op_dfma_pred_on.avg.peak_sustained`, the ratio can reveal whether it is `FP32` or `FP64`
	- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
	- Compute/Memory: `sm__inst_executed.avg.per_cycle_active`and `dram__bytes.avg.per_second`. 
		- Smaller Model -> higher Compute/Memory. 
		- Because smaller model can put parameters inside L1/L2 caches. 

Tensor Core bypass? not use Tensor Core
- Efficiency drops

**🔹 判断数据类型（bf16, fp32, fp64）**

• sm__sass_thread_inst_executed_op_ffma_pred_on.avg.peak_sustained: 代表 fp32 FMA 指令

• sm__sass_thread_inst_executed_op_dfma_pred_on.avg.peak_sustained: 代表 fp64 FMA 指令

  

👉 比较它们的比例，可以粗略判断：

• 是否用了 fp32 / bf16

• 是否用了 Tensor Core（通常 ffma 非常少）

  

**🔹 加载行为 (Load)**

• l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum

• l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

  

👉 可以和 store 类似，推断出：

• 输入 activation / weight 是什么大小

• load 是否 coalesced，是否重复读

  

**🔹 Compute vs Memory**

• sm__inst_executed.avg.per_cycle_active: 算力活跃度

• dram__bytes.avg.per_second: 内存带宽占用

  

👉 观察 compute/mem ratio：

• 高表示可能为 **小模型或高 compute kernel**（参数/中间变量在 L1/L2）

• 低则可能是 **大模型/大量内存访问**