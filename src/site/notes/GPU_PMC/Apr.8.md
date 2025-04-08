---
{"dg-publish":true,"permalink":"/GPU_PMC/Apr.8/"}
---

## Data Type Benchmark

### Key Metrics
`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed`,
`smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed`

### Essential Code Snippet
```c
extern "C" __global__
void fma_kernel({dtype}* A, {dtype}* B, {dtype}* C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		{dtype} a = A[idx];
		{dtype} b = B[idx];
		{dtype} c = C[idx];
		C[idx] = a * b + c; // FMA
	}
}
```

### Observed Results
| Type | ffma      | dfma      |
| ---- | --------- | --------- |
| fp32 | 30.892630 | 0         |
| fp64 | 0         | 16.668483 |

### Current PMC

| Type                     | ffma | dfma | hfma |
| ------------------------ | ---- | ---- | ---- |
| llama_fp32               | 4373 | 0    |      |
| llama_bf16               | 0    | 0    |      |
| GPT2_fp32_seq_length_128 | 3883 | 0    |      |

### Conclusion

The data type can be added to the workflow. The size can be determined.

---

## Load Behavior Analysis Report

### Key Metrics
- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`

### Microbenchmark Setup

To investigate global load coalescing behavior, we implemented two simple GPU kernels using Numba:
- **Coalesced Load:** each thread reads a contiguous element
- **Uncoalesced Load:** each thread reads from a strided index
### Essential Code Snippet
```python
@cuda.jit
def coalesced_load_kernel(arr, out):
    idx = cuda.grid(1)
    if idx < arr.size:
        out[idx] = arr[idx]

@cuda.jit
def uncoalesced_load_kernel(arr, out):
    idx = cuda.grid(1)
    if idx * 2 < arr.size:
        out[idx] = arr[idx * 2]
```

| **Kernel Type** | **Requests** | **Sectors** | **Sectors / Request** |
| --------------- | ------------ | ----------- | --------------------- |
| Coalesced       | 256          | 1024        | 4.0                   |
| Uncoalesced     | 256          | 2048        | 8.0                   |

### Current GPT

| **Model**                | seqlen | headDim × n_heads | **Data Type** | **Measured Sectors** | **Expected**                                           |
| ------------------------ | ------ | ----------------- | ------------- | -------------------- | ------------------------------------------------------ |
| GPT2_fp32_seq_length_128 | 128    | 768               | fp32 (4B)     | 55,296               | $\frac{128 \times 768 \times 3 \times 4}{32} = 36,864$ |

### Conclusion
The measured global load behavior in the GPT2 attention kernel is **not fully coalesced**.
This suggests that the Q/K/V load operations in the attention kernel may involve:
- Unaligned memory access
- Interleaved memory layout
- Per-head or per-block striding

Therefore, **load metrics cannot directly determine shape**, and require an additional **coalescing factor** in the model. ($\text{Measured / Ideal} = \frac{55296}{36864} = 1.5$)

---

## Next Step
- [ ] Find out the source code for `pytorch_flash::flash_fwd_kernel`. 
	- Temporarily it is hard to find. No obvious record on Github
	- If no, find out what this kernel is doing
- Other possible Metrics
	- [x] `sm__sass_thread_inst_executed_op_ffma_pred_on.avg.peak_sustained` and `sm__sass_thread_inst_executed_op_dfma_pred_on.avg.peak_sustained`, the ratio can reveal whether it is `FP32` or `FP64`
	- [x] `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
	- [ ] Compute/Memory: `sm__inst_executed.avg.per_cycle_active`and `dram__bytes.avg.per_second`. 
		- Smaller Model -> higher Compute/Memory. 
		- Because smaller model can put parameters inside L1/L2 caches. 
		- Compare prefilling and decoding

- [ ] Tensor Core bypass? not use Tensor Core
	- Efficiency drops

smaller datatype benchmark
- look into pmc

GPGPU-Sim
- Useful for formal method