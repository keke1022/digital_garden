---
{"dg-publish":true,"permalink":"/GPU_PMC/Apr.15/"}
---

# Potential Metrics
- `dram__sectors_read.sum`
	- `n_layer` unrelated.
	- `hidden_dim` and `seq_len` linear
	- Expected Calculation for "fmha": $3\times seqlen\times n\_embed\times data\_size/sector$. 
		- Relatively Close
		- 128: 36864

| **seq_length_size** | 64     | 128    | 256    | 512    | 1024    |
| ------------------- | ------ | ------ | ------ | ------ | ------- |
| fmha                | 17652  | 33020  | 63765  | 125224 | 248119  |
| gemm                | 326518 | 337303 | 358542 | 696757 | 1176270 |
| layer_norm          | 4111   | 7179   | 13325  | 25617  | 50191   |
- `sm__inst_executed.sum.per_cycle_active` and `sm__cycles_active.sum`
	- The density of instructions
		- Observation is that for "gemm" computes, the values are larger than normal.
		- Could be a way to identify gemms
- `smsp__sass_inst_executed_op_global_st.sum`
	- `n_layer` unrelated.
	- `seq_len` linear
	- Partially linear for `hidden_dim`

| **seq_length_size** | 64  | 128 | 256  | 512  | 1024 |
| ------------------- | --- | --- | ---- | ---- | ---- |
| fmha                | 198 | 390 | 774  | 1542 | 3078 |
| layer_norm          | 260 | 516 | 1028 | 2052 | 4100 |

| **Hidden_dim** | 192 | 384 | 768 | 1536 | 3072 | 6144 |
| -------------- | --- | --- | --- | ---- | ---- | ---- |
| fmha           | 390 | 390 | 390 | 780  | 1584 | 3168 |
| layer_norm     | 258 | 322 | 516 | 903  | 1677 | 3225 |

# Problems
- Kernel functions with similar names behave completely different
	- `fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, arch::Sm80, 1, 64, 64, 64, 1, 1>::Params)` faster than `fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::Sm80, 1, 64, 64, 64, 1, 1>::Params)`

- Dramatic drop for metrics

| **seq_length_size** | 64    | 128   | 256   | 512   | 1024   |
| ------------------- | ----- | ----- | ----- | ----- | ------ |
| fmha                | 0.56  | 1.05  | 2.04  | 4.0   | 0.0079 |
| gemm                | 10.44 | 10.79 | 11.47 | 22.29 | 0.037  |
| layer_norm          | 0.13  | 0.229 | 0.426 | 0.819 | 0.0016 |

# Metrics Classification
## Hidden Size
- sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active
- sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active
- sm__inst_executed.sum.pct_of_peak_sustained_elapsed
- sm__issue_active.sum.pct_of_peak_sustained_elapsed
## Sequence Length
- dram__bytes_read.sum
	    `fmha`Doubles
- dram__bytes_write.sum

# Possible Next Step
- Continue finding bounds for hardwares.
- Try `GPGPU`/`Accel-Sim`? One needs PTX, and another needs SASS. (We can get SASS code)
- Write Paper to have some ideas.
- Test on other LLM open source training code.

# Observations
For GPT2 model:
- Only one with `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` non-zero
	- `fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::Sm80, 1, 64, 64, 64, 1, 1>::Params)`

For Llama:
- Six with `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` non-zero
	- `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
	- `void pytorch_flash::flash_fwd_kernel<pytorch_flash::Flash_fwd_kernel_traits<128, 64, 64, 4, 0, 0, cutlass::bfloat16_t, pytorch_flash::Flash_kernel_traits<128, 64, 64, 4, cutlass::bfloat16_t>>, 0, 1, 0, 0, 1, 1, 0>(pytorch_flash::Flash_fwd_params) `
	- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params) `
	- `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params) `
	- `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::Flash_fwd_kernel_traits<128, 64, 128, 4, 0, 0, cutlass::bfloat16_t, pytorch_flash::Flash_kernel_traits<128, 64, 128, 4, cutlass::bfloat16_t>>, 0, 0, 0, 0, 1, 1, 0>(pytorch_flash::Flash_fwd_params) `
	- `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)`