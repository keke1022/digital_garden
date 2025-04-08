---
{"dg-publish":true,"permalink":"/GPU_PMC/Feb. 4/"}
---

## Problems

在`./data/csv/seq_len/report_seq_length_256_run_1.csv`中, 前几行数据:

| Kernel Name                          | l1tex__t_sector<br>_hit_rate.pct | l1tex__t_sector_pipe_lsu_<br>mem_global_op_ld_hit_rate<br>.pct |
| ------------------------------------ | -------------------------------- | -------------------------------------------------------------- |
| ampere_sgemm<br>_128x32_nn           | 0.313420                         | 0.360755                                                       |
| ampere_sgemm<br>_64x32_sliced1x4_nn  | 0.000000                         | 0.000000                                                       |
| ampere_sgemm<br>_128x32_sliced1x4_nn | 0.000000                         | 0.000000                                                       |
可能的原因: 数据直接复制到 shared memory ,避免经过L1的中转

kernel name能不能修改? 
可以, 但是需要修改二进制文件

## Potential Ways of Solving
![Screenshot 2025-02-03 at 12.49.28 AM.png](/img/user/GPU_PMC/img/Screenshot%202025-02-03%20at%2012.49.28%20AM.png)

- Use Invariants
	- 有的列有pattern. 
		- 相同kernel func, 列是一样的
		- load和store的比例在一定范围内
 - Microbenches
	 - 设计一个微内核
		 - 为了建立模型，我们可以固定数据布局和内存分配方式(A、B、C 均存放在全局内存中)
		 - 仅改变矩阵规模 N
	 - 收集PMC数据
		 - sm__inst_executed_pipe_fma.sum(FMA指令总数)理论应该接近$N^3$(内积次数)
		 - smsp__sass_inst_executed_op_global_ld.sum 和smsp__sass_inst_executed_op_global_st.sum 反映了全局内存加载和存储的次数，与矩阵$O(N^2)$有关。
	 - 可以利用这些数据做回归分析或曲线拟合验证

- 默认小页为 **4 KB**
- CUDA 运行时支持 **64 KB 和 2 MB** 页，适用于大规模内存管理

## GPU矩阵乘法流程
- 加载矩阵到Global Memory
	- 两个输入矩阵加载到Global Memory
- 将数据加载到Shared Memory
	- 多个Blocks来提供共享内存: tile \* tile
	- 假设tile size = block size, 每一个thread对应一个输出的元素
- 线程内部计算
- 存储计算结果到Global Memory
	- 结果矩阵会被写回到global memory
![Screenshot 2025-02-04 at 12.24.55 AM.png](/img/user/GPU_PMC/img/Screenshot%202025-02-04%20at%2012.24.55%20AM.png)
![Screenshot 2025-02-04 at 4.41.42 PM.png](/img/user/GPU_PMC/img/Screenshot%202025-02-04%20at%204.41.42%20PM.png)

- **L2 Cache**(RTX 4090 是 72MB)
	- 作用：所有Blocks共享L2 Cache，可以缓存Global Memory访问的数据
	- 影响：如果 `A` 和 `B` 没有完全加载到共享内存，下一次访问相同的数据块时，可以从 L2 Cache 读取，而不需要重新访问 Global Memory
	- Page Size常见大小 4 KB
	- **访问策略**：
	    - `Global Memory → L2 Cache → Shared Memory`
	    - 大多数全局内存的访问都会先经过 L2 Cache

- **L1 Cache(通常 48KB 或 128KB/SM)**
    - 作用：每个 SM都有自己的 L1 Cache，加速加载到 Shared Memory 之前的数据读取
    - L1 Cache通常128B
    - 影响：
        - 如果Global Memory访问是合并的(coalesced)，则 L1 Cache 会优化访问模式。
- 128SM, 4SP, 32 Cuda Core, 16 int FP32, Reg Size 64KB

--- 

## Plans
### 写文章
Introduction
对模型的还原, 对那些参数感兴趣, 为什么感兴趣 + 表格
### 最重要
是不是可以还原13B的参数