---
{"dg-publish":true,"permalink":"/GPU_PMC/Apr.29/"}
---

# Pattern Matching

For GPT with `seq_len=64`
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241408389.png)
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241408216.png)
For GPT with `seq_len=512`
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241409606.png)
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241410336.png)
For Llama fp32
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241416436.png)
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241416893.png)
For Llama bf16
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241531213.png)
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241532154.png)
	- ![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504241532951.png)

## How to Confirm the fma calculation

---

# Ideas
1. 指令组合与比例
	- 每个阶段的FMA与其他算术指令的精确比例
	- 多精度指令混合模式（如FP16/FP32/FP64的精确分布）
	- 特殊数学函数使用模式（Softmax中的exp操作、LayerNorm中的sqrt）
2. 内存访问模式
	- L1/L2缓存命中率与缓存行重用模式
	- 内存访问时序特征和stride patterns
	- 全局/共享内存读写比例和时间戳分布
	- TLB命中率和页面访问模式
3. 原子操作特征
	- 注意力计算中的并行规约操作产生的独特原子操作模式
	- 原子操作与普通内存访问的时序相关性
4. 波前（warp）执行特征
	- 波前分支散度模式（注意力计算与FFN的差异）
	- 每阶段波前执行效率和SIMD利用率的微小变化
5. 硬件单元利用率时域特征
	- 张量核心使用率的精确时域波动
	- SM占用率随时间变化的精确特征

---

小模型, 大batch size.

Story: Regulation/Cloud Service

为什么有的kernel一定存在
- 因为grid原因
- 基于时间: LLM一定有这个步骤, 找的是步骤的pattern
- 隐藏不了. 隐藏不如直接计算. 

Simulator or abstract

General:
- 任意给定一个ML算法,都可以用这个framework
- 算法到时间预测: simulator
- formal: search space是一些attack方式