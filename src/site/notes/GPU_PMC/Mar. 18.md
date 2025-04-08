---
{"dg-publish":true,"permalink":"/GPU_PMC/Mar. 18/"}
---

### 现有参数
gst_transactions = $B \times L \times H \times d_h \times 4/32$
Reasoning: 输出张量的形状为$[B, H, L, d_v]$, 全局写入事务的有效大小是32字节，和一个sector大小一样, 一个数据大小是4 Byte

如果能说明1.4x的原因, 普适性?
kernel name可不可信?
- 如果不可信, 怎么先找到attention kernel? 什么信号认为他在执行attention?
	- 看那些PMC? 怎么还原参数?
	- 可能比例不是1.4, 检测的流程是根据什么确定这个比例?
	- 别的模型会不会有小的修改? variance?

gemma, qwen, 
deepseek(MLA技术)

理论是必要的. 
分成大块: 没有attention kernel的话, 应该是很多小kernel组成的, 找到这些kernel

可不可以ebpf不用顺行的方式采取pmc数据