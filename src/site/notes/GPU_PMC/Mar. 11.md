---
{"dg-publish":true,"permalink":"/GPU_PMC/Mar. 11/"}
---


| N    | PMC flops    | Expected flops ($2\cdot N^3$) |
| ---- | ------------ | ----------------------------- |
| 1024 | 1.099512e+12 | 2147483648                    |
| 512  | 1.374390e+11 | 268435456                     |
| 256  | 1.717987e+10 | 33554432                      |
| 64   | 2.684355e+08 | 524288                        |
### Formal Method
- GPU固有的硬件能力(内存大小，cache 结构，SM数量)
- 静态指标: 例如gst_transactions(代表全局存储请求的次数, 预测值$B \times L \times H \times d_h \times 4/32$)这种基于硬件设计参数计算得到的数据，视为固定参数，直接在形式化模型中作为已知的常量使用
- 动态指标: 划定置信区间
	- 如果某些指标（如cache hit rate）波动较大, 尝试建模/丢弃
$□((M≤M_{upper}​∧C∈C_{range}​∧T≤T_{upper}​)→GPU处于xxx状态)$

- HW
- Model Setting: Precision, Model size, etc. 
- PMC
- Efficiency: model's training/inference stage. Where the model is?
=> No counter examples. 

可能的尝试
Loss function calls kernel function A
1000次A
500次A >= 1h

例如Deepseek可以bypass efficiency

--- 

偷懒: 小模型, sparsity
但是都会影响performance
只看size不能反映模型是不是跳过了一些参数

动态runtime检测: 
- 为什么一定要做动态? 为什么不来load model的时候就做检测? 比如看model size
- 比如运行一个小时, 这整个时间都要用某个model
	- sampling?

Grid size造假:
- launch了16个, 但是只用到了8个

Next Step
- 总结model: 提出三个model都可以work的方法