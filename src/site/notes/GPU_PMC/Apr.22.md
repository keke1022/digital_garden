---
{"dg-publish":true,"permalink":"/GPU_PMC/Apr.22/"}
---

# Overview
## Try New Ideas
### What is the Address Unit (ADU)?
The ADU is the hardware block that computes memory addresses for each load or store instruction.
In a tile‑based kernel the ADU is invoked only when loading a new tile.
### Find patterns
![](https://raw.githubusercontent.com/keke1022/picgo/main/pic/202504220351400.png)

The tall peaks correspond to tiled matrix multiplications:
- QKV projection
- Wo projection
- FFN layers
In those stages each block of data is loaded once into registers or shared memory and then reused for many FMA instructions, so the ratio of FMA cycles to address‑unit cycles is very large. 

The low valleys mark element‑wise phases
- Add & Norm
- activation
every arithmetic step reads or writes data and invokes the address unit. 

### Why Matrix Multiplication has higher ratio?
If you remove tiling and reload operands for each multiply, the address unit must run on every FMA, and the FMA/ADU ratio falls toward one.

## Trials about FLOPs

---

For a normal LLM inference pattern, confirm that this is true.

- CUDA Graph
- Difference between "Global Storer Metrics"
- PMC collect values using time intervals
- Fused Kernel? 