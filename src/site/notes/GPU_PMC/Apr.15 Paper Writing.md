---
{"dg-publish":true,"permalink":"/GPU_PMC/Apr.15 Paper Writing/"}
---

# Section 4: Implementation

We need to define a "scope" for the paper. 
- Can we assume that the kernel names cannot be attacked? Can we assume that cloud-servers use the same public GPT2/Llama model as we do?
	- if no, how to find the KQV calculation kernel? What if the kernel is split finely?

About "Optimizations"
- A cloud server can use a lot of Caches to reduce the compute and memory, which will trigger warnings in our system.
- We assume that the cloud server should avoid optimizations, because customers buy the compute services.

About Custom Rewriting
- We need to look at PMC. However, if cloud-server uses a completely different CUDA code to simulate some PyTorch functions in order to bypass key functions, we cannot collect the right PMC.