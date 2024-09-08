---
{"dg-publish":true,"permalink":"/Ryan/Invariant Log/"}
---

## 9-7
For Debug Purposes, add code in `mnist.py`
```python
##############################
#Additional test for func arg#
##############################

linear = nn.Linear(2, 2)
ll = linear.to(dtype=torch.float64)
print(ll)

##############################
#Additional test for func arg#
##############################
```
Output Trace for func `torch.nn.modules.module.Module.to`. The upper is the trace of `self` argument, and the later is the trace of the return value.
```json
{
	{"is_method": true, "var_type": "torch.nn.modules.linear.Linear", "var_name": "self", "func_name": "torch.nn.modules.module.Module.to", "self_stat": {"min": -0.3121015429496765, "max": 0.36434608697891235, "mean": 0.0617644302546978, "std": 0.3385463058948517, "shape": [2, 2]}, "time": 1725751387.739062}

	...

	{"process_id": 524689, "thread_id": 139850984167232, "var_type": "torch.nn.modules.linear.Linear", "func_name": "torch.nn.modules.module.Module.to", "result_stat": {"min": -0.3121015429496765, "max": 0.36434608697891235, "mean": 0.0617644302546978, "std": 0.33854630301859995, "shape": [2, 2]}, "time": 1725751387.744574}
}
```