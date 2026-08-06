# Security finding #866: GetGatherAxis (gather_fusion.cc:13-24) reads the attacker-supplied …

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/66**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read |
| Location | [`targets/onnxruntime/onnxruntime/core/optimizer/gather_fusion.cc:200`](https://github.com/microsoft/onnxruntime/blob/baa0d32bbdf7557772e0678e146de9e0b81c00cc/onnxruntime/core/optimizer/gather_fusion.cc#L200-L203) — `GatherSliceToSplitFusion::ApplyImpl()` |
| Trust boundary | Model bytes supplied via OrtApi CreateSessionFromArray / CompileModel, parsed into a Graph and run through the L1 optimizer; the Gather 'axis' node-attribute (and Slice 'axes' initializer) are attacker-controlled. |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/66 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #866.