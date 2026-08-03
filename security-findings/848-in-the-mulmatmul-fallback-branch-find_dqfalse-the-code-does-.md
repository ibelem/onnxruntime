# Security finding #848: In the Mul/MatMul fallback branch (find_dq==false) the code does `f…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/46**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read (empty std::vector .back()/operator[]) |
| Location | `targets/onnxruntime/onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:118` — `GraphNode::apply_scale_to_graph()` |
| Trust boundary | attacker-shaped ONNX QDQ graph loaded via ORT/WebNN; CustomGraph nodes/edges built in generate_graph_from_onnx() from GetInputs/GetOutputs and input/output name matching |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/46 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #848.