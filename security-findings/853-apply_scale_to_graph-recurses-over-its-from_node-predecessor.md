# Security finding #853: apply_scale_to_graph recurses over its from_node predecessors for A…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/54**

| | |
| --- | --- |
| CWE | CWE-674: Uncontrolled Recursion (stack exhaustion) |
| Location | `targets/onnxruntime/onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:87` — `GraphNode::apply_scale_to_graph()` |
| Trust boundary | Attacker-supplied ONNX QDQ model graph structure (from_node edges) entering the OpenVINO EP scale-fix transform via Transform()->scale_graph()->apply_scale_to_graph(). |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/54 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #853.